# Растеризация пера и нарезка строки в тензор — точный порт src/pen.js и
# src/line-preprocess.js из web_onnx_v2 (sample-путь: dpr=1, крой по маске +
# CROP_PAD=4, resize в H=64, инк 0..1 row-major). Обучающий препроцессинг
# обязан совпадать с инференсным — иначе дообученная модель «поплывёт».
#
# vendor-копия (inknet-online): порт растра и LineNet из inknet-train; правки проект: port растра и LineNet лежат в
# vendor/ (копии из inknet-train/inknet; правки клиента web_onnx_v2 обязаны
# синхронизироваться в vendor вручную — это порт, не зависимость).
#
# Ограничение порта: мягкое гало пера (hardness < 100, canvas shadowBlur) не
# воспроизводится побитово — рендерится только ядро эллипса. Дефолтное перо
# (PEN_DEFAULTS, hardness=100) и все синтетические настройки обучения — с
# hardness=100, для них порт точен с учётом AA (supersample + box-фильтр).
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

LINE_H = 64          # каноническая высота строки (LINE_H в line-preprocess.js)
LINE_W_MAX = 2048    # LINE_W_MAX
INK_THRESHOLD = 200  # g < INK_THRESHOLD -> пиксель «чернила»
CROP_PAD = 4         # CROP_PAD как в ml/line_data.py / line-preprocess.js
SS = 4               # supersample: AA канваса ≈ box-усреднение маски SS x SS

ELLIPSE_SEGMENTS = 48  # хорда эллипса при SS=4 даёт погрешность < 0.02 px


@dataclass(frozen=True)
class Pen:
    """PEN_DEFAULTS из pen.js (pressure/tilt есть, но данных-то без них)."""
    R: float = 1.0
    focal: float = 0.0
    angle: float = 0.0      # градусы
    pressure: bool = True
    pscale: float = 0.6
    tilt: bool = False
    hardness: float = 100.0
    color: str = "#1a1a2e"


def color_gray(hex_color: str) -> float:
    """Средняя яркость цвета пера (канвас композитит solid поверх белого)."""
    n = int(hex_color[1:], 16)
    return (((n >> 16) & 255) + ((n >> 8) & 255) + (n & 255)) / 3.0


def ink_point(pt: dict, pen: Pen) -> tuple[float, float, float, float, float]:
    """inkPoint из pen.js: точка штриха -> {x, y, rx, ry, rot}.

    Точки из collect имеют только {x, y, t}: pressure отсутствует -> 0,
    при pen.pressure R не растёт (0 * pscale). tilt-параметры отсутствуют аналогично.
    """
    x, y = float(pt["x"]), float(pt["y"])
    r = pen.R
    if pen.pressure:
        r += (pt.get("pressure") or 0.0) * pen.pscale
    c = min(pen.focal, 2.0 * r) / 2.0
    rot = math.radians(pen.angle)
    if pen.tilt:
        tx, ty = float(pt.get("tiltX") or 0.0), float(pt.get("tiltY") or 0.0)
        mag = min(1.0, math.hypot(tx, ty) / 90.0)
        if mag > 0.02:
            c = min(c + mag * (r - 0.1), r - 0.1)
            rot += math.atan2(tx, -ty)
    ry_sq = r * r - c * c
    ry = math.sqrt(ry_sq) if ry_sq > 0.01 else 0.1
    return x, y, r, ry, rot


def _lerp_angle(u: float, v: float, t: float) -> float:
    """lerpAngle из pen.js: доводка до угла кратчайшим направлением."""
    d = v - u
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return u + d * t


def _stroke_stamps(stroke: list[dict], pen: Pen):
    """Генератор пятен штриха — порт drawStroke + drawSegment.

    Пятно: (x, y, rx*core, ry*core, rot); core = 0.35 + 0.65*hardness/100.
    Мягкое гало (shadowBlur при hardness < 100) не портится — только ядро.
    """
    if not stroke:
        return
    if pen.hardness < 100.0:
        raise NotImplementedError(
            "гало пера (hardness < 100) в Python-порту не воспроизводится; "
            "для обучения используйте hardness=100 (PEN_DEFAULTS)"
        )
    core = 0.35 + 0.65 * (pen.hardness / 100.0)  # = 1.0 при hardness=100
    last = ink_point(stroke[0], pen)
    yield last[0], last[1], max(last[2], 0.1) * core, max(last[3], 0.1) * core, last[4]
    for pt in stroke[1:]:
        b = ink_point(pt, pen)
        dist = math.hypot(b[0] - last[0], b[1] - last[1])
        step = max(0.2, min(last[3], b[3]) * 0.4)
        n = max(1, math.ceil(dist / step))
        for k in range(1, n + 1):
            t = k / n
            yield (
                last[0] + (b[0] - last[0]) * t,
                last[1] + (b[1] - last[1]) * t,
                last[2] + (b[2] - last[2]) * t,
                last[3] + (b[3] - last[3]) * t,
                _lerp_angle(last[4], b[4], t),
            )
        last = b


def _ellipse_poly(cx, cy, rx, ry, rot) -> list[tuple[float, float]]:
    """Полигон повёрнутого эллипса (для solid-фолла PIL на SS-растре)."""
    pts = []
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    for k in range(ELLIPSE_SEGMENTS):
        a = 2.0 * math.pi * k / ELLIPSE_SEGMENTS
        px, py = rx * math.cos(a), ry * math.sin(a)
        pts.append((cx + px * cos_r - py * sin_r, cy + px * sin_r + py * cos_r))
    return pts


def rasterize_strokes(strokes: list[list[dict]], pen: Pen, margin: float = 12.0):
    """Штрихи -> растр покрытия чернил. Порт rasterize (dpr=1).

    Не нужно хранить весь холст страницы: перед resize в 64 строка кроется до
    bbox маски, поэтому растеризуем только bbox штрихов + margin.
    Возвращает (cov float32 [h, w], ox, oy) — cov 0..1, ox/oy смещение региона
    в координатах штрихов.
    """
    stamps = []
    min_x = min_y = math.inf
    max_x = max_y = -math.inf
    for stroke in strokes:
        for s in _stroke_stamps(stroke, pen):
            stamps.append(s)
            # половина протяжённости повёрнутого эллипса по осям
            ex = math.hypot(s[2] * math.cos(s[4]), s[3] * math.sin(s[4]))
            ey = math.hypot(s[2] * math.sin(s[4]), s[3] * math.cos(s[4]))
            min_x = min(min_x, s[0] - ex)
            max_x = max(max_x, s[0] + ex)
            min_y = min(min_y, s[1] - ey)
            max_y = max(max_y, s[1] + ey)
    if not stamps:
        return None
    x0 = math.floor(min_x - margin)
    y0 = math.floor(min_y - margin)
    w = int(math.ceil(max_x - x0) + 1 + margin)  # запас: margin уже учтён, +1 на округление
    h = int(math.ceil(max_y - y0) + 1)
    big = Image.new("L", (w * SS, h * SS), 0)
    dr = ImageDraw.Draw(big)
    for sx, sy, rx, ry, rot in stamps:
        poly = _ellipse_poly((sx - x0) * SS, (sy - y0) * SS, rx * SS, ry * SS, rot)
        dr.polygon(poly, fill=255)
    small = big.resize((w, h), Image.BOX)  # усреднение ~ AA-покрытие канваса
    cov = np.asarray(small, dtype=np.float32) / 255.0
    return cov, x0, y0


def strip_to_tensor(cov: np.ndarray, pen: Pen):
    """Растр покрытия -> (float32 [64, dw], dw). Порт stripToTensor + sample().

    Маска (g < INK_THRESHOLD) -> bbox + CROP_PAD -> crop -> resize в H=64 с
    сохранением пропорций (dw clamp 16..2048) -> инк (255 - g) / 255 row-major.
    """
    col_g = color_gray(pen.color)
    # канвас хранит композит целыми 8-бит пикселями — округляем так же
    g = np.clip(np.rint(255.0 - cov * (255.0 - col_g)), 0, 255).astype(np.uint8)
    mask = g < INK_THRESHOLD
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    h, w = g.shape
    x_lo = max(int(xs.min()) - CROP_PAD, 0)
    x_hi = min(int(xs.max()) + CROP_PAD, w - 1)
    y_lo = max(int(ys.min()) - CROP_PAD, 0)
    y_hi = min(int(ys.max()) + CROP_PAD, h - 1)
    cw = x_hi - x_lo + 1
    ch = y_hi - y_lo + 1
    # Math.round JS округляет .5 вверх — int(x + 0.5), не banker's round
    dw = max(16, min(LINE_W_MAX, int((cw * LINE_H) / ch + 0.5)))
    img = Image.fromarray(g[y_lo : y_hi + 1, x_lo : x_hi + 1])
    img = img.resize((dw, LINE_H), Image.BILINEAR)  # drawImage smoothing при увеличении
    ink = (255.0 - np.asarray(img, dtype=np.float32)) / 255.0
    return ink, dw
