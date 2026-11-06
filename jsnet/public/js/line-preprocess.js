// Строчный препроцессинг: холст со штрихами -> бинарная маска -> нарезка на
// строки -> тензоры [64 x W] для CTC-модели. Порт split_lines/_split_waist/
// crop_margins из test-py/common.py (пороги и эвристики те же, что в сервисе
// и в обучении: оценкаCer совпадает с демо).
import { drawStroke } from "./pen.js";

export const LINE_H = 64;      // каноническая высота строки (как H в модели)
export const LINE_W_MAX = 2048;

// Пороги test-py/common.py заданы для ширины канваса CANVAS_REF_WIDTH,
// при других ширинах масштабируются (dpr 1x/2x на планшете).
const REF_WIDTH = 1600;
const LINE_MIN_GAP = 3;
const LINE_MIN_HEIGHT = 10;
const LINE_FRAGMENT_RATIO = 0.3;
const WAIST_MIN_RUN = 80;
const WAIST_MIN_BODY = 25;
// CROP_PAD = 4 как в ml/line_data.py (обучающий препроцессинг), а не 16 как
// в test-py/common.py: модель обучена на плотных строках, поля должны совпадать.
const CROP_PAD = 4;
export const INK_THRESHOLD = 200;

// Canvas for raster: OffscreenCanvas — rasterize is now called from Worker,
// where document does not exist; fallback to ordinary canvas for main thread (legacy path).
const mkCanvas = (w, h) =>
  typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : document.createElement("canvas");

/**
 * Растровизует штрихи (координаты CSS px) на offscreen-канвас размера w x h.
 * Возвращает {canvas, ink: Uint8Array(w*h), w, h} или null, если чернил нет.
 * drawStroke из pen.js — тот же рендер, что на видимом холсте («что видно,
 * то и распознаётся»), клетка-разметка (#d8e2ed) в маску не попадает.
 */
export function rasterize(strokes, settings, w, h, dpr = 1) {
  const c = mkCanvas(w, h);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // штрихи в CSS px, растер в device px
  for (const pts of strokes) drawStroke(ctx, settings.pen || {}, pts);
  const d = ctx.getImageData(0, 0, w, h).data;
  const ink = new Uint8Array(w * h);
  let any = false;
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    const g = (d[i] + d[i + 1] + d[i + 2]) / 3;
    if (g < INK_THRESHOLD) {
      ink[p] = 1;
      any = true;
    }
  }
  return any ? { canvas: c, ink, w, h } : null;
}

/** Строки маски: список [y0, y1). Порт split_lines. */
export function splitLines(ink, w, h) {
  const rowHas = new Uint8Array(h);
  for (let y = 0; y < h; y++) {
    let any = 0;
    const base = y * w;
    for (let x = 0; x < w; x++) {
      if (ink[base + x]) {
        any = 1;
        break;
      }
    }
    rowHas[y] = any;
  }
  if (!rowHas.some((v) => v)) return [];
  const scale = Math.max(0.5, w / REF_WIDTH);
  const minGap = Math.max(1, Math.round(LINE_MIN_GAP * scale));
  const minHeight = Math.max(1, Math.round(LINE_MIN_HEIGHT * scale));

  let lines = [];
  let y0 = null;
  let gap = 0;
  for (let y = 0; y < h; y++) {
    if (rowHas[y]) {
      if (y0 === null) y0 = y;
      gap = 0;
    } else if (y0 !== null) {
      gap += 1;
      if (gap >= minGap) {
        lines.push([y0, y - gap + 1]);
        y0 = null;
        gap = 0;
      }
    }
  }
  if (y0 !== null) lines.push([y0, h]);

  // Разрез по «талии» (перенесённое слово мостит провал между строками).
  let expanded = [];
  for (const [a, b] of lines) expanded = expanded.concat(splitWaist(ink, w, a, b, scale));
  lines = expanded;

  // Слияние огрызков и «вершинок» (тонкая/узкая полоса — часть соседней).
  if (lines.length > 1) {
    const widths = lines.map(([a, b]) => {
      let cnt = 0;
      for (let x = 0; x < w; x++) {
        for (let y = a; y < b; y++) {
          if (ink[y * w + x]) {
            cnt++;
            break;
          }
        }
      }
      return cnt;
    });
    const maxW = Math.max(...widths);
    const merged = lines.map(() => true);
    for (let i = 0; i < lines.length; i++) {
      const [a, b] = lines[i];
      const height = b - a;
      const thin = height < minHeight;
      const narrow = widths[i] < maxW * LINE_FRAGMENT_RATIO;
      if (!(thin || narrow)) continue;
      const gaps = [];
      for (const j of [i - 1, i + 1]) {
        if (j >= 0 && j < lines.length && merged[j]) {
          gaps.push([j < i ? a - lines[j][1] : lines[j][0] - b, j]);
        }
      }
      if (!gaps.length) continue;
      gaps.sort((p, q) => p[0] - q[0]);
      const [, j] = gaps[0];
      if (thin || gaps[0][0] < height) {
        merged[i] = false;
        lines[j] = [Math.min(lines[j][0], a), Math.max(lines[j][1], b)];
      }
    }
    lines = lines.filter((_, i) => merged[i]);
  }
  return lines.filter(([a, b]) => {
    for (let i = a * w; i < b * w; i++) if (ink[i]) return true;
    return false;
  });
}

/** Порт _split_waist: узкий провал высоты между широкими областями. */
function splitWaist(ink, w, y0, y1, scale) {
  const widths = [];
  let maxW = 0;
  for (let y = y0; y < y1; y++) {
    let cnt = 0;
    const base = y * w;
    for (let x = 0; x < w; x++) if (ink[base + x]) cnt++;
    widths.push(cnt);
    if (cnt > maxW) maxW = cnt;
  }
  if (maxW <= 0) return [[y0, y1]];
  const limit = maxW * LINE_FRAGMENT_RATIO;
  const minRun = Math.max(1, Math.round(WAIST_MIN_RUN * scale));
  const minBody = Math.max(1, Math.round(WAIST_MIN_BODY * scale));
  const splits = [];
  let y = 0;
  while (y < widths.length) {
    if (widths[y] >= limit) {
      y++;
      continue;
    }
    let y2 = y;
    while (y2 < widths.length && widths[y2] < limit) y2++;
    const runLen = y2 - y;
    if (runLen >= minRun) {
      let above = 0;
      let below = 0;
      for (let i = 0; i < y; i++) if (widths[i] >= limit) above++;
      for (let i = y2; i < widths.length; i++) if (widths[i] >= limit) below++;
      if (above >= minBody && below >= minBody) splits.push(y + (runLen >> 1));
    }
    y = y2;
  }
  if (!splits.length) return [[y0, y1]];
  const parts = [];
  let prev = y0;
  for (const s of splits) {
    parts.push([prev, y0 + s]);
    prev = y0 + s;
  }
  parts.push([prev, y1]);
  return parts;
}

/**
 * Полоса [y0,y1) -> Float32Array длины LINE_H*w' (чернила 0..1, row-major)
 * и w'. Порт crop_margins (поля по маске полосы + CROP_PAD) и resize в H=64
 * с сохранением пропорций. null, если полоса слишком узкая/высокая.
 */
export function stripToTensor(src, ink, w, y0, y1) {
  const pad = CROP_PAD;
  let x0 = w;
  let x1 = -1;
  let t0 = y1;
  let t1 = y0;
  for (let y = y0; y < y1; y++) {
    const base = y * w;
    for (let x = 0; x < w; x++) {
      if (ink[base + x]) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < t0) t0 = y;
        if (y > t1) t1 = y;
      }
    }
  }
  if (x1 < x0) return null;
  x0 = Math.max(0, x0 - pad);
  x1 = Math.min(w - 1, x1 + pad);
  t0 = Math.max(0, t0 - pad);
  t1 = Math.min(y1 - 1, t1 + pad);
  const cw = x1 - x0 + 1;
  const ch = t1 - t0 + 1;
  let dw = Math.max(16, Math.min(LINE_W_MAX, Math.round((cw * LINE_H) / ch)));
  const hc = mkCanvas(dw, LINE_H);
  const ctx = hc.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, dw, LINE_H);
  ctx.drawImage(src, x0, t0, cw, ch, 0, 0, dw, LINE_H);
  const d = ctx.getImageData(0, 0, dw, LINE_H).data;
  const out = new Float32Array(dw * LINE_H);
  for (let i = 0, p = 0; i < d.length; i += 4, p++) {
    out[p] = (255 - (d[i] + d[i + 1] + d[i + 2]) / 3) / 255;
  }
  return { data: out, w: dw };
}

/* Компактная передача штрихов в воркер: вместо структурированного клонирования
 * тысяч объектов {x,y,...} (тормозило main-поток на длинных сессиях — ровно те
 * «пробросы» при распознавании) — один Float32Array + Int32Array длин штрихов,
 * буферы передаются transfer-ом без копирования вовсе. Поля ровно те, что нужны
 * растеру (pen.js); точность float32 (~0.0002 px при координатах холста) на
 * растеризацию не влияет — тест pack/unpack в tests/decoding.mjs. */
export const PT_STRIDE = 5;

export function packStrokes(strokes) {
  let n = 0;
  for (const s of strokes) n += s.length;
  const pts = new Float32Array(n * PT_STRIDE);
  const counts = new Int32Array(strokes.length);
  let k = 0;
  for (let i = 0; i < strokes.length; i++) {
    const s = strokes[i];
    counts[i] = s.length;
    for (const p of s) {
      pts[k++] = p.x;
      pts[k++] = p.y;
      pts[k++] = p.pressure || 0;
      pts[k++] = p.tiltX || 0;
      pts[k++] = p.tiltY || 0;
    }
  }
  return { pts, counts };
}

export function unpackStrokes(pts, counts) {
  const strokes = new Array(counts.length);
  let k = 0;
  for (let i = 0; i < counts.length; i++) {
    const s = new Array(counts[i]);
    for (let j = 0; j < counts[i]; j++, k += PT_STRIDE)
      s[j] = { x: pts[k], y: pts[k + 1], pressure: pts[k + 2], tiltX: pts[k + 3], tiltY: pts[k + 4] };
    strokes[i] = s;
  }
  return strokes;
}

/**
 * Incremental preprocessing of strokes: strips (lines -> tensors) without full canvas re-rasterize
 * on every pen-off. Inside, keeps ink-mask (OffscreenCanvas): on each pen-off only
 * new strokes get drawn, and getImageData reads only their bbox (in grow). Full canvas
 * re-rasterize (reset) only when pen settings change or canvas size (dpr) changes.
 * If strokes got fewer (undo), mask is no longer correct — full re-rasterize.
 */
export class StripPrep {
  constructor() { this.mask = null; }

  // gen — «поколение» штрихов (см. lines.js): инкрементальный grow предполагает,
  // что список штрихов растёт только хвостом; CLEAR обнуляет список, и без смены
  // gen новые слова дорисовывались бы поверх стёртых (или стёртое читалось вместо
  // нового при равном числе штрихов). gen в ключе — любое повреждение истории
  // (clear и будущие массовые удаления) заставляет перестроить маску с нуля.
  prep(strokes, pen, w, h, dpr, gen = 0) {
    if (!strokes || !strokes.length) return [];
    const key = JSON.stringify([gen, w, h, Math.round(dpr * 1000), pen]);
    const m = this.mask;
    if (!(m && m.key === key && m.count <= strokes.length)) this.reset(strokes, pen, w, h, dpr, gen);
    else if (strokes.length > m.count) this.grow(strokes, pen);
    if (!this.mask) return [];
    const mm = this.mask;
    const bands = splitLines(mm.ink, w, h);
    const strips = [];
    for (const [y0, y1] of bands) {
      const s = stripToTensor(mm.canvas, mm.ink, w, y0, y1);
      if (s && s.w >= 16) strips.push(s);
    }
    return strips;
  }

  // whole-ink QC strip for profile/collect — dpr темя 1, same as before
  sample(strokes, pen, w, h) {
    if (!strokes || !strokes.length) return null;
    const r = rasterize(strokes, { pen }, w, h, 1);
    if (!r) return null;
    let y0 = r.h, y1 = -1;
    for (let y = 0; y < r.h; y++)
      for (let x = 0; x < r.w; x++)
        if (r.ink[y * r.w + x]) { if (y < y0) y0 = y; if (y > y1) y1 = y; }
    if (y1 < y0) return null;
    return stripToTensor(r.canvas, r.ink, r.w, y0, y1 + 1);
  }

  reset(strokes, pen, w, h, dpr, gen = 0) {
    const ras = rasterize(strokes, { pen }, w, h, dpr);
    this.mask = ras
      ? {
          key: JSON.stringify([gen, w, h, Math.round(dpr * 1000), pen]),
          canvas: ras.canvas, ctx: ras.canvas.getContext("2d"),
          ink: ras.ink, count: strokes.length, w, h, dpr,
        }
      : null;
  }

  grow(strokes, pen) {
    const m = this.mask;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = m.count; i < strokes.length; i++) {
      for (const pt of strokes[i]) {
        if (pt.x < x0) x0 = pt.x;
        if (pt.x > x1) x1 = pt.x;
        if (pt.y < y0) y0 = pt.y;
        if (pt.y > y1) y1 = pt.y;
      }
    }
    for (let i = m.count; i < strokes.length; i++) drawStroke(m.ctx, pen, strokes[i]);
    m.count = strokes.length;
    if (x1 < x0) return;
    // запас point bbox: radius (R + pscale) + halo (shadowBlur), in CSS px (then multiply by dpr)
    const reach = 2 * ((pen.R || 0) + (pen.pscale || 0)) + 8;
    const rx0 = Math.max(0, Math.floor((x0 - reach) * m.dpr));
    const ry0 = Math.max(0, Math.floor((y0 - reach) * m.dpr));
    const rx1 = Math.min(m.w, Math.ceil((x1 + reach) * m.dpr));
    const ry1 = Math.min(m.h, Math.ceil((y1 + reach) * m.dpr));
    if (rx1 <= rx0 || ry1 <= ry0) return;
    const rw = rx1 - rx0;
    const d = m.ctx.getImageData(rx0, ry0, rw, ry1 - ry0).data;
    for (let y = ry0; y < ry1; y++) {
      const src = (y - ry0) * rw * 4;
      let p = y * m.w + rx0;
      for (let x = 0; x < rw; x++, p++) {
        const i = src + x * 4;
        if (d[i] + d[i + 1] + d[i + 2] < INK_THRESHOLD * 3) m.ink[p] = 1;
      }
    }
  }
}
