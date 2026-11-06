// Настройки и движок пера: порт движка из test-py/static/brush.js.
// Штрих строится «пятнами»-эллипсами: радиус (R), фокусы (focal), угол,
// давление (pressure/pscale), наклон (tilt), сглаживание и жёсткость краёв.
// Хранение настроек — в settings.js; здесь только нормализация и рендер.

export const PEN_DEFAULTS = {
  R: 1,             // толщина (большая полуось пятна), px
  focal: 0,         // расстояние между фокусами 2c, px (0 = круг)
  angle: 0,         // угол наклона пятна, градусы
  pressure: true,   // учитывать event.pressure
  pscale: 0.6,      // прибавка радиуса при полном давлении, px
  tilt: false,      // учитывать event.tiltX / event.tiltY
  smoothing: 30,    // сглаживание координат, %
  hardness: 100,    // жёсткость краёв пятна, %
  color: "#1a1a2e",
};

// Диапазоны: заодно источник min/max/step для слайдеров.
export const PEN_LIMITS = {
  R:         { min: 0.1, max: 3, step: 0.1 },
  focal:     { min: 0, max: 3, step: 0.1 },
  angle:     { min: 0, max: 180, step: 1 },
  pscale:    { min: 0, max: 3, step: 0.1 },
  smoothing: { min: 0, max: 100, step: 1 },
  hardness:  { min: 0, max: 100, step: 1 },
};

export const PEN_COLORS = [
  { hex: "#1a1a2e", title: "Чёрный" },
  { hex: "#0044cc", title: "Синий" },
  { hex: "#cc2200", title: "Красный" },
  { hex: "#228833", title: "Зелёный" },
];

// Пресеты — как в модалке «Настройки кисти» из test-py.
export const PEN_PRESETS = [
  { name: "Гелевое перо", patch: { R: 0.4, focal: 0, angle: 0, hardness: 100, smoothing: 20, pressure: true, pscale: 1.2, tilt: false } },
  { name: "Капилляр",     patch: { R: 0.8, focal: 0, angle: 0, hardness: 90, smoothing: 30, pressure: true, pscale: 1.6, tilt: false } },
  { name: "Маркер",       patch: { R: 1.8, focal: 0, angle: 0, hardness: 25, smoothing: 45, pressure: false, pscale: 0, tilt: false } },
  { name: "Кисть",        patch: { R: 2.8, focal: 2, angle: 60, hardness: 45, smoothing: 55, pressure: true, pscale: 3, tilt: true } },
];

// Подписи и единицы — общий источник для панелей настроек.
export const PEN_META = {
  R: "Толщина", focal: "Вытянутость", angle: "Угол пера", pressure: "Давление пера",
  tilt: "Наклон пера", pscale: "Выразительность", smoothing: "Сглаживание", hardness: "Жёсткость",
};
export const PEN_UNITS = { angle: "°", smoothing: "%", hardness: "%" };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
// Двоичные дроби ломают сравнение значений слайдеров с пресетами — округляем до шага.
const quantize = (v, step) => Math.round(Math.round(v / step) * step * 1e6) / 1e6;

export function normalizePen(raw) {
  const src = raw && typeof raw === "object" ? raw : {};
  const out = {};
  for (const [key, lim] of Object.entries(PEN_LIMITS)) {
    const num = parseFloat(src[key]);
    const base = Number.isFinite(num) ? num : PEN_DEFAULTS[key];
    out[key] = quantize(clamp(base, lim.min, lim.max), lim.step);
  }
  out.pressure = typeof src.pressure === "boolean" ? src.pressure : PEN_DEFAULTS.pressure;
  out.tilt = typeof src.tilt === "boolean" ? src.tilt : PEN_DEFAULTS.tilt;
  const color = String(src.color || "");
  out.color = /^#[0-9a-fA-F]{6}$/.test(color) ? color.toLowerCase() : PEN_DEFAULTS.color;
  // Фокус не может превышать большой радиус эллипса: 2c ≤ 2R.
  out.focal = clamp(out.focal, 0, 2 * out.R);
  return out;
}

export const penRgba = (hex, a) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};

/**
 * Фон холста «как в тетради»: белая бумага + клетка (из brush.js, test-py).
 * Клетка существенно влияет на почерк (размер и базовая линия букв),
 * поэтому рисуется прямо на canvas — и попадает в калибровочные эталоны визуально.
 */
export const GRID = 20;
export function fillPaper(ctx, cssW, cssH, withGrid) {
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, cssW, cssH);
  if (withGrid) {
    ctx.strokeStyle = "#d8e2ed";
    ctx.lineWidth = 1;
    for (let x = GRID; x <= cssW; x += GRID) {
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 0);
      ctx.lineTo(x + 0.5, cssH);
      ctx.stroke();
    }
    for (let y = GRID; y <= cssH; y += GRID) {
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(cssW, y + 0.5);
      ctx.stroke();
    }
  }
  ctx.restore();
}

// Одно пятно: эллипс (rx, ry) под углом rot с центром (x, y).
function stamp(ctx, pen, x, y, rx, ry, rot) {
  rx = Math.max(rx, 0.1);
  ry = Math.max(ry, 0.1);
  const soft = (100 - pen.hardness) / 100;
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rot);
  if (soft > 0) {
    // Мягкое гало: прозрачная заливка + тень → размытый край (shadowBlur).
    ctx.shadowColor = pen.color;
    ctx.shadowBlur = soft * Math.max(rx, ry);
    ctx.fillStyle = penRgba(pen.color, 0);
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
  }
  // Ядро: при hardness=100 заполняет весь эллипс (чёткий край),
  // при малой жёсткости — только центр, остальное даёт гало.
  const core = 0.35 + 0.65 * (pen.hardness / 100);
  ctx.shadowBlur = 0;
  ctx.fillStyle = penRgba(pen.color, 1);
  ctx.beginPath();
  ctx.ellipse(0, 0, rx * core, ry * core, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Параметры эллипса из точки штриха: давление масштабирует R, наклон
// раскрывает 2c (пятно сплющивается) и доворачивает угол по азимуту tilt.
function penParams(pen, pt) {
  let R = pen.R;
  if (pen.pressure) R += (pt.pressure || 0) * pen.pscale;
  const halfC = Math.min(pen.focal, 2 * R) / 2;
  let c = halfC;
  let rot = (pen.angle * Math.PI) / 180;
  if (pen.tilt) {
    const tx = pt.tiltX || 0, ty = pt.tiltY || 0;
    const mag = Math.min(1, Math.hypot(tx, ty) / 90); // 0 — стилус вертикально
    if (mag > 0.02) {
      c = Math.min(c + mag * (R - 0.1), R - 0.1);     // наклон → к «линии»
      rot += Math.atan2(tx, -ty);                     // азимут наклона
    }
  }
  // radiusY = sqrt(R² − c²); мнимый/нулевой → 0.1 (тонкая линия).
  const rySq = R * R - c * c;
  return { rx: R, ry: rySq > 0.01 ? Math.sqrt(rySq) : 0.1, rot };
}

// Дооборот до угла кратчайшим направлением (иначе 350°→10° разворачивает через 180°).
function lerpAngle(u, v, t) {
  let d = v - u;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return u + d * t;
}

/**
 * Сегмент штриха (a, b]: пятна между точками с параметрами {x,y,rx,ry,rot}.
 * Пятно в a считается уже нарисованным — chained-вызовы дают побитово тот же
 * растр, что drawStroke по всему штриху (инкрементальная отрисовка в capture.js:
 * на каждое pointermove рисуем только новый кусок, а не весь штрих).
 */
export function drawSegment(ctx, pen, a, b) {
  const dist = Math.hypot(b.x - a.x, b.y - a.y);
  // Интерполяция: пятна вдоль отрезка с шагом ≤ 0.4·малой полуоси;
  // низкий пол (0.2 px) держит сплошной линию и при минимальной толщине 0.1.
  const step = Math.max(0.2, Math.min(a.ry, b.ry) * 0.4);
  const n = Math.max(1, Math.ceil(dist / step));
  for (let k = 1; k <= n; k++) {
    const t = k / n;
    stamp(
      ctx, pen,
      a.x + (b.x - a.x) * t,
      a.y + (b.y - a.y) * t,
      a.rx + (b.rx - a.rx) * t,
      a.ry + (b.ry - a.ry) * t,
      lerpAngle(a.rot, b.rot, t),
    );
  }
}

/** Точка штриха с параметрами пятна — для инкрементальной отрисовки. */
export function inkPoint(pen, pt) {
  return { x: pt.x, y: pt.y, ...penParams(pen, pt) };
}

/**
 * Перерисовывает штрих (массив точек {x,y,pressure,tiltX,tiltY}) пятна-эллипсами.
 * Точки уже сглажены на этапе сбора (capture.js) — что видим, то и распознаётся.
 */
export function drawStroke(ctx, pen, pts) {
  if (!pts.length) return;
  let last = inkPoint(pen, pts[0]);
  stamp(ctx, pen, last.x, last.y, last.rx, last.ry, last.rot);
  for (let i = 1; i < pts.length; i++) {
    const b = inkPoint(pen, pts[i]);
    drawSegment(ctx, pen, last, b);
    last = b;
  }
}
