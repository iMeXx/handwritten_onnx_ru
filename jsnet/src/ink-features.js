// Векторная репрезентация штриха: «поле векторов вдоль траектории» вместо растра.
//
// Идея (первая): не растеризовать перо в картинку, а кормить сеть локальными
// векторами, снятыми вдоль самой траектории. Каждый отсечённый по длине дуги
// точка несёт свой репер (касательная T, нормаль N), и в этом репере выражены
// векторы-хорды вперёд/назад — это «параллельно переносимый репер» вдоль кривой.
// Такое описание инвариантно к повороту и сдвигу по построению и (после нормы
// по размеру глифа) к масштабу: сеть видит ФОРМУ штриха, а не его положение на
// холсте. К геометрии добавлена динамика: угол поворота касательной (кривизна),
// нормированная скорость (ds/dt из реального t), нажим, наклон пера и граница
// «перо поднято» между штрихами.
//
// Модуль чистый: без DOM/OffscreenCanvas — работает и в воркере, и в node
// (тесты). Вход — штрихи как в capture/packStrokes; выход — компактная матрица
// признаков (Float32Array + counts), которую можно передать transfer-ом, ровно
// как packStrokes передаёт растру координаты.

// Шаг пересэмплирования ds задаём так, чтобы каждый штрих давал примерно
// PER_STROKE_POINTS точек. Так плотность инвариантна к повороту/сдвигу/масштабу
// (ds выведен из длины чернил — тоже инварианта) и адекватна и коротким, и
// длинным глифам (число точек растёт с числом/длиной штрихов, а не фиксировано).
const PER_STROKE_POINTS = 16;
const DS_MIN = 1.0;    // не дробим короткий штрих слишком сильно
export const STRIDE = 10; // признаков на точку (см. индексы ниже)

// Индексы признака в векторе длины STRIDE.
export const VF = {
  FX: 0, FY: 1,   // хорда вперёд (i -> i+k) в репере (T,N), нормирована по размеру глифа
  BX: 2, BY: 3,   // хорда назад (i-k -> i) в репере
  DTH: 4,         // подписанный угол поворота касательной в точке / π  (≈ кривизна)
  SPEED: 5,       // нормированный логарифм скорости ds/dt
  PRESS: 6,       // нажим 0..1 (как пришёл)
  TILT_X: 7,      // tiltX/90
  TILT_Y: 8,      // tiltY/90
  PENUP: 9,       // 1 у первой точки штриха (перо было поднято) — видно «лифты»
};

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

function hypot(dx, dy) { return Math.sqrt(dx * dx + dy * dy); }

// Кумулятивная длина дуги штриха: [0, s1, s2, ...].
function arcLengths(pts) {
  const s = new Array(pts.length);
  s[0] = 0;
  for (let i = 1; i < pts.length; i++)
    s[i] = s[i - 1] + hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y);
  return s;
}

/**
 * Пересэмплировать штрих к РАВНЫМ шагам длины дуги ds. Координаты, t, нажим и
 * tilt интерполируются линейно вдоль исходных сегментов — значит «dt» здесь
 * равномерный по геометрии, а скорость считаем из настоящего времени t (см. ниже).
 * Возвращает новый массив точек (>=2, либо 1 для точки/нулевого штриха).
 */
export function resampleStroke(pts, ds) {
  if (!pts || pts.length === 0) return [];
  if (pts.length === 1) return [{ ...pts[0] }];
  const s = arcLengths(pts);
  const L = s[s.length - 1];
  if (L < 1e-6) return [{ ...pts[0] }]; // точка-клякса
  const n = Math.max(1, Math.round(L / ds)); // сегментов; точек = n+1
  const out = new Array(n + 1);
  let seg = 1;
  for (let j = 0; j <= n; j++) {
    const target = (L * j) / n;
    while (seg < pts.length - 1 && s[seg] < target) seg++;
    const s0 = s[seg - 1], s1 = s[seg];
    const f = s1 - s0 > 1e-9 ? (target - s0) / (s1 - s0) : 0;
    const A = pts[seg - 1], B = pts[seg];
    const num = (a, b) => (a == null ? b ?? 0 : b == null ? a : a + (b - a) * f);
    out[j] = {
      x: A.x + (B.x - A.x) * f,
      y: A.y + (B.y - A.y) * f,
      t: num(A.t, B.t),
      pressure: num(A.pressure ?? 0, B.pressure ?? 0),
      tiltX: num(A.tiltX ?? 0, B.tiltX ?? 0),
      tiltY: num(A.tiltY ?? 0, B.tiltY ?? 0),
    };
  }
  return out;
}

// «Масштаб глифа» для нормировки. Берём СУММУ длин дуг штрихов (чернила), а не
// диагональ bbox: ось-выровненный bbox РАЗНОНАПРАВЛЕН с поворотом (повёрнутый
// глиф даёт другую bbox-диагональ -> другой шаг ds -> другой набор точек ->
// признаки плывут). Длина чернил инвариантна к повороту/сдвигу и линейна по
// размеру (→ инвариантна и к масштабу), и устойчива к плотности сэмплирования.
// Возвращает длину чернил (px); 0 — если все штрихи вырождены (одни точки).
function glyphScale(strokes) {
  let len = 0;
  for (const st of strokes)
    if (st.length > 1) { const s = arcLengths(st); len += s[s.length - 1]; }
  return len;
}

// Медиана массива (для устойчивой нормировки скорости).
function median(arr) {
  if (!arr.length) return 0;
  const a = [...arr].sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/**
 * Собрать матрицу признаков «поля векторов» для набора штрихов одного глифа.
 * strokes: массив штрихов; штрих — массив точек {x,y,t?,pressure?,tiltX?,tiltY?}
 *   (координаты — любая общая система; поворот/сдвиг/масштаб не влияют на признаки).
 * opt.k — полуокно хорды в пересэмплированных точках (по умолчанию 2).
 * opt.field — true: дополнительно вернуть «поле» (координаты пересэмплированных
 *   точек и их реперы T/N), чтобы нарисовать расслоение поверх чернил (glyphlab);
 *   это те же T/N, из которых считаются признаки, а не пересчёт заново.
 * Возвращает { feats: Float32Array(counts.sum * STRIDE), counts: Int32Array,
 *              scale, medSpeed, field|null }.
 */
export function buildGlyphVectors(strokes, opt = {}) {
  const ink = glyphScale(strokes);
  const scale = Math.max(1e-6, ink || 1);
  const nonEmpty = strokes.reduce((a, st) => a + (st.length > 0 ? 1 : 0), 0) || 1;
  const ds = Math.max(DS_MIN, (ink > 0 ? ink / nonEmpty : PER_STROKE_POINTS) / PER_STROKE_POINTS);
  const k = Math.max(1, opt.k ?? 2);

  // 1) пересэмплировать каждый штрих равным шагом дуги
  const rs = strokes.map((st) => resampleStroke(st, ds));

  // 2) медиана скорости ds/dt по всем внутренним точкам (устойчивая нормировка)
  const speeds = [];
  for (const st of rs) {
    for (let j = 1; j < st.length - 1; j++) {
      const dt = st[j + 1].t - st[j - 1].t;
      if (dt > 0) {
        const dsLen = hypot(st[j + 1].x - st[j - 1].x, st[j + 1].y - st[j - 1].y);
        speeds.push(dsLen / dt);
      }
    }
  }
  const medSpeed = median(speeds) || 1;

  // 3) признаки в локальном репере (T,N)
  let total = 0;
  const counts = new Int32Array(rs.length);
  for (let i = 0; i < rs.length; i++) { counts[i] = rs[i].length; total += rs[i].length; }
  const feats = new Float32Array(total * STRIDE);
  // «Поле» для визуализации: на каждую пересэмплированную точку — её координаты
  // (x, y, номер штриха, номер точки) и её репер (Tx, Ty, Nx, Ny).
  const field = opt.field
    ? { ds, k, pts: new Float32Array(total * 4), frames: new Float32Array(total * 4) }
    : null;
  let o = 0;
  for (let si = 0; si < rs.length; si++) {
    const st = rs[si];
    for (let j = 0; j < st.length; j++) {
      const p = st[j];
      // касательная: центральная разность с окном (устойчивее к шуму), край -> крайний сосед
      const a = st[Math.max(0, j - k)], b = st[Math.min(st.length - 1, j + k)];
      let Tx = b.x - a.x, Ty = b.y - a.y;
      const Tlen = hypot(Tx, Ty);
      if (Tlen > 1e-9) { Tx /= Tlen; Ty /= Tlen; } else { Tx = 1; Ty = 0; } // репер точки
      const Nx = -Ty, Ny = Tx; // нормаль = повернуть T на +90°
      if (field) {
        const q = (o / STRIDE) * 4;
        field.pts[q] = p.x; field.pts[q + 1] = p.y; field.pts[q + 2] = si; field.pts[q + 3] = j;
        field.frames[q] = Tx; field.frames[q + 1] = Ty; field.frames[q + 2] = Nx; field.frames[q + 3] = Ny;
      }

      // векторы-хорды вперёд/назад, выраженные в репере -> инвариантны к повороту
      const fwd = st[Math.min(st.length - 1, j + k)];
      const bwd = st[Math.max(0, j - k)];
      const dfx = fwd.x - p.x, dfy = fwd.y - p.y;
      const dbx = p.x - bwd.x, dby = p.y - bwd.y;
      const fx = (dfx * Tx + dfy * Ty) / scale;
      const fy = (dfx * Nx + dfy * Ny) / scale;
      const bx = (dbx * Tx + dby * Ty) / scale;
      const by = (dbx * Nx + dby * Ny) / scale;

      // угол поворота касательной (кривизна): между входным и выходным сегментом
      let dth = 0;
      if (j > 0 && j < st.length - 1) {
        const ix = p.x - st[j - 1].x, iy = p.y - st[j - 1].y;
        const ox = st[j + 1].x - p.x, oy = st[j + 1].y - p.y;
        const dot = ix * ox + iy * oy;
        const cross = ix * oy - iy * ox;
        if (hypot(ix, iy) > 1e-9 && hypot(ox, oy) > 1e-9) dth = Math.atan2(cross, dot) / Math.PI;
      }

      // скорость в точке: ds/dt, нормированная логарифмом относительно медианы
      let sp = 0;
      if (j > 0 && j < st.length - 1) {
        const dt = st[j + 1].t - st[j - 1].t;
        if (dt > 0) {
          const dsLen = hypot(st[j + 1].x - st[j - 1].x, st[j + 1].y - st[j - 1].y);
          sp = clamp(Math.log((dsLen / dt) / medSpeed || 1), -4, 4);
        }
      }

      feats[o + VF.FX] = fx;
      feats[o + VF.FY] = fy;
      feats[o + VF.BX] = bx;
      feats[o + VF.BY] = by;
      feats[o + VF.DTH] = dth;
      feats[o + VF.SPEED] = sp;
      feats[o + VF.PRESS] = clamp(p.pressure ?? 0, 0, 1);
      feats[o + VF.TILT_X] = clamp((p.tiltX ?? 0) / 90, -1, 1);
      feats[o + VF.TILT_Y] = clamp((p.tiltY ?? 0) / 90, -1, 1);
      feats[o + VF.PENUP] = j === 0 ? 1 : 0;
      o += STRIDE;
    }
  }
  return { feats, counts, scale: glyphScale(strokes), medSpeed, field };
}

// Индексы признаков, по которым строим «вектор-описание глифа» (без PENUP).
const EMB_FEATS = [VF.FX, VF.FY, VF.BX, VF.BY, VF.DTH, VF.SPEED, VF.PRESS, VF.TILT_X, VF.TILT_Y];
const HIST_BINS = 8; // гистограмма DTH по [-π, π] — «профиль формы» глифа

// Каналы для ПОРЯДКОВОГО шаблона (сохраняем порядок вдоль траектории — это и есть
// суть «поля векторов»; глобальные средние порядок теряют). Экспортируем: страница
// /glyphlab.html подписывает ряды дескриптора по этому же списку.
export const TEMPLATE_CH = [VF.FY, VF.BY, VF.DTH, VF.SPEED];
export const TEMPLATE_CH_NAMES = ["FY", "BY", "DTH", "SPEED"]; // порядок совпадает с TEMPLATE_CH (проверено тестом)

/**
 * Фиксразмерное описание глифа (для метрик/проб без обучающей сети): среднее и
 * дисперсия каждого признака + гистограмма углов поворота + число штрихов.
 * Инвариантно к повороту/сдвигу/масштабу (все признаки уже инвариантны).
 */
export function glyphEmbedding(g) {
  const n = g.counts.reduce((a, b) => a + b, 0);
  const emb = new Float64Array(EMB_FEATS.length * 2 + HIST_BINS + 2);
  const mean = new Float64Array(EMB_FEATS.length);
  const varr = new Float64Array(EMB_FEATS.length);
  const hist = new Float64Array(HIST_BINS);
  if (n > 0) {
    for (let i = 0; i < n; i++) {
      const base = i * STRIDE;
      for (let f = 0; f < EMB_FEATS.length; f++) mean[f] += g.feats[base + EMB_FEATS[f]];
      const dth = g.feats[base + VF.DTH]; // [-1,1]
      const bin = clamp(Math.floor(((dth + 1) / 2) * HIST_BINS), 0, HIST_BINS - 1);
      hist[bin] += 1;
    }
    for (let f = 0; f < EMB_FEATS.length; f++) mean[f] /= n;
    for (let i = 0; i < n; i++) {
      const base = i * STRIDE;
      for (let f = 0; f < EMB_FEATS.length; f++) {
        const d = g.feats[base + EMB_FEATS[f]] - mean[f];
        varr[f] += d * d;
      }
    }
    for (let f = 0; f < EMB_FEATS.length; f++) varr[f] = Math.sqrt(varr[f] / n);
    for (let b = 0; b < HIST_BINS; b++) hist[b] /= n;
  }
  let p = 0;
  for (let f = 0; f < EMB_FEATS.length; f++) emb[p++] = mean[f];
  for (let f = 0; f < EMB_FEATS.length; f++) emb[p++] = varr[f];
  for (let b = 0; b < HIST_BINS; b++) emb[p++] = hist[b];
  emb[p++] = g.counts.length / 8;     // число штрихов (норм.)
  emb[p++] = g.scale >= 1e-6 ? 1 : 0; // глиф нетривиален (не точка)
  return emb;
}

// Евклидово расстояние двух описаний (одинаковой длины).
export function embedDist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

/**
 * ПОРЯДКОВЫЙ шаблон глифа: упорядоченную вдоль траектории последовательность
 * признаков (TEMPLATE_CH) пересэмплируем к L отсчётам и кладём каналы рядом.
 * Порядок сохранён (в отличие от glyphEmbedding) — «где именно изгиб», а не
 * «сколько изгибов всего». Инвариантен к повороту/сдвигу/масштабу (каналы уже
 * инвариантны). Это честный no-training прокси «поля векторов» для одной буквы.
 * + два глобальных числа: число штрихов и число лифтов (норм.).
 */
export function glyphTemplate(g, L = 24) {
  const n = g.counts.reduce((a, b) => a + b, 0);
  const out = new Float64Array(L * TEMPLATE_CH.length + 2);
  if (n > 0) {
    for (let c = 0; c < TEMPLATE_CH.length; c++) {
      const ch = TEMPLATE_CH[c];
      for (let j = 0; j < L; j++) {
        const pos = (n - 1) * (j / (L - 1)); // доля пути вдоль последовательности точек
        const lo = Math.floor(pos), hi = Math.min(n - 1, lo + 1), fr = pos - lo;
        const a = g.feats[lo * STRIDE + ch], b = g.feats[hi * STRIDE + ch];
        out[j * TEMPLATE_CH.length + c] = a + (b - a) * fr;
      }
    }
  }
  let penups = 0;
  for (let i = 0; i < n; i++) if (g.feats[i * STRIDE + VF.PENUP] === 1) penups++;
  out[L * TEMPLATE_CH.length] = g.counts.length / 8;
  out[L * TEMPLATE_CH.length + 1] = penups / 8;
  return out;
}

/**
 * Восстановить штрихи из одной «сплошной» траектории (xs, ys, dt) — так refs
 * хранят в БД (getTrajectory склеивает штрихи, вставляя реальную паузу в dt).
 * Разрыв = «перо поднято»: большой dt при заметном скачке координаты (одна
 * большая пауза на месте без движения — это залом пера, а не лифт).
 */
export function splitTrajectory(xs, ys, dt) {
  const n = Math.min(xs.length, ys.length, dt.length);
  if (n === 0) return [];
  const dts = dt.slice(0, n).filter((v) => v > 0).sort((a, b) => a - b);
  const medDt = dts.length ? dts[dts.length >> 1] : 16;
  let sum = 0, c = 0;
  for (let i = 1; i < n; i++) { const d = hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]); sum += d; c++; }
  const medStep = c ? sum / c : 1;
  const strokes = [[{ x: xs[0], y: ys[0], t: 0 }]];
  let t0 = 0;
  for (let i = 1; i < n; i++) {
    const jump = hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]);
    t0 += dt[i];
    if (dt[i] > medDt * 4 && jump > medStep * 1.5) strokes.push([{ x: xs[i], y: ys[i], t: t0 }]);
    else strokes[strokes.length - 1].push({ x: xs[i], y: ys[i], t: t0 });
  }
  return strokes;
}

/**
 * Аугментация набора штрихов одного глифа (для раздувания тонкой выборки).
 * Реалистичная для online-чернил: микро-дрожание руки (гауссов шум к точкам),
 * лёгкий поворот/масштаб «на странице», вариативность скорости (множитель ко
 * временным интервалам -> меняет SPEED-признак). Всё относительно центроида,
 * поэтому не уносит глиф в сторону. rng — генератор [0,1) (инжектируется, чтобы
 * было детерминировано в тестах и воспроизводимо в обучении).
 */
export function augmentStrokes(strokes, rng, opt = {}) {
  const jit = opt.jitter ?? 0.004;            // σ дрожания = доля «размера» глифа
  const rotMax = opt.rot ?? (12 * Math.PI) / 180;
  const scaleMax = opt.scale ?? 0.12;
  const speedMax = opt.speed ?? 0.2;
  let cx = 0, cy = 0, n = 0;
  for (const st of strokes) for (const p of st) { cx += p.x; cy += p.y; n++; }
  if (!n) return strokes.map((s) => s.map((q) => ({ ...q })));
  cx /= n; cy /= n;
  let sp2 = 0;
  for (const st of strokes) for (const p of st) { const dx = p.x - cx, dy = p.y - cy; sp2 += dx * dx + dy * dy; }
  const spread = Math.sqrt(sp2 / n) || 1;
  const gauss = () => { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const a = (rng() * 2 - 1) * rotMax, s = 1 + (rng() * 2 - 1) * scaleMax;
  const ca = Math.cos(a), sa = Math.sin(a);
  return strokes.map((st) => {
    const f = Math.exp(gauss() * speedMax); // множитель времени штриха (меняет SPEED)
    const out = new Array(st.length);
    let tAcc = 0;
    for (let i = 0; i < st.length; i++) {
      const p = st[i], dx = p.x - cx, dy = p.y - cy;
      const t = p.t == null ? undefined
        : i === 0 ? 0
        : (tAcc += Math.max(0.5, (p.t - st[i - 1].t) * f * (1 + gauss() * speedMax)));
      out[i] = {
        x: cx + (dx * ca - dy * sa) * s + gauss() * jit * spread,
        y: cy + (dx * sa + dy * ca) * s + gauss() * jit * spread,
        t, pressure: p.pressure, tiltX: p.tiltX, tiltY: p.tiltY,
      };
    }
    return out;
  });
}
