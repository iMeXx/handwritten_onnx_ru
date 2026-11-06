// Эвристическая проверка написания перед сохранением — ловит явный мусор:
// касания, точки, линии, «простыни». Байт-копия src/calib-verify.js
// (семантика гейтов совпадает с гейтами билдера датасета).
// штрихи — как в PenCapture.strokes: [[{x, y, t}, ...], ...]

const MIN_POINTS = 4;      // как в билдере датасета: < 4 точек prepare_data отбрасывает
const MAX_POINTS = 8000;   // один символ столько не пишет — это закрашивание холста
const MIN_SIDE = 10;       // px: меньше — точка/крапинка, а не буква
const MAX_STROKES = 8;     // больше штрихов — это не один символ
const MAX_ASPECT = 7;      // w/h: длинная горизонтальная линия (случайный «мазок»)
const FAST_MS = 120;       // суммарное время ниже — быстрее морга, случайное касание
const FAST_POINTS = 30;    // и при этом штрих короткий

export function junkCheck(strokes, canvasW, canvasH) {
  const list = (strokes || []).filter((s) => s && s.length);
  let n = 0, t0 = Infinity, t1 = -Infinity;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of list) {
    n += s.length;
    for (const p of s) {
      if (Number.isFinite(p.t)) { t0 = Math.min(t0, p.t); t1 = Math.max(t1, p.t); }
      x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x);
      y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y);
    }
  }
  if (!list.length || n < MIN_POINTS)
    return `Слишком короткое написание (< ${MIN_POINTS} точек) — перепишите символ.`;
  if (n > MAX_POINTS)
    return "Слишком много точек — похоже на закрашивание, а не на символ. Перепишите.";
  if (list.length > MAX_STROKES)
    return `Слишком много штрихов (${list.length}) — напишите один символ целиком.`;
  const w = x1 - x0, h = y1 - y0;
  if (w < MIN_SIDE || h < MIN_SIDE)
    return "Похоже на точку, а не на букву — перепишите покрупнее, поверх подсказки.";
  if (canvasW && w / h > MAX_ASPECT)
    return "Похоже на горизонтальную линию, а не на символ — перепишите.";
  if (Number.isFinite(t0) && t1 - t0 < FAST_MS && n < FAST_POINTS)
    return "Слишком быстрый штрих — похоже на случайное касание. Перепишите.";
  return null; // мусор не найден
}
