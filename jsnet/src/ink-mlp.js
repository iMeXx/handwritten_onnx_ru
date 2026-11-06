// Инференс-слой векторного MLP: штрихи -> буква. Тот же код матиматики, что и в
// тренировке (scripts/train-vectors.mjs импортирует forward/initMLP отсюда), и
// ровно то, что воркер прогонит напрямую (ONNX не нужен: веса — JSON inknet-mlp/1).
//
// Пайплайн предсказания: штрихи -> buildGlyphVectors -> glyphTemplate(L) ->
// стандартизация (mean/sd из модели) -> MLP (ReLU + softmax) -> argmax по буквам,
// виденным в train (маска trainChars: unseen-букву не выдумываем).
//
// Чистый модуль (без DOM) — работает в node (тесты) и в воркере.
import { buildGlyphVectors, glyphTemplate } from "./ink-features.js";

export const MLP_FORMAT = "inknet-mlp/1";

// ————— Стандартизация порядкового шаблона (общая для тренировки и инференса) —————
// У slотов, которые по всему корпусу имеют одно и то же значение (например BY в
// самом конце траектории — туда хорда не дотягивается), разброс вырожденно мал:
// sd выходит ~1e-18 при машинном шуме того же порядка. Деление на такую sd
// усиливает шум до единиц, и ОДИН И ТОТ ЖЕ глиф после поворота становится другой
// буквой — инвариантность репрезентации ломается на ровном месте. Поэтому:
//  • sd снизу ограничена (SD_MIN): константный слот даёт z ≈ 0, а не шум;
//  • z обрезается по модулю (Z_CLAMP): выброс не может унести вход в бесконечность.
// Признаки нормированы по длине чернил, их масштаб ~1, так что SD_MIN=1e-6 —
// это «меньше любого осмысленного различия написаний».
export const SD_MIN = 1e-6;
export const Z_CLAMP = 12;

// mean/sd по набору векторов; sd сразу с нижней границей — чтобы train и infer
// считали стандартизацию ОДНИМ кодом и не разъезжались.
export function fitStandardize(vecs) {
  const d = vecs[0].length, mean = new Float64Array(d), varr = new Float64Array(d);
  for (const v of vecs) for (let i = 0; i < d; i++) mean[i] += v[i];
  for (let i = 0; i < d; i++) mean[i] /= vecs.length;
  for (const v of vecs) for (let i = 0; i < d; i++) { const z = v[i] - mean[i]; varr[i] += z * z; }
  const sd = new Float64Array(d);
  let degenerate = 0;
  for (let i = 0; i < d; i++) {
    const s = Math.sqrt(varr[i] / vecs.length);
    sd[i] = Number.isFinite(s) && s > SD_MIN ? s : SD_MIN;
    if (sd[i] === SD_MIN) degenerate++;
  }
  return { mean, sd, degenerate };
}

// z = (v - mean)/sd с ограничением по модулю; out (опционально) — переиспользуемый буфер.
export function applyStandardize(v, st, out) {
  const z = out && out.length === v.length ? out : new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) {
    const s = st.sd[i] > SD_MIN ? st.sd[i] : SD_MIN; // защита от старых файлов с sd=0
    const q = (v[i] - st.mean[i]) / s;
    z[i] = q > Z_CLAMP ? Z_CLAMP : q < -Z_CLAMP ? -Z_CLAMP : q;
  }
  return z;
}

// Инициализация MLP (He-инициализация) — общий вход для тренировщика.
export function initMLP(d, H, C, rng) {
  const randn = () => { let u = 0, v = 0; while (!u) u = rng(); while (!v) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const W1 = new Float64Array(d * H), W2 = new Float64Array(H * C), b1 = new Float64Array(H), b2 = new Float64Array(C);
  for (let i = 0; i < d * H; i++) W1[i] = randn() * Math.sqrt(2 / d);
  for (let i = 0; i < H * C; i++) W2[i] = randn() * Math.sqrt(2 / H);
  return { d, H, C, W1, b1, W2, b2 };
}

// Прямой проход: x[d] -> softmax(W2·ReLU(W1 x + b1) + b2). Пишет в scratch a/h/p.
export function forward(m, x, a, h, p) {
  const { d, H, C, W1, b1, W2, b2 } = m;
  let mx = -Infinity;
  for (let j = 0; j < H; j++) { let s = b1[j]; for (let i = 0; i < d; i++) s += x[i] * W1[i * H + j]; a[j] = s; h[j] = s > 0 ? s : 0; }
  for (let k = 0; k < C; k++) { let s = b2[k]; for (let j = 0; j < H; j++) s += h[j] * W2[j * C + k]; p[k] = s; if (s > mx) mx = s; }
  let sum = 0; for (let k = 0; k < C; k++) { p[k] = Math.exp(p[k] - mx); sum += p[k]; }
  for (let k = 0; k < C; k++) p[k] /= sum;
}

export function argmaxMasked(p, mask) { let b = -1, bv = -Infinity; for (let k = 0; k < p.length; k++) if (!mask || mask[k]) { if (p[k] > bv) { bv = p[k]; b = k; } } return b; }

// encoded-образец (как в fold<i>.json) -> объект-глиф для glyphTemplate.
const toGlyph = (e) => ({
  feats: e.feats instanceof Float32Array ? e.feats : Float32Array.from(e.feats),
  counts: e.counts instanceof Int32Array ? e.counts : Int32Array.from(e.counts),
  scale: e.scale,
});

/**
 * Движок из сохранённой модели (inknet-mlp/1). predictStrokes — путь воркера,
 * predictEncoded — путь набора (fold<i>.json); дают идентичный результат.
 */
export function makeEngine(model) {
  const { H, C, L, alphabet } = model;
  // mean/sd из файла прогоняются через тот же «безопасный» фильтр, что и при
  // тренировке (fitStandardize): вырожденно малая sd не должна усиливать машинный
  // шум — иначе один и тот же глиф после поворота даёт другую букву.
  const st = {
    mean: Float64Array.from(model.standardize.mean),
    sd: Float64Array.from(model.standardize.sd, (v) => (Number.isFinite(v) && v > SD_MIN ? v : SD_MIN)),
  };
  const m = {
    d: model.W1.length / H, H, C,
    W1: Float64Array.from(model.W1), b1: Float64Array.from(model.b1),
    W2: Float64Array.from(model.W2), b2: Float64Array.from(model.b2),
  };
  const mask = new Uint8Array(C);
  for (const ch of model.trainChars || alphabet) { const k = alphabet.indexOf(ch); if (k >= 0) mask[k] = 1; }
  const a = new Float64Array(H), h = new Float64Array(H), p = new Float64Array(C);
  const encode = (glyph) => applyStandardize(glyphTemplate(glyph, L), st);
  const predictGlyph = (glyph) => {
    const x = encode(glyph);
    forward(m, x, a, h, p);
    const id = argmaxMasked(p, mask);
    return { id, char: id >= 0 ? alphabet[id] : "", p: Array.from(p), conf: id >= 0 ? p[id] : 0 };
  };
  return {
    d: m.d, H, C, L, mask, alphabet,
    //mean/sd в «безопасном» виде (sd с нижней границей) — то, чем реально кодируется вход
    st,
    // z-кодирование готового вектора шаблона: страница/галерея кодируют эталоны
    // ровно тем же кодом, что и вход сети, иначе «ближайший сосед» меряет не то
    zscore: (v) => applyStandardize(v, st),
    // стандартный «вектор глифа» (та же кодировка, что у входа MLP) — для метрик
    // ближайшего соседа и визуализации дескриптора; путь воркера — encodeStrokes.
    encode,
    encodeStrokes: (strokes) => encode(buildGlyphVectors(strokes)),
    encodeEncoded: (e) => encode(toGlyph(e)),
    predictGlyph,
    predictEncoded: (e) => predictGlyph(toGlyph(e)),
    predictStrokes: (strokes) => predictGlyph(buildGlyphVectors(strokes)),
    charOf: (id) => (id >= 0 ? alphabet[id] : ""),
  };
}

// Сериализация обученного MLP в JSON-объект формата inknet-mlp/1 (тренировщик
// пишет этим же, чтобы тест раунд-трипа сверялся с реальным форматом).
export function serializeMLP(m, meta) {
  return {
    format: MLP_FORMAT, L: meta.L, H: m.H, C: m.C, alphabet: meta.alphabet, stride: meta.stride,
    standardize: { mean: Array.from(meta.mean), sd: Array.from(meta.sd) },
    W1: Array.from(m.W1), b1: Array.from(m.b1), W2: Array.from(m.W2), b2: Array.from(m.b2),
    trainChars: meta.trainChars,
  };
}
