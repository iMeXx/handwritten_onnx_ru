// Web Worker: CTC-распознавание строк (страница lines.html); vendor-адаптация
// одна — onnxruntime-web без npm/сборки:
//  • ort берётся из local'ной копии js/vendor/onnxruntime/ort.wasm.bundle.min.mjs
//    (wasm-only сборка 1.31.1, байт-копия npm-пакета);
//  • WebGPU-ветка убрана: jsnet — локальное приложение, пишет с планшета по
//    сети на http://IP (не secure context), там navigator.gpu и Cache Storage
//    недоступны в принципе; jsep/wsgb-сборка (~29 МБ wasm) грузится только на
//    https/localhost и на Kirin/Huawei всё равно не заводится. Остаётся WASM.
// Сообщения воркеру:
//  {id, modelUrl, pts, counts, pen, w, h, dpr, gen, alpha, terms} — штрихи
//  упакованными буферами (packStrokes); растеризация, нарезка на строки и крой
//  полос в тензор делаются прямо в воркере (StripPrep): main-поток свободен
//  во время письма, pointermove не теряет точки.
//  {id, modelUrl, sample: true, strokes, pen, w, h} — QC-полоса всего инка.
//  {id, modelUrl, strips} — готовые полосы (обратная совместимость).
//  alpha + terms (norm-термины) — включают beam search со словарным бонусом
//  (ctc-decode.js); тришка строится по первому сообщению с непустым словарём.
import * as ort from "./vendor/onnxruntime/ort.wasm.bundle.min.mjs";
import { StripPrep, unpackStrokes } from "./line-preprocess.js";
import { decodeCTC, beamDecode } from "./ctc-decode.js";

// wasm-бинарь и mjs-загрузчик потоков лежат рядом с этим bundle; фиксируем путь
// от корня сервера — воркер резолвит ort-ассеты не по import.meta, а по wasmPaths
ort.env.wasm.wasmPaths = "/js/vendor/onnxruntime/";

// Инференс ort гонится в пуле WASM-потоков (busy-poll) и на слабых мобильных
// (Huawei/Kirin) выжигает все ядра: Android начинает отбрасывать события пера,
// письмо идёт «пробросами». На тач-устройствах — один WASM-поток: инференс
// медленнее, но перо не теряется; при повторных распознаваниях почти всё
// и так отдаётся stripCache без инференса. Без COOP/COEP-заголовков (jsnet их
// не шлёт) SharedArrayBuffer нет и ort сам кладёт потоки в 1.
const isTouch = (navigator.maxTouchPoints || 0) > 0;
const cores = navigator.hardwareConcurrency || 2;
ort.env.wasm.numThreads = isTouch || cores <= 4 ? 1 : Math.min(4, cores);

const prep = new StripPrep();
const stripCache = new Map(); // hash(тензор) -> decode result
let decoder = null; // {alpha, words} — beam search со словарём (страница присылает однажды)

function hashBytes(f32) {
  const u = new Uint8Array(f32.buffer);
  let h = 0x811c9dc5;
  for (let i = 0; i < u.length; i++) h = ((h ^ u[i]) * 16777619) >>> 0;
  return h;
}

let session = null;
let loadPromise = null;

// сигнатура модели: protobuf-ONNX (поле 1 varint) или ORT-формат (magic "ORTM"
// в байтах 4..7 — flatbuffer, см. onnxruntime docs «ORT model format»)
function looksLikeOnnx(buf) {
  if (!buf || buf.byteLength < 65536) return false;
  const u = new Uint8Array(buf);
  if (u[0] === 0x08) return true;
  return u[4] === 0x4f && u[5] === 0x52 && u[6] === 0x54 && u[7] === 0x4d; // "ORTM"
}

// Cache Storage работает только в secure context (http://localhost / https);
// на планшете по http://IP caches нет — все обращения под try/catch, модель
// просто качается каждый заход (3.6 МБ по локальной сети — терпимо).
const MODEL_CACHE = "inknet-models-v2"; // v2: модель со «своими» пробелами в словах

async function getModelBuffer(url) {
  try {
    const cache = await caches.open(MODEL_CACHE);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      if (looksLikeOnnx(buf)) return buf;
    }
  } catch { /* caches недоступен */ }
  const buf = await downloadWithProgress(url);
  if (looksLikeOnnx(buf)) {
    try {
      const cache = await caches.open(MODEL_CACHE);
      await cache.put(url, new Response(buf.slice(0)));
    } catch { /* ignore */ }
  }
  return buf;
}

function createSession(buf) {
  return ort.InferenceSession.create(buf, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
}

// .ort — предоптимизированный формат ONNX Runtime (без protobuf-парсинга в JS:
// создание сессии ~13x быстрее).
// Если .ort нет на сервере или сборка его не поднимает — прозрачный откат на .onnx.
function ortUrlOf(modelUrl) {
  return modelUrl.replace(/\.onnx$/i, ".ort");
}

function load(modelUrl) {
  if (!loadPromise) {
    loadPromise = (async () => {
      const ortUrl = ortUrlOf(modelUrl);
      let buf = null;
      if (ortUrl !== modelUrl) {
        try { buf = await getModelBuffer(ortUrl); } catch { buf = null; } // 404 — откат
      }
      let usedOrt = !!buf;
      if (!buf) buf = await getModelBuffer(modelUrl);
      self.postMessage({ type: "progress", loaded: buf.byteLength, total: buf.byteLength });
      try {
        return await createSession(buf);
      } catch (err) {
        if (!usedOrt) throw err;
        // сборка не подняла .ort (несовместимый opset) — качаем и поднимаем .onnx
        buf = await getModelBuffer(modelUrl);
        self.postMessage({ type: "progress", loaded: buf.byteLength, total: buf.byteLength });
        return await createSession(buf);
      }
    })().catch((err) => {
      loadPromise = null; // ошибка загрузки — можно перепробовать другой url
      throw err;
    });
  }
  return loadPromise;
}

async function downloadWithProgress(modelUrl) {
  const resp = await fetch(modelUrl, { cache: "reload" });
  if (!resp.ok) throw new Error("model http " + resp.status);
  const total = Number(resp.headers.get("content-length")) || 0;
  if (!resp.body || total === 0) return resp.arrayBuffer();
  const chunks = [];
  let loaded = 0;
  const reader = resp.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    self.postMessage({ type: "progress", loaded, total });
  }
  const merged = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) { merged.set(c, off); off += c.byteLength; }
  if (!looksLikeOnnx(merged.buffer)) {
    throw new Error("по адресу " + modelUrl + " отдаётся не ONNX (проверьте public/models/)");
  }
  return merged.buffer;
}

self.onmessage = async (e) => {
  const { id, modelUrl } = e.data;
  // словарь для beam search: страница шлёт alpha (символы классов) + terms (norm)
  // в каждом сообщении; тришка строится один раз (первое непустое сообщение).
  if (!decoder && Array.isArray(e.data.alpha) && Array.isArray(e.data.terms) && e.data.terms.length) {
    decoder = { alpha: e.data.alpha, words: e.data.terms };
  }
  try {
    const t0 = performance.now();
    session = session || (await load(modelUrl));
    // штрихи: упакованные (Float32Array+Int32Array transfer) или объектами (старый путь)
    const strokes = e.data.pts ? unpackStrokes(e.data.pts, e.data.counts) : e.data.strokes;
    let strips;
    if (e.data.strips) {
      strips = e.data.strips;
    } else if (e.data.sample) {
      const s = prep.sample(strokes, e.data.pen || {}, e.data.w, e.data.h);
      strips = s && s.w >= 16 ? [s] : [];
    } else {
      strips = prep.prep(strokes || [], e.data.pen || {}, e.data.w, e.data.h, e.data.dpr || 1, e.data.gen || 0);
    }
    const out = [];
    // QC-образцы (sample) декодируем жадно: эталон должен читаться «как есть»,
    // словарный бонус не должен скрывать кривое написание от QC в profile/collect
    const useBeam = decoder && !e.data.sample;
    for (const s of strips) {
      // ключ кэша учитывает режим декодера: одна и та же полоса в beam и greedy
      // читается по-разному (QC vs распознавание)
      const key = hashBytes(s.data) + (useBeam ? ":b" : ":g");
      const hit = stripCache.get(key);
      if (hit) { out.push(hit); continue; }
      const tensor = new ort.Tensor("float32", s.data, [1, 1, 64, s.w]);
      const res = await session.run({ image: tensor });
      const dec = useBeam
        ? beamDecode(res.logits.data, res.logits.dims[2],
            // бонус 2.5 + oovPen 0.6 — откалибровано на writer-disjoint стенде
            // (web_onnx/scripts/dec-bench.mjs, 600 строк): CER 0.199->0.182,
            // точные строки 0.33->0.40; сильный oovPen(>1) ещё чуть лучше на
            // словарных строках, но затирает имена — для школьного текста 0.6
            { words: decoder.words, alpha: decoder.alpha, bonus: 2.5, beam: 64, oovPen: 0.6 })
        : decodeCTC(res.logits.data, res.logits.dims[2]);
      stripCache.set(key, dec);
      out.push(dec);
    }
    self.postMessage({ id, lines: out, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message ? err.message : err) });
  }
};
