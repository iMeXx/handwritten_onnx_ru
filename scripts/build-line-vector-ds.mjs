// Сборка ОБУЧАЮЩЕГО датасета «векторные строки» (M1 baseline для inknet-online).
//
// Идея: строковые написания в БД не собираются (train_samples — отдельные
// символы), поэтому строки СИНТЕЗИРУЮТСЯ из глифов одного автора: берём слово
// из словаря (terms.json), чьи буквы есть у автора, и склеиваем векторные
// признаки его написаний этих букв в один поток кадров. Метка — классы букв
// слова (CTC). Оценка writer-disjoint: строки одного автора синтезируются из
// его же почерка, поэтому train/test фолды не пересекаются по авторам и
// метрика честная (тестируется рукописная база автора, а не чужая).
//
// Выход (data/line-vectors/ по умолчанию):
//   fold<i>.train.json / fold<i>.test.json — формат inknet-line-vectors/1
//   manifest.json — vocab, фолды, mean/sd стандартизации
// Признаки считаются в JS (src/ink-features.js — эталонный extractor, он же
// будет на serving), стандартизация — fitStandardize из src/ink-mlp.js с
// ограничениями SD_MIN/Z_CLAMP. Python-тренировщик читает уже тензоры.
//
// v3 (фиксируются в manifest): несколько слов на строку. Пробел представлен
// КАДРАМИ ПЕРЕНОСА ПЕРА (виртуальный штрих без чернил от конца одного слова до
// начала следующего): векторный путь видит перенос как кадры с PENUP=1 и
// метку «пробел»; растровый путь его НЕ рисует — в растре пробел это чистый
// пробел между словами. Граница символа в кадрах уже есть: каждая глиф-база
// начинается с PENUP=1 (buildGlyphVectors считается отдельно на глиф).
// запуск: make ds   (или node scripts/build-line-vector-ds.mjs [OUT])
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
// пути и параметры — из .env (через make) или аргументов; WEB_ONNX — корень web_onnx_v2,
// DB_PATH подхватывает server/db.js сам
const WEB = resolve(process.env.WEB_ONNX || resolve(HERE, "..", "..", "web_onnx_v2"));
const { db } = await import(resolve(WEB, "server/db.js"));
const { buildGlyphVectors, splitTrajectory, augmentStrokes, STRIDE } =
  await import(resolve(WEB, "src/ink-features.js"));
const { fitStandardize, applyStandardize } = await import(resolve(WEB, "src/ink-mlp.js"));

const outDir = resolve(process.argv[2] || process.env.DATA_DIR || "data/line-vectors");
mkdirSync(outDir, { recursive: true });

const MIN_PTS = 4;            // «мусор»-гейт как в collect/calib-verify
const MIN_WRITER_CHARS = Number(process.env.MIN_WRITER_CHARS || 10); // автор с меньшим покрытием букв не участвует
const TRAIN_LINES = Number(process.env.LINES || 150); // синтезированных строк на автора в train
const TEST_LINES = Number(process.env.TEST_LINES || 30);
const MIN_WORD = 2, MAX_WORD = 10;
const MULTIWORD_PROB = Number(process.env.MULTIWORD_PROB || 0.4); // доля двухсловных строк (одинакова в train и test — иначе распределение смещено)

// splitmix32, а НЕ mulberry32: сиды строк — арифметические прогрессии, а у
// mulberry32 состояние линейно по сидам (малая разница сидов слабо диффундирует
// в старшие биты выхода) — соседние строки выбирали почти одинаковые слова и
// глифы (первый выход ~0.95 при любом сиде с шагом 6151; реальный баг v1/v2).
// Вейлево состояние + murmur3-финализатор: полный avalanche по всем битам.
const m32 = (seed) => { let a = seed >>> 0;
  return () => { a = (a + 0x9e3779b9) | 0;
    let z = a;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return ((z ^ (z >>> 16)) >>> 0) / 4294967296; }; };
const pick = (rng, arr) => arr[Math.floor(rng() * arr.length) % arr.length];

// ————— алфавит модели: класс 0 = blank, класс i>0 = line_vocab.json[i-1] —————
const vocab = JSON.parse(readFileSync(resolve(WEB, "public/models/line_vocab.json"), "utf8"));
const clsOf = new Map(vocab.map((ch, i) => [ch, i + 1]));

// ————— словарь слов (terms.json страницы beam-декодера) —————
const terms = JSON.parse(readFileSync(resolve(WEB, "public/terms.json"), "utf8"));
const wordPool = new Set();
for (const list of [terms.terms_ru, terms.terms_en]) for (const w of list || [])
  for (const tok of String(w).toLowerCase().split(/[\s\-–—,.()]+/))
    if (tok.length >= MIN_WORD && tok.length <= MAX_WORD && [...tok].every((c) => clsOf.has(c)))
      wordPool.add(tok);
const words = [...wordPool].sort();

// ————— глифы авторов из БД (refs + train_samples), к нижнему регистру —————
const byWriter = new Map(); // writer -> Map(char -> [strokes...])
for (const u of db.prepare("SELECT id, username FROM users ORDER BY id").all()) {
  const pool = byWriter.get(u.username) || new Map();
  const add = (ch, strokes) => {
    ch = String(ch).toLowerCase();
    if (!clsOf.has(ch)) return;
    if (strokes.reduce((a, s) => a + s.length, 0) < MIN_PTS) return;
    (pool.get(ch) || pool.set(ch, []).get(ch)).push(strokes);
  };
  for (const r of db.prepare("SELECT ch, xs, ys, dt, qc FROM refs WHERE user_id = ?").all(u.id)) {
    if (r.qc) continue; // не принятые QC эталоны — как в inknet-train
    add(r.ch, splitTrajectory(JSON.parse(r.xs), JSON.parse(r.ys), JSON.parse(r.dt)));
  }
  for (const r of db.prepare("SELECT ch, strokes FROM train_samples WHERE user_id = ?").all(u.id))
    add(r.ch, JSON.parse(r.strokes).map((s) => s.map((p) => ({ x: p.x, y: p.y, t: p.t }))));
  if (pool.size >= MIN_WRITER_CHARS) byWriter.set(u.username, pool);
  else console.log(`пропуск автора ${u.username}: букв ${pool.size} < ${MIN_WRITER_CHARS}`);
}
const writers = [...byWriter.keys()];
if (writers.length < 2) { console.error("нужно >= 2 автора с покрытием >= " + MIN_WRITER_CHARS + " букв"); process.exit(1); }

// ————— синтез строки: кадры глифов по буквам слова, метка = классы букв —————
// buildGlyphVectors считается ОТДЕЛЬНО на каждый глиф (нормировка ds/scale по
// длине чернил глифа — эталонная семантика extractor'а), кадры склеиваются.
// Дополнительно строится ГЕОМЕТРИЯ строки (штрихи с раскладкой по базовой
// линии): тот же набор штрихов растрится растером inknet-train — растровый
// baseline учится ровно на тех же строках, сравнение честное. Точки квантуются
// до 0.1px — размер JSON вдвое меньше, геометрии это не вредит.
const q1 = (v) => Math.round(v * 10) / 10;

function buildLine(rng, writer, words, aug) {
  const pool = byWriter.get(writer);
  const feats = [], counts = [], classes = [], strokes = [], moveStrokes = [], charSpans = [];
  let cursorX = 0, frameOff = 0;
  const placeGlyph = (ch) => {
    let glyph = pick(rng, pool.get(ch));
    if (aug) glyph = augmentStrokes(glyph, rng);
    const g = buildGlyphVectors(glyph);
    for (const v of g.feats) feats.push(v); // плоский поток кадров nFrames*STRIDE
    counts.push(...g.counts);
    classes.push(clsOf.get(ch));
    const nFr = g.counts.reduce((a, b) => a + b, 0);
    charSpans.push([frameOff, frameOff + nFr]); // границы символа в кадрах (анализ)
    frameOff += nFr;
    // раскладка: нижняя граница bbox глифа на базовую линию y=0, курсор по X
    let minX = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const st of glyph) for (const p of st) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y > maxY) maxY = p.y;
    }
    if (!Number.isFinite(minX)) return;
    const dx = cursorX - minX, dy = -maxY;
    for (const st of glyph) strokes.push(st.map((p) => [q1(p.x + dx), q1(p.y + dy), p.t == null ? 0 : Math.round(p.t)]));
    cursorX += (maxX - minX) + (5 + rng() * 9); // междынный интервал
  };
  // перенос пера между словами: виртуальный штрих по базовой линии. Чернил нет
  // (в rasterds ход не рисуется), но кадры дают вектору физический смысл
  // паузы/перелёта: PENUP=1 на первой точке, SPEED ~ скорость переноса.
  const placeMove = (gap, dur) => {
    const n = 9;
    const st = [];
    for (let j = 0; j <= n; j++)
      st.push({ x: cursorX + (gap * j) / n, y: 0, t: Math.round((dur * j) / n) });
    const g = buildGlyphVectors([st]); // scale по длине самого хода — как у глифа
    for (const v of g.feats) feats.push(v);
    counts.push(...g.counts);
    classes.push(clsOf.get(" "));
    const nFr = g.counts.reduce((a, b) => a + b, 0);
    charSpans.push([frameOff, frameOff + nFr]);
    frameOff += nFr;
    moveStrokes.push(st.map((p) => [q1(p.x), q1(p.y), p.t]));
    cursorX += gap;
  };
  words.forEach((word, wi) => {
    if (wi) placeMove(20 + rng() * 18, 150 + rng() * 250);
    for (const ch of word) placeGlyph(ch);
  });
  const nFrames = counts.reduce((a, b) => a + b, 0);
  if (nFrames < 2 * classes.length - 1) return null; // CTC: кадров должно хватать на метку
  if (!strokes.length) return null;
  const word = words.join(" ");
  return { writer, word, classes, counts, nFrames, feats, strokes, moveStrokes, charSpans };
}

// стандартизация по обучающим кадрам фолда (mean/sd в файл — serving parity)
function standardize(splitDoc, st) {
  for (const s of splitDoc.samples) {
    const flat = s.feats;
    const out = new Float32Array(flat.length);
    for (let i = 0; i < flat.length; i += STRIDE) {
      const z = applyStandardize(flat.slice(i, i + STRIDE), st);
      out.set(z, i);
    }
    s.feats = Array.from(out);
  }
}

// ————— leave-one-writer-out фолды —————
const folds = [];
for (let f = 0; f < writers.length; f++) {
  const testWriter = writers[f];
  const trainWriters = writers.filter((w) => w !== testWriter);
  // для каждого автора — свои слова (все буквы слова есть у автора в пуле)
  const wordsFor = (w) => words.filter((word) => [...word].every((c) => byWriter.get(w).has(c)));
  const wTest = wordsFor(testWriter);
  if (!wTest.length) { console.log(`пропуск фолда ${testWriter}: нет слов из его букв`); continue; }

  const train = [];
  for (const w of trainWriters) {
    const wl = wordsFor(w);
    for (let i = 0; i < TRAIN_LINES; i++) {
      const rng = m32((f * 100003 + i * 7919 + w.length * 104729) >>> 0);
      const nWords = rng() < MULTIWORD_PROB ? 2 : 1;
      const s = buildLine(rng, w, Array.from({ length: nWords }, () => pick(rng, wl)), true);
      if (s) train.push(s);
    }
  }
  const test = [];
  for (let i = 0; i < TEST_LINES; i++) {
    const rng = m32((f * 999983 + i * 6151) >>> 0);
    const nWords = rng() < MULTIWORD_PROB ? 2 : 1;
    const s = buildLine(rng, testWriter, Array.from({ length: nWords }, () => pick(rng, wTest)), false);
    if (s) test.push(s);
  }

  const framesTrain = train.flatMap((s) => {
    const out = []; for (let i = 0; i < s.feats.length; i += STRIDE) out.push(s.feats.slice(i, i + STRIDE));
    return out;
  });
  const st = fitStandardize(framesTrain.map((fr) => Float64Array.from(fr)));
  const trDoc = { format: "inknet-line-vectors/3", stride: STRIDE, split: "train", fold: f, testWriter, samples: train };
  const teDoc = { format: "inknet-line-vectors/3", stride: STRIDE, split: "test", fold: f, testWriter, samples: test };
  standardize(trDoc, st); standardize(teDoc, st);
  for (const doc of [trDoc, teDoc])
    for (const s of doc.samples)
      if (s.feats.length !== s.nFrames * STRIDE || s.feats.some((v) => !Number.isFinite(v)))
        throw new Error(`битый образец ${doc.split}/fold${f} «${s.word}»: feats ${s.feats.length} при nFrames ${s.nFrames}`);

  const trainCls = new Set(train.flatMap((s) => s.classes));
  const unseen = [...new Set(test.flatMap((s) => [...s.word]))].filter((c) => !trainCls.has(clsOf.get(c)));
  writeFileSync(resolve(outDir, `fold${f}.train.json`), JSON.stringify(trDoc));
  writeFileSync(resolve(outDir, `fold${f}.test.json`), JSON.stringify(teDoc));
  folds.push({
    fold: f, testWriter, trainWriters, trainLines: train.length, testLines: test.length,
    mean: Array.from(st.mean), sd: Array.from(st.sd), unseenTestChars: unseen,
  });
}

writeFileSync(resolve(outDir, "manifest.json"), JSON.stringify({
  format: "inknet-line-vectors/3",
  stride: STRIDE,
  vocabFile: "line_vocab.json (класс 0 = CTC blank, i>0 = vocab[i-1])",
  writers, folds,
  multiwordProb: MULTIWORD_PROB,
  rng: "splitmix32 (fix: mulberry32 при арифметических сидах давал почти одинаковые строки — датасеты v1/v2 ему заражены, цифры прошлых прогонов сравнимы только между собой)",
  limits: "v3: до 2 слов на строку; пробел = кадры переноса пера (moveStrokes — виртуальный штрих без чернил, " +
    "в растре не рисуется, метка — класс пробела); strokes — только чернильные штрихи для растрового baseline; " +
    "charSpans — границы символов в кадрах; строки синтезированы из глифов того же автора",
  note: "feats — z-scored Float32 плоская последовательность nFrames*stride кадров; метка classes — последовательность CTC-классов строки",
}, null, 2));

// ————— отчёт —————
console.log(`авторов: ${writers.join(", ")} | словарь слов: ${words.length}`);
for (const fr of folds) {
  console.log(`фолд ${fr.fold} (тест = ${fr.testWriter}): train строк ${fr.trainLines}, test ${fr.testLines}` +
    (fr.unseenTestChars.length ? ` | букв теста, невиданных в train: ${fr.unseenTestChars.join(" ")}` : ""));
}
console.log(`датасет → ${outDir}/fold<i>.{train,test}.json + manifest.json`);
