// Веб-приложение inknet-hybrid: страницы тренировки почерка (калибровка эталонов),
// сбора датасета и статистики — всё над той же базой jsnet/db/app.db, из которую
// читает билдер датасета (scripts/build-line-vector-ds.mjs).
//
// Без внешних зависимостей: node:http + node:sqlite (server/db.js), Node >= 23.
// ВХОД ПО ИМЕНИ, без пароля и регистрации: имя — это автор написаний в датасете,
// сессия — cookie sid. Приложение локальное (пишут со своего планшета в своей
// сети), поэтому «защита автора» здесь равна cookie-сессии: подменить имя на
// чужой машине можно, и это осознанный размен простоты. Наружу — только за TLS
// (COOKIE_SECURE=1).
// API-контракты refs/train зафиксированы форматом датасета (написания из
// train_samples читает билдер); добавлены
// /api/refs/emb (векторные описания эталонов для QC «ближайшего соседа») и
// /api/stats (статистика датасета + готовность к make ds).
//
// QC написаний здесь делает НЕ CTC-модель (ONNX не возим), а nearest-neighbor
// по glyphEmbedding из src/ink-features.js — клиент вызывает /api/refs/emb и
// сравнивает своё написание с сохранёнными эталонами. Код признаков ОДИН И ТОТ
// ЖЕ файл, что у билдера, — рассинхрон представлений исключён.
import { createServer } from "node:http";
import { readFileSync, existsSync, statSync, createReadStream } from "node:fs";
import { resolve, extname, normalize, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { db, DB_PATH } from "./db.js";
import { splitTrajectory, buildGlyphVectors, glyphEmbedding } from "../src/ink-features.js";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url))); // jsnet/

const PORT = Number(process.env.PORT) || 3210;
const HOST = process.env.HOST || "127.0.0.1";
const SESSION_TTL_DAYS = 7;
const NAME_MIN = 2;
// гейты билдера датасета (scripts/build-line-vector-ds.mjs) — статистика честна
// ровно по тем же порогам, что реально участвуют в make ds
const MIN_PTS = 4;
const MIN_WRITER_CHARS = Number(process.env.MIN_WRITER_CHARS || 10);
const COOKIE_SECURE = process.env.COOKIE_SECURE === "1";
const cookieAttrs = (maxAge) =>
  `HttpOnly; SameSite=Lax; Path=/; Max-Age=${maxAge}${COOKIE_SECURE ? "; Secure" : ""}`;

// ————— мелкие хелперы —————
function parseCookies(req) {
  const out = {};
  for (const part of (req.headers.cookie || "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0) out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

function currentUser(req) {
  const token = parseCookies(req).sid;
  if (!token) return null;
  return db.prepare(
    `SELECT u.id, u.username, u.created_at, s.expires_at FROM sessions s
     JOIN users u ON u.id = s.user_id
     WHERE s.token = ? AND s.expires_at > datetime('now')`
  ).get(token) || null;
}

const normUsername = (raw) => String(raw || "").trim().replace(/\s+/g, " ").slice(0, 32);

function newSession(res, userId) {
  const token = randomUUID();
  db.prepare(
    `INSERT INTO sessions (token, user_id, expires_at)
     VALUES (?, ?, datetime('now', '+' || ? || ' days'))`
  ).run(token, userId, SESSION_TTL_DAYS);
  res.setHeader("Set-Cookie", `sid=${token}; ${cookieAttrs(SESSION_TTL_DAYS * 86400)}`);
}

function send(res, code, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json; charset=utf-8", "Content-Length": Buffer.byteLength(body) });
  res.end(body);
}

const okNumArray = (a) =>
  Array.isArray(a) && a.length > 0 && a.length <= 20000 && a.every((n) => Number.isFinite(n));

const validStrokes = (strokes) =>
  Array.isArray(strokes) && strokes.length > 0 && strokes.length <= 64 &&
  strokes.every((st) =>
    Array.isArray(st) && st.length >= 2 && st.length <= 20000 &&
    st.every((p) => p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.t)));

// ————— модель «ближайшего соседа» для QC: embedding-описания эталонов —————
// считаются на сервере тем же кодом (src/ink-features.js), что и при сборке
// датасета, — страница вердикта получает ровно то представление, которое
// потом увидит тренажёр векторной ветки
function refEmbeddings(userId) {
  const rows = db.prepare(
    "SELECT id, ch, xs, ys, dt, qc FROM refs WHERE user_id = ? ORDER BY id"
  ).all(userId);
  const out = [];
  for (const r of rows) {
    if (r.qc) continue; // спорные/мусорные эталоны в QC не участвуют (как в билдере)
    let strokes;
    try { strokes = splitTrajectory(JSON.parse(r.xs), JSON.parse(r.ys), JSON.parse(r.dt)); }
    catch { continue; }
    const pts = strokes.reduce((a, s) => a + s.length, 0);
    if (pts < MIN_PTS) continue;
    out.push({ id: r.id, ch: String(r.ch).toLowerCase(), emb: Array.from(glyphEmbedding(buildGlyphVectors(strokes))) });
  }
  return out;
}

// ————— статистика датасета: покрытие авторов/букв + готовность к make ds —————
// считаем по тем же правилам, что build-line-vector-ds.mjs: qc-флаг отсекается,
// написание короче MIN_PTS не считается, буква должна быть в vocab, автор с
// покрытием < MIN_WRITER_CHARS в фолды не попадает; буквы пула — нижний регистр
function datasetStats() {
  const vocab = JSON.parse(readFileSync(resolve(ROOT, "public/models/line_vocab.json"), "utf8"));
  const inVocab = new Set(vocab);
  const terms = JSON.parse(readFileSync(resolve(ROOT, "public/terms.json"), "utf8"));
  const wordPool = new Set();
  for (const list of [terms.terms_ru, terms.terms_en]) for (const w of list || [])
    for (const tok of String(w).toLowerCase().split(/[\s\-–—,.()]+/))
      if (tok.length >= 2 && tok.length <= 10 && [...tok].every((c) => inVocab.has(c)))
        wordPool.add(tok);
  const words = [...wordPool];

  const writers = [];
  const charWriters = new Map(); // char(нижн.) -> Set(username)
  const charSamples = new Map(); // char(нижн.) -> количество написаний
  let refs = 0, refsClean = 0, refsFlagged = 0, trainTotal = 0, trainClean = 0;

  const users = db.prepare("SELECT id, username, consent, created_at FROM users ORDER BY id").all();
  for (const u of users) {
    const pool = new Map(); // char -> n
    const add = (ch, n) => {
      ch = String(ch).toLowerCase();
      if (!inVocab.has(ch) || n < MIN_PTS) return;
      pool.set(ch, (pool.get(ch) || 0) + 1);
    };
    let uRefs = 0, uClean = 0, uFlag = 0;
    for (const r of db.prepare("SELECT ch, xs, ys, dt, qc FROM refs WHERE user_id = ?").all(u.id)) {
      refs++; uRefs++;
      if (r.qc) { refsFlagged++; uFlag++; continue; }
      let strokes;
      try { strokes = splitTrajectory(JSON.parse(r.xs), JSON.parse(r.ys), JSON.parse(r.dt)); }
      catch { continue; }
      const ok = strokes.reduce((a, s) => a + s.length, 0) >= MIN_PTS;
      if (ok) { refsClean++; uClean++; add(r.ch, strokes.reduce((a, s) => a + s.length, 0)); }
    }
    let uTrain = 0, uTrainClean = 0;
    for (const r of db.prepare("SELECT ch, strokes FROM train_samples WHERE user_id = ?").all(u.id)) {
      trainTotal++; uTrain++;
      let st;
      try { st = JSON.parse(r.strokes); } catch { continue; }
      const pts = st.reduce((a, s) => a + (s?.length || 0), 0);
      if (pts >= MIN_PTS) { trainClean++; uTrainClean++; add(r.ch, pts); }
    }
    for (const [ch, n] of pool) {
      if (!charWriters.has(ch)) { charWriters.set(ch, new Set()); charSamples.set(ch, 0); }
      charWriters.get(ch).add(u.username);
      charSamples.set(ch, charSamples.get(ch) + n);
    }
    const inDs = pool.size >= MIN_WRITER_CHARS;
    const wordsAvail = inDs ? words.filter((w) => [...w].every((c) => pool.has(c))).length : 0;
    const last = db.prepare(
      "SELECT MAX(created_at) AS t FROM (SELECT created_at FROM refs WHERE user_id = ?" +
      " UNION ALL SELECT created_at FROM train_samples WHERE user_id = ?)"
    ).get(u.id, u.id).t;
    if (uRefs || uTrain)
      writers.push({
        username: u.username, createdAt: u.created_at, consent: !!u.consent,
        refs: uRefs, refsClean: uClean, refsFlagged: uFlag,
        train: uTrain, trainClean: uTrainClean,
        chars: pool.size, words: wordsAvail, inDs, lastActivity: last,
      });
  }
  writers.sort((a, b) => b.chars - a.chars || b.refsClean - a.refsClean || a.username.localeCompare(b.username));

  // буквы кириллицы из vocab + цифры 1–9 (план сбора) — по ним показываем покрытие;
  // латиницуvocab содержит (для terms_en), но писать её некому — вне плана
  const cyr = vocab.filter((c) => /[а-яё]/i.test(c)).map((c) => c.toLowerCase()).sort();
  const digits = vocab.filter((c) => /[0-9]/.test(c));
  const cov = (list) => list.map((ch) => ({
    ch, writers: charWriters.get(ch)?.size || 0, samples: charSamples.get(ch) || 0,
  }));
  const inDsCount = writers.filter((w) => w.inDs).length;
  return {
    format: "inknet-ds-stats/1",
    generatedAt: new Date().toISOString(),
    gates: { minWriterChars: MIN_WRITER_CHARS, minPts: MIN_PTS },
    totals: {
      users: users.length, writersWithData: writers.length, writersInDs: inDsCount,
      refs, refsClean, refsFlagged, trainSamples: trainTotal, trainClean,
      cyrCovered: cyr.filter((c) => (charWriters.get(c)?.size || 0) > 0).length,
      cyrTotal: cyr.length,
      digitsCovered: digits.filter((c) => (charWriters.get(c)?.size || 0) > 0).length,
      digitsTotal: digits.length,
    },
    writers,
    charsCyr: cov(cyr),
    charsDigits: cov(digits),
    note: "автор попадает в make ds при покрытии >= gates.minWriterChars букв " +
      "(qc-отклонённые и короче gates.minPts точек не считаются; ЗАГЛАВНЫЕ считаются вместе со строчными)",
  };
}

// ————— маршруты API —————
const routes = [];
const route = (method, path, handler) => routes.push({ method, path, handler });

// поиск автора по имени: COLLATE NOCASE в SQLite складывает только латиницу,
// для кириллицы «Инна» и «инна» — разные строки, и автор заводится второй
// (это ломает writer-disjoint: один почерк считается двумя авторами). Поэтому
// регистр нормализуем в JS; база маленькая (десятки имён), скан дешёвый
function findUserByName(name) {
  const key = name.toLowerCase();
  return db.prepare("SELECT id, username, created_at FROM users").all()
    .find((u) => u.username.toLowerCase() === key) || null;
}

// вход по имени: известный автор — садим в сессию, нового создаём сразу
// (паролей нет; consent=1 — форма входа открыто говорит, что написания идут
// в обучающий датасет; уже заведённым авторам существующее значение поля не трогаем)
route("POST", "/api/login", (req, res) => {
  const username = normUsername(req.body?.username);
  if (username.length < NAME_MIN)
    return send(res, 400, { error: `Имя — от ${NAME_MIN} символов` });
  let user = findUserByName(username);
  if (!user) {
    try {
      // pass_hash не используется (колонка NOT NULL)
      const info = db.prepare("INSERT INTO users (username, pass_hash, consent) VALUES (?, '', 1)").run(username);
      user = db.prepare("SELECT id, username, created_at FROM users WHERE id = ?").get(info.lastInsertRowid);
    } catch (err) {
      if (!/UNIQUE/i.test(String(err?.message))) { console.error(err); return send(res, 500, { error: "Ошибка сервера" }); }
      user = findUserByName(username); // гонка двух вкладок с новым именем
    }
  }
  newSession(res, user.id);
  send(res, 200, { user });
});

route("POST", "/api/logout", (req, res) => {
  const token = parseCookies(req).sid;
  if (token) db.prepare("DELETE FROM sessions WHERE token = ?").run(token);
  res.setHeader("Set-Cookie", "sid=; " + cookieAttrs(0));
  send(res, 200, { ok: true });
});

route("GET", "/api/me", (req, res) => {
  const user = currentUser(req);
  if (!user) return send(res, 401, { error: "не авторизован" });
  send(res, 200, { user: { id: user.id, username: user.username, created_at: user.created_at } });
});

// имена уже заведённых авторов — подсказка на входе (публично: форму видит гость):
// опечатка создаёт нового автора с пустым датасетом, а writer-disjoint фолды
// считают авторов по id в users
route("GET", "/api/authors", (req, res) => {
  const authors = db.prepare("SELECT username FROM users ORDER BY username").all().map((r) => r.username);
  send(res, 200, { authors });
});

// {have: {буква: count}, total, flagged} — прогресс тренировки и очередь спорных
route("GET", "/api/refs/summary", (req, res) => {
  const me = currentUser(req);
  if (!me) return send(res, 401, { error: "не авторизован" });
  const rows = db.prepare("SELECT ch, COUNT(*) AS c FROM refs WHERE user_id = ? GROUP BY ch").all(me.id);
  const have = {};
  let total = 0;
  for (const r of rows) { have[r.ch] = r.c; total += r.c; }
  const flagged = db.prepare(
    "SELECT id, ch FROM refs WHERE user_id = ? AND qc IS NOT NULL AND qc <> '' ORDER BY id"
  ).all(me.id);
  send(res, 200, { have, total, flagged });
});

// эталоны автора с embedding-описаниями — QC «ближайшего соседа» на странице
route("GET", "/api/refs/emb", (req, res) => {
  const me = currentUser(req);
  if (!me) return send(res, 401, { error: "не авторизован" });
  send(res, 200, { refs: refEmbeddings(me.id) });
});

// сохранить эталон: символ + траектория (xs/ys/dt из capture.getTrajectory);
// qc — вердикт проверки ('' / NULL = прочитано верно), replaceId — заменяемый спорный
route("POST", "/api/refs", (req, res) => {
  const me = currentUser(req);
  if (!me) return send(res, 401, { error: "не авторизован" });
  const { ch, xs, ys, dt, qc, replaceId } = req.body || {};
  if (typeof ch !== "string" || ch.length !== 1)
    return send(res, 400, { error: "нужен один символ (ch)" });
  if (!okNumArray(xs) || !okNumArray(ys) || !okNumArray(dt) ||
      xs.length !== ys.length || xs.length !== dt.length)
    return send(res, 400, { error: "некорректная траектория" });
  if (qc != null && typeof qc !== "string")
    return send(res, 400, { error: "qc — строка или null" });
  const qcVal = typeof qc === "string" && qc.trim() ? qc.trim().slice(0, 16) : null;
  let old = null;
  if (replaceId != null) {
    old = db.prepare("SELECT id FROM refs WHERE id = ? AND user_id = ?").get(Number(replaceId), me.id);
    if (!old) return send(res, 404, { error: "заменяемый эталон не найден" });
  }
  db.exec("BEGIN");
  try {
    const info = db.prepare("INSERT INTO refs (user_id, ch, xs, ys, dt, qc) VALUES (?, ?, ?, ?, ?, ?)")
      .run(me.id, ch, JSON.stringify(xs), JSON.stringify(ys), JSON.stringify(dt), qcVal);
    if (old) db.prepare("DELETE FROM refs WHERE id = ?").run(old.id);
    db.exec("COMMIT");
    const { c } = db.prepare("SELECT COUNT(*) AS c FROM refs WHERE user_id = ?").get(me.id);
    send(res, 200, { ok: true, id: Number(info.lastInsertRowid), replaced: old ? old.id : null, total: c });
  } catch (err) {
    db.exec("ROLLBACK");
    console.error(err);
    send(res, 500, { error: "Ошибка сервера" });
  }
});

route("DELETE", "/api/refs", (req, res) => {
  const me = currentUser(req);
  if (!me) return send(res, 401, { error: "не авторизован" });
  db.prepare("DELETE FROM refs WHERE user_id = ?").run(me.id);
  send(res, 200, { ok: true });
});

// прогресс сбора автора для полосы чипов
route("GET", "/api/train/summary", (req, res) => {
  const me = currentUser(req);
  if (!me) return send(res, 401, { error: "не авторизован" });
  const rows = db.prepare("SELECT ch, COUNT(*) AS c FROM train_samples WHERE user_id = ? GROUP BY ch").all(me.id);
  const have = {};
  let total = 0;
  for (const r of rows) { have[r.ch] = r.c; total += r.c; }
  send(res, 200, { have, total });
});

// сдать сессию: партия написаний [{char, k, strokes, startedAt}] + метаданные устройства
route("POST", "/api/train", (req, res) => {
  const me = currentUser(req);
  if (!me) return send(res, 401, { error: "не авторизован" });
  const { samples, device } = req.body || {};
  if (!Array.isArray(samples) || !samples.length || samples.length > 80)
    return send(res, 400, { error: "нужны написания (samples)" });
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (typeof s.char !== "string" || s.char.length !== 1 || !validStrokes(s.strokes)) {
      // с индексом, символом и счётом битых точек: без деталей «один брак
      // отклоняет весь батч» было нечем диагностировать (на планшете видно
      // только «сервер не принял сессию», см. NaN от coalesced-событий на Android)
      const st = Array.isArray(s.strokes) ? s.strokes : [];
      const bad = st.reduce((a, pts) => a + (Array.isArray(pts)
        ? pts.filter((p) => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.t)).length
        : 0), 0);
      const name = typeof s.char === "string" && s.char.length === 1 ? ` «${s.char}»` : "";
      console.warn(`[api/train] ${me.username}: отклонено написание №${i + 1}${name} (штрихов: ${st.length}, неконечных точек: ${bad})`);
      return send(res, 400, {
        error: `некорректное написание в сессии (№${i + 1}${name}, штрихов: ${st.length}, битых точек: ${bad})`,
      });
    }
  }
  const dev = device ? JSON.stringify(device) : null;
  const ins = db.prepare(
    "INSERT INTO train_samples (user_id, ch, k, strokes, started_at, device) VALUES (?, ?, ?, ?, ?, ?)"
  );
  for (const s of samples)
    ins.run(me.id, s.char, Number(s.k) || 0, JSON.stringify(s.strokes),
      Number.isFinite(s.startedAt) ? s.startedAt : null, dev);
  const { c } = db.prepare("SELECT COUNT(*) AS c FROM train_samples WHERE user_id = ?").get(me.id);
  send(res, 200, { ok: true, accepted: samples.length, total: c });
});

// выгрузка всех написаний автора в формате inknet-collect/1
route("GET", "/api/train/export", (req, res) => {
  const me = currentUser(req);
  if (!me) return send(res, 401, { error: "не авторизован" });
  const rows = db.prepare(
    "SELECT ch, k, strokes, started_at, device, created_at FROM train_samples WHERE user_id = ? ORDER BY id"
  ).all(me.id);
  let device = null;
  const samples = rows.map((r) => {
    if (!device && r.device) { try { device = JSON.parse(r.device); } catch { /* ignore */ } }
    return { char: r.ch, k: r.k, strokes: JSON.parse(r.strokes), startedAt: r.started_at, submittedAt: r.created_at };
  });
  send(res, 200, {
    format: "inknet-collect/1", writer: me.username,
    exportedAt: new Date().toISOString(), device, samples,
  });
});

// статистика датасета (для страницы stats.html)
route("GET", "/api/stats", (req, res) => {
  const me = currentUser(req);
  if (!me) return send(res, 401, { error: "не авторизован" });
  send(res, 200, datasetStats());
});

// ————— статика: public/ + src/ (общие экстракторы — один файл с билдером) —————
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  // распознавание строк: бинарные артефакты модели и рантайма ONNX
  ".onnx": "application/octet-stream",
  ".ort": "application/octet-stream",
  ".wasm": "application/wasm",
};
function serveStatic(req, res, urlPath) {
  const root = urlPath.startsWith("/src/")
    ? resolve(ROOT, "src")
    : resolve(ROOT, "public");
  const rel = urlPath.startsWith("/src/") ? urlPath.slice(5) : urlPath === "/" ? "/index.html" : urlPath;
  const file = resolve(root, "." + sep + normalize(rel).replace(/^([/\\])+/, ""));
  if (!file.startsWith(root + sep) && file !== root) { res.writeHead(403); return res.end(); }
  if (!existsSync(file) || !statSync(file).isFile()) { res.writeHead(404); return res.end("not found"); }
  // Content-Length обязателен: браузер по нему считает прогресс докачки модели (~18 МБ),
  // без него страница распознавания висит на «загрузка модели» без индикатора
  res.writeHead(200, {
    "Content-Type": MIME[extname(file)] || "application/octet-stream",
    "Content-Length": statSync(file).size,
    "Cache-Control": "no-cache",
  });
  createReadStream(file).pipe(res);
}

// ————— диспетчер —————
const server = createServer(async (req, res) => {
  const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
  try {
    for (const r of routes) {
      if (r.method === req.method && r.path === urlPath) {
        if (req.method === "POST") {
          // тело копится БУФЕРОМ и декодируется один раз: конкатенация строк
          // («body += chunk») расщепляет многобайтовый UTF-8 на границе сетевого
          // чанка — кириллица в JSON ломалась, и большая посылка с планшета
          // падала с «некорректный JSON» («сервер не принял сессию»)
          const chunks = [];
          let size = 0;
          let tooBig = false;
          req.on("data", (c) => {
            size += c.length;
            if (size > 8_000_000) { tooBig = true; res.writeHead(413); req.destroy(); }
            else chunks.push(c);
          });
          await new Promise((done) => { req.on("end", done); req.on("close", done); });
          if (tooBig) return;
          const raw = chunks.length ? Buffer.concat(chunks).toString("utf8") : "";
          try { req.body = raw ? JSON.parse(raw) : {}; }
          catch { return send(res, 400, { error: "некорректный JSON" }); }
        }
        return r.handler(req, res);
      }
    }
    if (req.method === "GET") return serveStatic(req, res, urlPath);
    res.writeHead(404);
    res.end("not found");
  } catch (err) {
    console.error(err);
    if (!res.headersSent) send(res, 500, { error: "Ошибка сервера" });
  }
});

server.listen(PORT, HOST, () => {
  console.log(`веб-приложение: http://${HOST}:${PORT} · база: ${DB_PATH}`);
  console.log(`страницы: / (вход) · /train.html · /collect.html · /stats.html · /lines.html (распознавание)`);
});
