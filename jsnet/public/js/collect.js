// Страница «Сбор датасета» — плановый сбор написаний: план символов,
// очередь «кому меньше всех», буфер несданной сессии в localStorage,
// сдача на сервер, выгрузка inknet-collect/1.
// QC «как это читает модель» — nearest-neighbor по эталонам автора (js/qc.js):
// предупреждение, не блокировка — на тренировке спорное решает панель вердикта.
import { $, requireMe, mountShell, plural, LETTERS, UPPER, DIGITS, HARD, TARGET, HARD_TARGET, SESSION_SIZE, IS_UPPER } from "./shell.js";
import { PenCapture } from "./capture.js";
import { settings } from "./settings.js";
import { mountDrawingSettings } from "./pensettings.js";
import { NNQC, expectedRead } from "./qc.js";

const ALL_CHARS = [...LETTERS, ...UPPER, ...DIGITS];
const EXPORT_FORMAT = "inknet-collect/1";

const taskTextEl = $("#task-text");
const taskChipEl = $("#task-chip");
const ghostCharEl = $("#ghost-char");
const progressEl = $("#progress");
const sessionInfoEl = $("#session-info");

const capture = new PenCapture($("#pad"), { settings });

let me = null;
let serverHave = {}; // {char: n} — уже сданное на сервере
let queue = [];      // очередь символов текущей сдачи
let sample = null;   // {char, k} текущего задания
let startedAt = 0;   // performance.now() показа задания

function setStatus(t) { taskTextEl.textContent = t; }

// --- буфер несданной сессии (localStorage) + прогресс = сервер + буфер ---
const bufferKey = () => `collect-buffer-${me.id}`;

function buffer() {
  try {
    const v = JSON.parse(localStorage.getItem(bufferKey()));
    return Array.isArray(v) ? v : [];
  } catch { return []; }
}

function saveBuffer(samples) {
  try {
    localStorage.setItem(bufferKey(), JSON.stringify(samples));
    return true;
  } catch {
    setStatus("Не удалось сохранить сессию в браузер (кончилось место) — сдайте сессию кнопкой.");
    return false;
  }
}

function progressMap() {
  const prog = { ...serverHave };
  for (const s of buffer()) prog[s.char] = (prog[s.char] || 0) + 1;
  return prog;
}

const targetFor = (ch) => (HARD.has(ch) ? HARD_TARGET : TARGET);

// очередь сдачи: сначала символы с наименьшим числом написов, вперемешку
function buildQueue() {
  const prog = progressMap();
  const under = ALL_CHARS.filter((ch) => (prog[ch] || 0) < targetFor(ch));
  under.sort(() => Math.random() - 0.5);
  under.sort((a, b) => (prog[a] || 0) - (prog[b] || 0));
  return under.slice(0, SESSION_SIZE);
}

// --- задания ---
function nextTask() {
  if (!queue.length) queue = buildQueue();
  if (!queue.length) {
    sample = null;
    ghostCharEl.hidden = true;
    taskChipEl.hidden = true;
    capture.clear();
    const n = buffer().length;
    setStatus(n
      ? `Цель автора закрыта. Сдайте сессию: «сдать сессию» (${n} ${plural(n, "образец", "образца", "образцов")}).`
      : "Цель сбора по этому автору закрыта (2 написания на символ, 4 — на путающие пары).");
    renderProgress();
    return;
  }
  const ch = queue[0];
  const prog = progressMap();
  const written = prog[ch] || 0;
  sample = { char: ch, k: written + 1 };
  const upper = IS_UPPER(ch);
  taskChipEl.textContent = ch;
  taskChipEl.hidden = false;
  taskChipEl.title = HARD.has(ch) ? "путающая пара — пишите медленно и разборчиво"
    : upper ? "заглавная (большая) буква" : "";
  ghostCharEl.textContent = ch;
  ghostCharEl.hidden = false;
  capture.clear();
  startedAt = performance.now();
  setStatus(`Напишите ${upper ? "ЗАГЛАВНУЮ " : ""}«${ch}» — написание ${written + 1} из ${targetFor(ch)}${HARD.has(ch) ? " (путающая пара)" : ""}`);
  renderProgress();
}

// --- сохранение написания ---
function roundedStrokes() {
  const strokes = capture.strokes.map((s) =>
    s.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y), t: Math.round(p.t) })));
  if (capture.drawing && capture.points.length)
    strokes.push(capture.points.map((p) => ({ x: Math.round(p.x), y: Math.round(p.y), t: Math.round(p.t) })));
  return strokes;
}

function saveCurrentSample() {
  if (!sample) return setStatus("Нет активного задания — начните заново («Убрать символ»).");
  if (capture.drawing) return setStatus("Оторвите перо и нажмите «ГОТОВО».");
  const strokes = roundedStrokes();
  const total = strokes.reduce((a, s) => a + s.length, 0);
  if (!strokes.length || total < 4)
    return setStatus("Слишком короткое написание (< 4 точек) — билдер датасета такое отбрасывает. Перепишите.");
  const { char } = sample;
  const prog = progressMap();
  const s = { char, k: (prog[char] || 0) + 1, startedAt, strokes };
  const buf = buffer();
  buf.push(s);
  if (!saveBuffer(buf)) return;
  queue = queue.filter((c) => c !== char);
  setStatus(`Сохранено «${char}» (${s.k} из ${targetFor(char)}).`);
  renderProgress();
  checkWithModel(s);
  nextTask();
}

function dropLastSample() {
  const buf = buffer();
  if (!buf.length) return setStatus("В этой сессии ещё нет сохранённых написаний.");
  const last = buf.pop();
  saveBuffer(buf);
  setStatus(`«${last.char}» убрано из сессии — можно переписать.`);
  renderProgress();
  if (!sample) nextTask();
}

// --- сервер: прогресс и сдача сессии ---
async function loadSummary() {
  const res = await fetch("/api/train/summary").catch(() => null);
  if (!res || !res.ok) return;
  serverHave = (await res.json()).have || {};
}

// Android-специфичная санитизация перед сдачей: Chromium в getCoalescedEvents
// подмешивает события с NaN-координатами (JSON.stringify превращает NaN в null),
// а pointercancel оставляет штрих из одной точки. Серверная валидация из-за
// такого образца отклоняет ВЕСЬ батч, и сдача вставала «сдано 0» намертво —
// битые точки/штрихи вычищаем здесь, нечитаемые написания убираем из буфера
// с явным списком, чтобы их переписать.
const validPoint = (p) => p && Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.t);

function sanitizeSample(s) {
  if (!s || typeof s.char !== "string" || s.char.length !== 1 || !Array.isArray(s.strokes)) return null;
  const strokes = s.strokes
    .map((st) => (Array.isArray(st) ? st.filter(validPoint) : []))
    .filter((st) => st.length >= 2);
  if (!strokes.length || strokes.length > 64) return null;
  return { ...s, strokes };
}

async function exportSession() {
  if (!buffer().length) return setStatus("Сессия пуста — сначала напишите символы.");
  // до нарезки пачками: один брак в буфере глушил всю сдачу (400 на весь батч)
  const kept = [], dropped = [];
  for (const s of buffer()) {
    const c = sanitizeSample(s);
    if (c) kept.push(c);
    else dropped.push(s && typeof s.char === "string" && s.char.length === 1 ? s.char : "?");
  }
  if (kept.length !== buffer().length) saveBuffer(kept);
  if (!kept.length)
    return setStatus(`Все написания в буфере нечитаемые (${dropped.join(", ")}) — убраны, перепишите их.`);
  setStatus("Отправляю сессию на сервер…");
  const device = {
    ua: navigator.userAgent,
    dpr: window.devicePixelRatio || 1,
    screen: [screen.width, screen.height],
    canvas: [Math.round($("#pad").clientWidth), Math.round($("#pad").clientHeight)],
    pen: settings.pen, // провенанс; на датасет не влияет — растр рисуется серверным PEN_DEFAULTS
  };
  // сервер принимает ≤ 80 написаний на запрос; несданный буфер может накопить
  // больше между сессиями — выдаём пачками, сданное сразу убираем из буфера,
  // чтобы при обрыве повторная сдача не задвоивала уже принятые написания
  let sent = 0, total = 0;
  for (;;) {
    const buf = buffer();
    if (!buf.length) break;
    const part = buf.slice(0, 80);
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ samples: part, device }),
    }).catch(() => null);
    if (!res) {
      setStatus(`Сервер не отвечает — сдано ${sent}, остальные написания останутся в буфере браузера.`);
      return;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      // без текста причины пользователь получал абстрактное «не принял» и не
      // понимал: сесть переписать, или перезайти, или буфер слишком большой
      const why = res.status === 401 ? "сессия входа истекла — обновите страницу и войдите"
        : data.error || `HTTP ${res.status}`;
      setStatus(`Сервер не принял сессию (${why}) — сдано ${sent}, остальное в буфере браузера.`);
      return;
    }
    sent += data.accepted ?? part.length;
    total = data.total;
    saveBuffer(buffer().slice(part.length));
  }
  await loadSummary();
  setStatus(`Сдано ${sent} написаний (всего у автора: ${total}).`
    + (dropped.length ? ` Нечитаемые убраны из буфера: ${dropped.join(", ")} — перепишите.` : ""));
  renderProgress();
}

// выгрузка несданной части файлом — формат inknet-collect/1
function downloadJSON() {
  const buf = buffer();
  if (!buf.length) return setStatus("Несданной части сессии нет — все написания уже на сервере.");
  const data = {
    format: EXPORT_FORMAT,
    writer: me.username,
    exportedAt: new Date().toISOString(),
    consent: true,
    device: {
      ua: navigator.userAgent,
      dpr: window.devicePixelRatio || 1,
      screen: [screen.width, screen.height],
      canvas: [Math.round($("#pad").clientWidth), Math.round($("#pad").clientHeight)],
      pen: settings.pen,
    },
    samples: buf,
  };
  const stamp = new Date().toISOString().slice(0, 16).replace(/[T:]/g, "-");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: "application/json" }));
  a.download = `inknet-collect-${me.username}-${stamp}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// --- предупреждающая QC: «как это читает ближайший сосед к эталонам» ---
const qc = new NNQC();

function checkWithModel(s) {
  try {
    const read = qc.read(s.strokes);
    if (!read) return; // нет эталонов — сравнения нет
    if (read.char !== expectedRead(s.char)) {
      setStatus(`⚠ Проверка читает это как «${read.char}» — если написали не «${s.char}», нажмите «Убрать символ».`);
      const chip = progressEl.querySelector(`[data-ch="${s.char}"]`);
      if (chip) chip.classList.add("review");
    }
  } catch { /* QC упала — не критично для сбора */ }
}

// --- панель прогресса (горизонтальная полоса чипов, скролл к активному) ---
function renderProgress() {
  const prog = progressMap();
  const cur = sample && sample.char;
  progressEl.innerHTML = ALL_CHARS.map((ch) => {
    const n = prog[ch] || 0;
    const cls = n >= targetFor(ch) ? "done" : n === 0 ? "todo" : "";
    const mark = ch === cur ? " active" : "";
    const hard = HARD.has(ch) ? "*" : "";
    const cap = IS_UPPER(ch) ? " (заглавная)" : "";
    return `<span class="cchip ${cls}${mark}" data-ch="${ch}" title="${HARD.has(ch) ? "путающая пара, цель " + HARD_TARGET : "цель " + TARGET}${cap}">${ch}${hard}<b>${n}/${targetFor(ch)}</b></span>`;
  }).join("");
  if (cur) {
    const el = progressEl.querySelector(`[data-ch="${cur}"]`);
    if (el) el.scrollIntoView({ block: "nearest", inline: "center" });
  }
  const done = ALL_CHARS.filter((ch) => (prog[ch] || 0) >= targetFor(ch)).length;
  sessionInfoEl.textContent = `сессия ${buffer().length}/${SESSION_SIZE} · символов закрыто ${done}/${ALL_CHARS.length}`;
}

// --- кнопки ---
$("#btn-done").onclick = saveCurrentSample;
$("#btn-undo").onclick = () => { if (!capture.undoLast()) setStatus("Отменять нечего."); };
$("#btn-clear").onclick = () => {
  capture.clear();
  startedAt = performance.now();
  setStatus("Холст чист — пишите заново.");
};
$("#btn-drop").onclick = dropLastSample;
$("#btn-json").onclick = downloadJSON;
$("#btn-export").onclick = exportSession;

window.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    if (!capture.undoLast()) setStatus("Отменять нечего.");
  }
});

// панель пера/холста (pensettings.js — байтовая копия) — та же, что на тренировке
const penPanel = mountDrawingSettings($("#pen-panel"), { capture });
$("#btn-pen").onclick = () => {
  $("#pen-panel").hidden = !$("#pen-panel").hidden;
  if (!$("#pen-panel").hidden) penPanel.render();
};

// --- старт: автор уже вошёл, гость уходит на вход ---
(async () => {
  const meRes = await requireMe();
  if (!meRes) return;
  me = meRes;
  mountShell(me);
  await loadSummary();
  setStatus(`Автор «${me.username}».`);
  nextTask();
  qc.load(); // эталоны автора для QC подгружаются фоном; без них сбор работает как обычно
})();
