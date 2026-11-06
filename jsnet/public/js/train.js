// Страница «Тренировка почерка» — калибровка эталонов.
//  • модельная QC — nearest-neighbor по glyphEmbedding (js/qc.js), считается
//    локально и мгновенно: нет докачки ONNX, нет QC_WAIT_MS;
//  • до запуска тренировки QC молчит — сравнить ещё не с чем, первые эталоны
//    проходят только гейты мусора (calib-verify.js);
//  • остальной цикл 1-в-1: автосохранение после отрыва пера с grace на отмену
//    штриха, спорный эталон НЕ сохраняется молча (панель вердикта), сетевые
//    сбои — автоповторы + «Повторить отправку», «Переписать спорные» берёт
//    очередь из qc-флагов сервера.
import { $, requireMe, mountShell, CALIB_SET, REF_PER_CHAR } from "./shell.js";
import { PenCapture } from "./capture.js";
import { settings } from "./settings.js";
import { mountDrawingSettings } from "./pensettings.js";
import { junkCheck } from "./calib-verify.js";
import { NNQC, expectedRead } from "./qc.js";

const REF_RETRIES = 3;     // попыток отправить эталон при сетевом сбое

const $banner = $("#calib-banner");
const $charPrompt = $("#calib-char");
const $ghost = $("#ghost-char");
const $status = $("#status");
const $review = $("#calib-review");
const $reviewText = $("#calib-review-text");
const $rvSave = $("#btn-review-save");
const $rvRedo = $("#btn-review-redo");
const $rvRetry = $("#btn-review-retry");
const $progress = $("#progress");

let me = null;
let have = {};     // символ -> количество эталонов
let flagged = [];  // [{id, ch}] — спорные с сервера
let calib = null;  // {queue: [{ch, replaceId}], char, total, seq, busy, reviewStrokes, reviewQc}
let saveTimer = null;
let capture = null;
const qc = new NNQC();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function setStatus(t) { $status.textContent = t; }

// ── прогресс и карточка ──────────────────────────────────────────
async function loadSummary() {
  const res = await fetch("/api/refs/summary").catch(() => null);
  if (!res || !res.ok) return;
  const data = await res.json();
  have = data.have || {};
  flagged = Array.isArray(data.flagged) ? data.flagged : [];
  const done = Object.keys(have).filter((ch) => have[ch] >= (CALIB_SET.includes(ch) ? REF_PER_CHAR : 0)).length;
  $("#pf-meta").textContent =
    `эталонов: ${done} из ${CALIB_SET.length}` +
    (flagged.length ? ` · спорных: ${flagged.length} (${flagged.map((f) => f.ch).join(", ")})` : "");
  const $btnReview = $("#btn-review");
  $btnReview.hidden = !flagged.length || !!calib;
  $btnReview.textContent = `Переписать спорные (${flagged.length})`;
  renderProgress();
}

function renderProgress() {
  const cur = calib && calib.queue.length ? calib.queue[0].ch : null;
  $progress.innerHTML = [...CALIB_SET].map((ch) => {
    const n = have[ch] || 0;
    const cls = n >= REF_PER_CHAR ? "done" : n === 0 ? "todo" : "";
    const mark = ch === cur ? " active" : "";
    return `<span class="cchip ${cls}${mark}" data-ch="${ch}">${ch}<b>${n}/${REF_PER_CHAR}</b></span>`;
  }).join("");
  if (cur) {
    const el = $progress.querySelector(`[data-ch="${cur}"]`);
    if (el) el.scrollIntoView({ block: "nearest", inline: "center" });
  }
}

// ── очередь тренировки: недостающие символы, по одному эталону ────
function buildNeedQueue() {
  const need = {};
  for (const ch of CALIB_SET) need[ch] = Math.max(0, REF_PER_CHAR - (have[ch] || 0));
  const queue = [];
  for (let r = 0; r < REF_PER_CHAR; r++)
    for (const ch of CALIB_SET) if (need[ch]-- > 0) queue.push({ ch, replaceId: null });
  return queue;
}

function startCalibration(queue) {
  if (!queue.length) {
    setStatus(`Все ${CALIB_SET.length} символов уже откалиброваны по ${REF_PER_CHAR} эталону — дозаполнять нечего`);
    return;
  }
  calib = { queue, char: queue[0].ch, total: queue.length, seq: 0, busy: false, reviewStrokes: null, reviewQc: null };
  if (!capture) capture = new PenCapture($("#pad"), { onStrokeEnd: scheduleSave, settings, showGrid: () => true });
  capture.clear();
  qc.load(true); // свежие эталонные описания — QC сравнивает с текущим набором
  hideReview();
  showCalibPrompt();
  renderProgress();
}

function showCalibPrompt() {
  $charPrompt.textContent = calib.char;
  $ghost.textContent = calib.char;
  $ghost.hidden = false;
  $banner.hidden = false;
  const what = calib.queue[0].replaceId ? "переписываем спорный" : "осталось";
  setStatus(`Тренировка: ${what} ${calib.queue.length} из ${calib.total}. Напишите «${calib.char}»`);
}

// ── автосохранение эталона после отрыва пера ──────────────────────
function scheduleSave() {
  if (!calib) return;
  clearTimeout(saveTimer);
  hideReview();
  calib.seq += 1; // новый штрих — прежняя незавершённая проверка неактуальна
  const seq = calib.seq;
  // настраиваемая пауза autoDelay (панель «Перо»), читается на
  // каждый штрих — многоштриховые символы (п, ш, 4) требуют паузы между штрихами,
  // «мгновенно» ускоряет сдачу, но не даст отменить штрих до автосохранения
  const grace = Math.max(0, Number(settings.autoDelay) || 0);
  saveTimer = setTimeout(() => attemptSave(seq), grace);
}

async function attemptSave(seq) {
  if (!calib || seq !== calib.seq || calib.busy) return;
  const strokes = capture.strokes;
  if (!strokes.length) return;
  const pad = $("#pad");
  const junk = junkCheck(strokes, Math.round(pad.clientWidth), Math.round(pad.clientHeight));
  if (junk) {
    setStatus(`Проверка: ${junk}`);
    return;
  }
  calib.busy = true;
  const read = qc.read(strokes); // null — QC недоступна (нет эталонов): пишем без проверки
  if (read && read.char !== expectedRead(calib.char)) {
    calib.busy = false;
    if (seq !== calib.seq) return;
    openReview("mismatch", strokes, read.char,
      `Проверка читает это как «${read.char}», а задание — «${calib.char}». ` +
      `Если написали именно «${calib.char}» — сохраняйте (эталон помечается спорным), иначе перепишите.`);
    return;
  }
  await postRef(strokes, null, seq);
}

function openReview(mode, strokes, qcText, text) {
  if (!calib) return;
  calib.busy = false;
  calib.reviewStrokes = strokes;
  calib.reviewQc = qcText || null;
  $review.classList.toggle("error", mode === "retry");
  $reviewText.textContent = text;
  $rvSave.hidden = mode !== "mismatch";
  $rvRetry.hidden = mode !== "retry";
  $rvRedo.hidden = false;
  $review.hidden = false;
}

function hideReview() {
  $review.hidden = true;
  if (calib) { calib.reviewStrokes = null; calib.reviewQc = null; }
}

// траектория по зафиксированному снимку штрихов (пока ждали ответ, холст мог измениться)
function flatten(strokes) {
  const xs = [], ys = [], dt = [];
  let prev = null;
  for (const s of strokes) for (const p of s) {
    xs.push(p.x); ys.push(p.y);
    dt.push(prev === null ? 16 : p.t - prev);
    prev = p.t;
  }
  return { xs, ys, dt };
}

async function postRef(strokes, qcFlag, seq = calib.seq) {
  if (!calib) return;
  calib.busy = true;
  hideReview();
  setStatus("Сохраняю эталон…");
  const item = calib.queue[0];
  const body = { ch: item.ch, ...flatten(strokes) };
  if (qcFlag) body.qc = qcFlag;
  if (item.replaceId) body.replaceId = item.replaceId;
  let ok = null;
  for (let i = 0; i < REF_RETRIES && !ok; i++) {
    if (i) await sleep(400 * 2 ** (i - 1));
    const res = await fetch("/api/refs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => null);
    if (res?.ok) ok = res;
    else if (res && res.status < 500) break; // 400/401/404 повторять бессмысленно
  }
  if (!ok) {
    openReview("retry", strokes, qcFlag,
      "Сервер не принял эталон — проверьте соединение. Штрихи на месте: «Повторить отправку» или перепишите.");
    return;
  }
  if (!calib || seq !== calib.seq) { calib.busy = false; return; }
  calib.queue.shift();
  calib.busy = false;
  await loadSummary();
  qc.load(true); // QC сразу видит новый эталон
  if (!calib.queue.length) return endCalibration(true);
  calib.char = calib.queue[0].ch;
  capture.clear();
  calib.seq += 1;
  showCalibPrompt();
}

async function endCalibration(saved) {
  clearTimeout(saveTimer);
  calib = null;
  $banner.hidden = true;
  $ghost.hidden = true;
  $("#btn-start").hidden = false;
  $("#btn-stop").hidden = true;
  if (capture) capture.clear();
  await loadSummary();
  if (!saved) return setStatus("Тренировка отменена");
  setStatus(flagged.length
    ? `Тренировка завершена. Спорных эталонов: ${flagged.length} — нажмите «Переписать спорные»`
    : "Тренировка завершена, все эталоны сохранены");
}

function cancelReview() {
  clearTimeout(saveTimer);
  if (calib) { calib.seq += 1; calib.busy = false; hideReview(); }
}

function undoStroke() {
  if (!calib) return;
  cancelReview();
  if (!capture || !capture.undoLast()) setStatus("Отменять нечего");
}

// ── кнопки ────────────────────────────────────────────────────────
$("#btn-start").onclick = () => { if (!calib) { $("#btn-start").hidden = true; $("#btn-stop").hidden = false; startCalibration(buildNeedQueue()); } };
$("#btn-review").onclick = () => {
  if (!calib) { $("#btn-start").hidden = true; $("#btn-stop").hidden = false; startCalibration(flagged.map((f) => ({ ch: f.ch, replaceId: f.id }))); }
};
$("#btn-stop").onclick = () => endCalibration(false);
$("#btn-undo").onclick = undoStroke;
$("#btn-clear").onclick = () => { cancelReview(); if (capture) capture.clear(); };
$rvSave.onclick = () => { if (calib?.reviewStrokes) postRef(calib.reviewStrokes, calib.reviewQc || "—"); };
$rvRetry.onclick = () => { if (calib?.reviewStrokes) postRef(calib.reviewStrokes, calib.reviewQc); };
$rvRedo.onclick = () => {
  if (!calib) return;
  cancelReview();
  if (capture) capture.clear();
  setStatus(`Перепишите «${calib.char}» — пишите поверх крупной подсказки`);
};
document.addEventListener("keydown", (e) => {
  if (calib && (e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    undoStroke();
  }
});

$("#btn-wipe").onclick = async () => {
  if (!confirm("Стереть все эталоны этого профиля? Датасет без них оскудеет.")) return;
  await fetch("/api/refs", { method: "DELETE" }).catch(() => null);
  have = {};
  await loadSummary();
};

// панель пера/холста (pensettings.js — байтовая копия): capture создаётся лениво
// при старте тренировки, поэтому панель получает позднюю обёртку redraw
const penPanel = mountDrawingSettings($("#pen-panel"), {
  capture: { redraw: () => capture?.redraw(), _setupDpr: () => capture?._setupDpr() },
});
$("#btn-pen").onclick = () => {
  $("#pen-panel").hidden = !$("#pen-panel").hidden;
  if (!$("#pen-panel").hidden) penPanel.render();
};

// ── старт: автор уже вошёл, гость уходит на вход ──────────────────
(async () => {
  me = await requireMe();
  if (!me) return;
  mountShell(me);
  await loadSummary();
  setStatus(`Автор «${me.username}». Нажмите «Начать тренировку».`);
})();
