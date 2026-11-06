// Страница «Распознавание строк»: 1–3 строки подряд на холсте, CTC-модель
// (models/line_int8.onnx) в Web Worker, автоматическая нарезка на строки.
// Словарная правка из интерфейса убрана; словарь терминов остался только
// внутри воркера (мягкий бонус beam search).
// Интеграция с jsnet (пайплайн 1-в-1):
//  • вход — shell.js: гость улетает на "/", страница работает под автором;
//  • модель фиксирована (models/line_int8.onnx): /api/config в jsnet нет,
//    fetch оставлен и молча даёт дефолт при 404 — клиентский контракт сохранён;
//  • подсказка калибровки ведёт на /train.html (у jsnet нет profile.html),
//    набор символов/эталонов — общие константы shell.js.
import { $, requireMe, mountShell, CALIB_SET, REF_PER_CHAR } from "./shell.js";
import { PenCapture } from "./capture.js";
import { settings } from "./settings.js";
import { buildVocab } from "./dict.js";
import { packStrokes } from "./line-preprocess.js";

const canvas = $("#pad");
const statusEl = $("#app-status");
const linesEl = $("#lines-out");
const metaEl = $("#lines-meta");

const state = {
  vocab: null, // словарь терминов [{orig, norm, chars}]
  alpha: null, // алфавит модели из line_vocab.json (индекс+1 = класс CTC)
  lines: [], // [{text, conf, cands, corrected}]
  ms: null, // длительность последнего распознавания — «время · средний балл» в панели строк
  busy: false,
};

// --- CTC-воркер (загрузка с прогрессом, кэш Cache Storage на localhost) ---
const worker = new Worker("/js/line.worker.js", { type: "module" });
const pending = new Map();
let reqId = 0;
worker.onmessage = (e) => {
  const { id, type } = e.data;
  if (type === "progress") {
    const pct = Math.round((e.data.loaded / Math.max(e.data.total, 1)) * 100);
    setStatus(`Загружаю строчную модель… ${pct}% (${(e.data.loaded / 1e6).toFixed(1)} МБ)`);
    return;
  }
  const cb = pending.get(id);
  if (cb) {
    pending.delete(id);
    cb(e.data);
  }
};

let lineModelUrl = "models/line_int8.onnx"; // дефолт; /api/config в jsnet нет (404 → остаёмся на дефолте)
const configReady = fetch("/api/config")
  .then((r) => (r.ok ? r.json() : null))
  .then((cfg) => { if (typeof cfg?.lineModel === "string") lineModelUrl = cfg.lineModel; })
  .catch(() => {});
// worker резолвит относительные URL от своего пути в js/ — фиксируем от страницы
const absModel = (p) => new URL(p, document.baseURI).href;
async function runLines(payload, transfer) {
  await configReady; // не стартуем, пока не знаем, какая модель назначена
  return new Promise((resolve, reject) => {
    const id = ++reqId;
    pending.set(id, (d) => (d.error ? reject(new Error(d.error)) : resolve(d)));
    worker.postMessage({ id, modelUrl: absModel(lineModelUrl), ...payload }, transfer || []);
  });
}

// --- рисование ---
let recogTimer = null;
// поколение штрихов: CLEAR обнуляет историю, а воркер держит инкрементальную
// чернильную маску — без смены gen новые слова дорисовывались бы поверх стёртого
let gen = 0;
// номер распознавания: clear и новый запрос делают запоздалый ответ старого
// запроса недействительным (иначе стёртое слово возвращается в список)
let recogSeq = 0;
const capture = new PenCapture(canvas, {
  settings,
  showGrid: () => !!settings.grid, // клетка та же, что на остальных страницах
  onStrokeEnd: () => {
    updateHistory(); // новый штрих: undo активен, цепочка redo обнулена в pointerdown
    if (recogTimer) clearTimeout(recogTimer);
    const delay = Math.max(600, Number(settings.autoDelay) || 0);
    recogTimer = setTimeout(recognize, delay);
  },
});

// canvas лежит в скрытом до входа #app: при показе измеряем заново
// (ResizeObserver ловит переход 0 → реальный размер) и прогреваем модель
let warmed = false;
new ResizeObserver(() => {
  if (!canvas.clientWidth) return;
  if (canvas.clientWidth !== Math.round(canvas.width / (window.devicePixelRatio || 1))) {
    capture._setupDpr();
    capture.redraw();
  }
  if (!warmed) {
    warmed = true;
    runLines({ strokes: [], w: canvas.width, h: canvas.height, dpr: window.devicePixelRatio || 1 }).then(() =>
      setStatus("Модель готова. Пишите строки — распознавание после отрыва пера.")
    ).catch(() => {});
  }
}).observe(canvas);

function setStatus(t) {
  statusEl.textContent = t;
}

async function recognize() {
  if (state.busy) {
    // инференс уже в полёте — перевзводим таймер, иначе штрихи, нарисованные
    // во время него, останутся без распознавания (их onStrokeEnd уже отстрелял)
    if (recogTimer) clearTimeout(recogTimer);
    recogTimer = setTimeout(recognize, 300);
    return;
  }
  // перо снова на холсте (пишем следующую строку) — инференс не стартует:
  // на слабых устройствах (Huawei) он грузит CPU и события пера теряются.
  // onStrokeEnd перезаводит таймер — распознавание догонится после паузы.
  if (capture.drawing) return;
  if (!capture.strokes.length) {
    state.lines = [];
    state.ms = null;
    renderLines();
    setStatus("Холст пуст — напишите одну-три строки подряд.");
    return;
  }
  state.busy = true;
  const seq = ++recogSeq;
  const t0 = performance.now();
  try {
    const rect = canvas.getBoundingClientRect();
    const dpr = canvas.width / Math.max(1, rect.width);
    // растер + крой полос происходят прямо в воркере (ink-mask живёт там же:
    // домальовываются только новые штрихи, getImageData читает их bbox) —
    // main-поток свободен при письме, на слабых устройствах штрихи не «пробросами»
    // упаковка штрихов в transfer-буферы: структурированное клонирование
    // тысяч точек на длинных сессиях грузило main-поток именно в момент
    // распознавания (packStrokes/unpackStrokes — line-preprocess.js)
    const pack = packStrokes(capture.strokes);
    const res = await runLines({
      pts: pack.pts,
      counts: pack.counts,
      pen: settings.pen,
      w: canvas.width,
      h: canvas.height,
      dpr,
      // beam search со словарём в воркере (тришка строится по первому сообщению)
      alpha: state.alpha || null,
      terms: state.vocab ? state.vocab.map((v) => v.norm) : null,
      gen,
    }, [pack.pts.buffer, pack.counts.buffer]);
    if (seq !== recogSeq) return; // холст очистили/написали новое, пока ждали ответ
    if (!res.lines.length) {
      state.lines = [];
      state.ms = null;
      renderLines();
      setStatus("Строк не найдено.");
      return;
    }
    state.lines = res.lines.map((l) => ({
      // модель выдаёт естественный текст: пробел = граница слова
      text: state.alpha ? l.ids.map((i) => state.alpha[i - 1]).join("") : "",
      conf: l.conf,
      cands: l.cands,
      corrected: null,
    }));
    state.ms = Math.round(res.ms);
    renderLines();
    setStatus(`Распознано ${state.lines.length} ${pluralLines(state.lines.length)}.`);
  } catch (err) {
    setStatus("Ошибка распознавания: " + err.message);
  } finally {
    state.busy = false;
  }
}

function pluralLines(n) {
  const m10 = n % 10, m100 = n % 100;
  if (m100 >= 11 && m100 <= 14) return "строк";
  if (m10 === 1) return "строка";
  if (m10 >= 2 && m10 <= 4) return "строки";
  return "строк";
}

function avgConf() {
  if (!state.lines.length) return "—";
  const v = state.lines.reduce((a, l) => a + l.conf, 0) / state.lines.length;
  return fmtPct(v);
}

// --- панель результата ---
// Балл ниже 0.5% округлялся к «0%» и выглядел сломанным; дробные проценты
// показываем как «<1%» (0 при пустом выводе остаётся честным нулём).
function fmtPct(c) {
  if (!(c > 0)) return "0%";
  if (c < 0.005) return "<1%";
  return Math.round(c * 100) + "%";
}

function renderLines() {
  linesEl.innerHTML = "";
  renderMeta();
  if (!state.lines.length) {
    const hint = document.createElement("span");
    hint.className = "muted";
    hint.textContent = "Напишите строки на холсте — они появятся здесь.";
    linesEl.appendChild(hint);
    return;
  }
  state.lines.forEach((line, i) => {
    const row = document.createElement("div");
    row.className = "line-row";
    const num = document.createElement("span");
    num.className = "line-num";
    num.textContent = String(i + 1);
    const txt = document.createElement("span");
    const low = (line.conf || 0) < LOW_CONF; // низкая уверенность — подсветка: строку стоит проверить
    txt.className = "line-text" + (line.corrected ? " fixed" : "") + (low ? " low" : "");
    txt.textContent = line.corrected || line.text || "—";
    txt.title = `уверенность ${fmtPct(line.conf || 0)} — нажмите, чтобы отредактировать`;
    txt.addEventListener("click", () => editLine(line, txt));
    const cf = document.createElement("span");
    cf.className = "line-conf" + (low ? " low" : "");
    cf.textContent = fmtPct(line.conf || 0);
    row.append(num, txt, cf);
    linesEl.appendChild(row);
  });
}

const LOW_CONF = 0.6; // ниже — строка подсвечивается в списке и в проценте

/** «N строк за X мс · средний балл Y» — справа в панели распознанных строк. */
function renderMeta() {
  if (!state.lines.length || state.ms == null) {
    metaEl.hidden = true;
    return;
  }
  metaEl.hidden = false;
  metaEl.textContent = `${state.lines.length} ${pluralLines(state.lines.length)} за ${state.ms} мс · средний балл ${avgConf()}`;
}

/** Инлайн-правка: холст не трогаем, распознавание не перезапускаем; исправленный
 * текст становится строкой при «Скопировать» (line.corrected). */
function editLine(line, txtEl) {
  const input = document.createElement("input");
  input.className = "line-edit";
  input.value = line.corrected || line.text || "";
  txtEl.replaceWith(input);
  let finished = false;
  const done = (commit) => {
    if (finished) return; // повторный вызов из blur после перерисовки
    finished = true;
    if (commit) {
      const v = input.value.trim();
      line.corrected = v && v !== line.text ? v : null;
    }
    renderLines();
  };
  input.addEventListener("blur", () => done(true));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") done(true);
    else if (e.key === "Escape") done(false);
  });
  input.focus();
}

// --- кнопки ---
$("#btn-recognize").addEventListener("click", () => {
  if (recogTimer) clearTimeout(recogTimer);
  recognize();
});
const $undo = $("#btn-undo");
const $redo = $("#btn-redo");

function updateHistory() {
  $undo.disabled = !capture.strokes.length;
  $redo.disabled = !capture.canRedo();
}

$undo.addEventListener("click", () => {
  if (capture.undoLast()) {
    updateHistory();
    recognize();
  }
});
$redo.addEventListener("click", () => {
  if (capture.redoLast()) {
    updateHistory();
    recognize();
  }
});

// Ctrl+Z / Ctrl+Shift+Z (Ctrl+Y) — отмена/возврат штриха; внутри поля ввода
// (инлайн-правка строки) работают как обычный отмена/повтор ввода текста
document.addEventListener("keydown", (e) => {
  if (!(e.ctrlKey || e.metaKey)) return;
  if (e.target && e.target.matches("input, textarea")) return;
  const k = e.key.toLowerCase();
  if (k !== "z" && k !== "y") return;
  e.preventDefault();
  const redo = k === "y" || e.shiftKey;
  if (redo ? capture.redoLast() : capture.undoLast()) {
    updateHistory();
    recognize();
  }
});

$("#btn-clear").addEventListener("click", () => {
  if (recogTimer) { clearTimeout(recogTimer); recogTimer = null; }
  gen++;       // воркер перестроит маску с нуля — стёртые чернила не должны остаться
  recogSeq++;  // ответ распознавания, отправленного до clear, игнорируем
  capture.clear();
  state.lines = [];
  state.ms = null;
  renderLines();
  updateHistory();
  setStatus("Холст очищен.");
});
$("#btn-copy").addEventListener("click", async () => {
  const text = state.lines.map((l) => l.corrected || l.text).join("\n");
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Скопировано в буфер обмена.");
  } catch {
    setStatus("Не удалось скопировать (разрешите доступ в настройках браузера).");
  }
});

// --- запуск: гость уходит на вход ---
(async function init() {
  const me = await requireMe();
  if (!me) return; // requireMe уже редирекнул гостя на "/"
  mountShell(me);
  $("#app").hidden = false;
  try {
    const [terms, alpha] = await Promise.all([
      fetch("/terms.json").then((r) => (r.ok ? r.json() : null)),
      fetch("/models/line_vocab.json").then((r) => (r.ok ? r.json() : null)),
    ]);
    if (Array.isArray(alpha)) {
      state.alpha = alpha;
    } else {
      setStatus("Не найден models/line_vocab.json.");
    }
    if (terms && Array.isArray(terms.terms_ru)) {
      state.vocab = buildVocab(terms.terms_ru);
    }
  } catch { /* офлайн-заглушки разберёт статус */ }
  setStatus("Модель загрузится автоматически.");
  renderLines();
  updateHistory();
  showCalibHint();
})();

// подсказка о калибровке: не все символы откалиброваны — ссылка на тренировку
async function showCalibHint() {
  const el = document.getElementById("calib-hint");
  if (!el) return;
  const res = await fetch("/api/refs/summary").catch(() => null);
  if (!res || !res.ok) { el.hidden = true; return; } // не авторизован — подсказки нет
  const have = (await res.json()).have || {};
  const missing = [...CALIB_SET].filter((ch) => (have[ch] || 0) < REF_PER_CHAR);
  if (!missing.length) {
    el.hidden = false;
    el.innerHTML = `<span class="ok">Калибровка полная: все ${CALIB_SET.length} символов</span>`;
    return;
  }
  const show = missing.length <= 12 ? missing.join("") : missing.slice(0, 12).join("") + "…";
  el.hidden = false;
  el.innerHTML =
    `Калибровка: ${CALIB_SET.length - missing.length} из ${CALIB_SET.length} · не хватает <b>${show}</b><br>` +
    `<a href="/train.html">Докалибровать на тренировке →</a>`;
}

// офлайн-кэш статических ассетов (sw.js): модель .onnx/.ort и ort-wasm ~18 МБ
// качаются один раз. Регистрация из этой страницы покрывает весь origin.
// На http://IP (планшет в локальной сети) service worker недоступен — тихий
// отказ, страница работает без кэша (порт в Secure Context, как весь jsnet).
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () =>
    navigator.serviceWorker.register("/sw.js").catch(() => {})
  );
}

// отладочный хук: state/settings/recognize/capture
window.line = { state, settings, recognize, capture };
