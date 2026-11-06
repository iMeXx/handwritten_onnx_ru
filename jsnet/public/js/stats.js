// Страница «Статистика датасета»: сводка, авторы, покрытие букв — по ответу
// /api/stats (сервер считает теми же гейтами, что build-line-vector-ds.mjs,
// поэтому «в make ds» на странице и реальное участие в фолдах совпадают).
import { $, requireMe, mountShell, fmtDate } from "./shell.js";

let me = null;

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function card(value, label) {
  return `<div class="card"><b>${value}</b><span>${label}</span></div>`;
}

function render(data) {
  const t = data.totals, g = data.gates;
  $("#cards").innerHTML = [
    card(`${t.writersInDs} / ${t.writersWithData}`, `авторов в датасете (всего пишущих; зарегистрировано ${t.users})`),
    card(`${t.cyrCovered}/${t.cyrTotal}`, "букв кириллицы с данными"),
    card(`${t.digitsCovered}/${t.digitsTotal}`, "цифр 0–9 с данными"),
    card(t.refsClean, `чистых эталонов (спорных: ${t.refsFlagged})`),
    card(t.trainClean, `написаний датасета (всего сдано: ${t.trainSamples})`),
    card(g.minWriterChars, "мин. букв автора для make ds"),
  ].join("");

  $("#writers-table tbody").innerHTML = data.writers.map((w) => `
    <tr>
      <td><b>${esc(w.username)}</b></td>
      <td>${fmtDate(w.createdAt)}</td>
      <td>${w.consent ? '<span class="tag ok">есть</span>' : '<span class="tag mid">нет</span>'}</td>
      <td class="num">${w.refsClean}<span class="muted">/${w.refs}</span></td>
      <td class="num">${w.refsFlagged || ""}</td>
      <td class="num">${w.trainClean}<span class="muted">/${w.train}</span></td>
      <td class="num"><b>${w.chars}</b></td>
      <td class="num">${w.words || "—"}</td>
      <td>${w.inDs ? '<span class="tag ok">да</span>' : `<span class="tag no">нет (${w.chars} < ${g.minWriterChars})</span>`}</td>
      <td class="muted">${fmtDate(w.lastActivity)}</td>
    </tr>`).join("") || '<tr><td colspan="10" class="muted">пишущих авторов пока нет</td></tr>';

  const covHtml = (list) => list.map((c) => {
    const cls = c.writers === 0 ? "zero" : c.writers < Math.max(2, data.totals.writersInDs) ? "thin" : "";
    return `<span class="cov ${cls}" title="написаний: ${c.samples}">${c.ch}<b>${c.writers}</b></span>`;
  }).join("");
  $("#chars-cyr").innerHTML = covHtml(data.charsCyr);
  $("#chars-digits").innerHTML = covHtml(data.charsDigits);

  const zero = [...data.charsCyr, ...data.charsDigits].filter((c) => c.writers === 0).map((c) => c.ch);
  $("#stats-note").textContent =
    `гейты: ${data.note}` +
    (zero.length ? ` · букв без данных: ${zero.join(" ")} — на них строки для этих букв не синтезируются` : "") +
    (t.writersInDs < 5 ? ` · для гейта M0 нужно ≥5 авторов в датасете, сейчас ${t.writersInDs}` : "");
}

async function refresh() {
  const res = await fetch("/api/stats").catch(() => null);
  if (!res || !res.ok) return ($("#stats-note").textContent = "Не удалось загрузить статистику");
  render(await res.json());
}

$("#btn-refresh").onclick = refresh;

(async () => {
  me = await requireMe();
  if (!me) return;
  mountShell(me);
  await refresh();
})();
