// Вход по имени (index.html): паролей и регистрации нет — имя автора и есть его
// профиль в датасете. Гость видит форму, вошедший — карточки страниц. Подсказка
// в datalist — уже заведённые авторы: опечатка в имени породила бы «нового
// автора» с пустым датасетом и исказила бы writer-disjoint разрезы.
import { $, requireMe, mountShell } from "./shell.js";

const $login = $("#login");
const $hub = $("#hub");
const $err = $("#login-error");

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function showError(text) {
  $err.textContent = text;
  $err.hidden = !text;
}

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("");
  const username = new FormData(e.target).get("username");
  const res = await fetch("/api/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  }).catch(() => null);
  const body = (await res?.json().catch(() => ({}))) ?? {};
  if (!res || !res.ok) return showError(body.error || "Сервер не отвечает");
  location.reload();
});

// известные авторы для подсказки (публичный GET — форма входа видна гостю)
(async () => {
  const res = await fetch("/api/authors").catch(() => null);
  if (!res || !res.ok) return;
  const { authors = [] } = await res.json();
  $("#known-authors").innerHTML = authors.map((a) => `<option value="${esc(a)}">`).join("");
})();

(async () => {
  // redirect:false — мы и есть страница входа: редирект на "/" здесь смыкал бы
  // цикл перезагрузок (401 → replace("/") → 401 → …), страница «мигала»
  const me = await requireMe({ redirect: false });
  if (!me) { $login.hidden = false; return; }
  mountShell(me);
  $("#hub-name").textContent = me.username;
  $hub.hidden = false;
})();
