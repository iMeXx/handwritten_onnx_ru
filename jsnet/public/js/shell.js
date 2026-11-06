// Общий слой страниц (вход/навигация/планы символов), урезанный до трёх
// местных страниц.

export const LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
export const UPPER = LETTERS.toUpperCase();
export const DIGITS = "123456789"; // как в плане сбора: цифры без нуля
export const CALIB_SET = LETTERS + UPPER + DIGITS; // 75 символов тренировки
export const REF_PER_CHAR = 1; // эталонов на символ при первой тренировке

// план сбора (ml-план старого проекта):
// путающие пары — из confusion matrix боевой модели, цели 2/4 написания
export const HARD = new Set("жхз3уцшщф4тг".split(""));
export const TARGET = 2;
export const HARD_TARGET = 4;
export const SESSION_SIZE = 12;

export const IS_UPPER = (ch) => ch !== ch.toLowerCase() && ch === ch.toUpperCase();

export const $ = (sel, root = document) => root.querySelector(sel);

export function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m100 >= 11 && m100 <= 14) return many;
  if (m10 === 1) return one;
  if (m10 >= 2 && m10 <= 4) return few;
  return many;
}

export const fmtDate = (s) =>
  s ? new Date(String(s).replace(" ", "T") + "Z").toLocaleDateString("ru-RU") : "—";

// гость уходит на вход; страница работает только под авторизованным автором.
// redirect=false — для самой страницы входа: редирект на "/" там был бы редиректом
// на себя, то есть бесконечной перезагрузкой (401 → replace("/") → снова 401).
export async function requireMe({ redirect = true } = {}) {
  const res = await fetch("/api/me").catch(() => null);
  if (!res) { return null; }
  if (res.status === 401) {
    if (redirect) location.replace("/");
    return null;
  }
  return (await res.json()).user;
}

// шапка: аватар + меню перехода между страницами (общая разметка в html)
export function mountShell(me) {
  const $avatar = $("#btn-avatar");
  const $menu = $("#app-menu");
  $avatar.hidden = false;
  $avatar.textContent = [...String(me.username).trim()][0]?.toUpperCase() || "?";
  $avatar.title = me.username;
  $avatar.onclick = (e) => { e.stopPropagation(); $menu.hidden = !$menu.hidden; };
  document.addEventListener("click", (e) => {
    if (!$menu.hidden && !$menu.contains(e.target) && e.target !== $avatar) $menu.hidden = true;
  });
  $("#btn-logout").onclick = async () => {
    await fetch("/api/logout", { method: "POST" }).catch(() => {});
    location.replace("/");
  };
}
