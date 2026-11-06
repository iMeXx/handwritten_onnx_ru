// Сопоставление распознанного текста эталонным терминам словаря (terms.json).
// Посимвольная страница: алфавит 33 буквы (рег слит) + цифры, пробел кнопкой.
// Строчная страница (CTC): модель выдаёт естественный текст, пробел = граница слова.
// Термины с другой графикой (латиница, формулы, пунктуация) в словарь не входят.
//
// Оценка термина — выравнивание «наблюдение ↔ термин» (Нидлмана-Вунша) в лог-вероятностях:
// подстановка = log P(буква термина | кандидаты написанного символа), разрывы — штраф GAP.
// Это исправляет типовые ошибки распознавания (ж↔х, 3↔з, у↔ц, ш↔щ, о↔0) в пользу словаря.

const SUPPORTED = /^[0-9а-яё ]+$/; // алфавит модели + пробел (в нижнем регистре)

/** termsRaw: массив строк → [{orig, norm, chars}] только полностью писуемых терминов. */
export function buildVocab(termsRaw) {
  const vocab = [];
  const seen = new Set();
  for (const t of termsRaw) {
    const norm = t.toLowerCase().trim().replace(/\s+/g, " ");
    if (!norm || !SUPPORTED.test(norm)) continue;
    if (seen.has(norm)) continue;
    seen.add(norm);
    vocab.push({ orig: t, norm, chars: [...norm] });
  }
  return vocab;
}

const GAP = -2.5; // лог-штраф вставки/пропуска символа
const P_FLOOR = 1e-3; // вероятность «не кандидат» не ниже этой

/** obs: [[{char, prob}], ...] — кандидаты на каждый написанный символ.
 *  Возвращает топ-limit терминов: [{orig, norm, pct}] по апостериорному совпадению. */
export function matchTerms(vocab, obs, limit = 6, filter = "") {
  const n = obs.length;
  if (!n) return [];
  const maps = obs.map((cands) => {
    const m = new Map();
    for (const c of cands) m.set(c.char, Math.max(m.get(c.char) || 0, c.prob));
    return m;
  });
  const f = filter.trim().toLowerCase();
  const scored = [];
  for (const v of vocab) {
    if (f && !v.norm.includes(f)) continue;
    const m = v.chars.length;
    // термин сильно другой длины — выравнивание всё равно проиграет, пропускаем
    if (Math.abs(m - n) > Math.max(2, Math.ceil(n / 3))) continue;
    const s = alignScore(v.chars, maps, n);
    scored.push({ orig: v.orig, norm: v.norm, pct: Math.round(100 * Math.exp(s / Math.max(m, n))) });
  }
  scored.sort((a, b) => b.pct - a.pct);
  return scored.slice(0, limit);
}

function alignScore(term, maps, n) {
  const m = term.length;
  let prev = new Float64Array(m + 1);
  for (let j = 1; j <= m; j++) prev[j] = prev[j - 1] + GAP;
  for (let i = 1; i <= n; i++) {
    const cur = new Float64Array(m + 1);
    cur[0] = prev[0] + GAP;
    const mp = maps[i - 1];
    for (let j = 1; j <= m; j++) {
      const sub = prev[j - 1] + Math.log(Math.max(mp.get(term[j - 1]) || 0, P_FLOOR));
      cur[j] = Math.max(sub, prev[j] + GAP, cur[j - 1] + GAP);
    }
    prev = cur;
  }
  return prev[m];
}
