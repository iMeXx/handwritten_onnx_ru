// CTC-декодирование строчной модели: жадная декодировка (исторически в line.worker.js)
// и beam search с мягким словарным бонусом. Модуль чистый (без DOM/canvas/ort) —
// тестируется в node: tests/decoding.mjs.

// softmax по строке логитов [C] в переиспользуемый буфер вероятностей
function softmax(logits, out) {
  let max = -Infinity;
  for (let c = 0; c < logits.length; c++) if (logits[c] > max) max = logits[c];
  let sum = 0;
  for (let c = 0; c < logits.length; c++) {
    out[c] = Math.exp(logits[c] - max);
    sum += out[c];
  }
  for (let c = 0; c < logits.length; c++) out[c] /= sum;
}

// топ-3 символьных кандидатов {i, p} (классы 1..C-1, blank исключён)
function topK(probs, C) {
  const top = [];
  for (let k = 0; k < 3; k++) {
    let bi = -1;
    let bv = -1;
    for (let c = 1; c < C; c++) {
      if (probs[c] > bv && !top.some((x) => x.i === c)) { bv = probs[c]; bi = c; }
    }
    if (bi < 0) break;
    top.push({ i: bi, p: bv });
  }
  return top;
}

/** Жадная CTC-декодировка + кандидаты и уверенность по строке. */
export function decodeCTC(logits, C) {
  const T = logits.length / C;
  const probs = new Float64Array(C);
  const ids = [];
  const cands = [];
  let logSum = 0;
  let prev = -1;
  let runChar = -1;
  let runStart = -1;
  const closeRun = (end) => {
    // кадр прогона с максимальной вероятностью символа — его топ-3 кандидаты
    let bestT = runStart;
    let bestP = -1;
    for (let t = runStart; t < end; t++) {
      const p = logits[t * C + runChar]; // без softmax — монотонно достаточно для максимума
      if (p > bestP) { bestP = p; bestT = t; }
    }
    softmax(logits.subarray(bestT * C, bestT * C + C), probs);
    ids.push(runChar);
    cands.push(topK(probs, C));
    logSum += Math.log(Math.max(probs[runChar], 1e-6));
  };
  for (let t = 0; t < T; t++) {
    let bi = 0;
    let bv = -Infinity;
    for (let c = 0; c < C; c++) {
      const v = logits[t * C + c];
      if (v > bv) { bv = v; bi = c; }
    }
    if (bi !== 0 && bi !== prev) {
      if (runChar >= 0) closeRun(t); // предыдущий прогон закрыт
      runChar = bi;
      runStart = t;
    } else if (bi === 0 || bi !== runChar) {
      if (runChar >= 0) { closeRun(t); runChar = -1; }
    }
    prev = bi;
  }
  if (runChar >= 0) closeRun(T);
  return { ids, cands, conf: ids.length ? Math.exp(logSum / ids.length) : 0 };
}

const LOGF = -1e9;
const logadd = (a, b) =>
  a <= LOGF ? b : b <= LOGF ? a : a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b));

/** Тришка norm-терминов: узлы {ch: Map(char -> node), end: bool}. */
export function buildTrie(normWords) {
  const root = { end: false, ch: new Map() };
  for (const w of normWords) {
    if (!w) continue;
    let n = root;
    for (const c of w) {
      let nx = n.ch.get(c);
      if (!nx) { nx = { end: false, ch: new Map() }; n.ch.set(c, nx); }
      n = nx;
    }
    n.end = true;
  }
  return root;
}

/**
 * CTC prefix beam search с МЯГКИМ словарным бонусом: beam остаётся
 * лексиконом не стеснённым (OOV — имена, редкие слова — распознаются как есть),
 * но за каждое слово, точно совпавшее с термином словаря (term.norm), к счёту
 * добавляется bonus. Балм перекалибровывает типовые ошибки модели (ж↔х, о↔а,
 * 3↔з, ш↔щ) в пользу терминов: словарная правка со страницы убрана, бонус
 * beam — единственная оставшаяся словарная помощь (внутри воркера).
 * words — norm-термины словаря (см. buildVocab), alpha — символ класса i = alpha[i-1].
 * Формат результата идентичен decodeCTC: {ids, cands, conf}.
 *
 * oovPen — вторая ручка той же силы правки: штраф за каждый символ,
 * вписанный вне словарного пути (тришка сошла на null). Бонус один
 * односторонний: словарный путь получает +bonus, путь «всё подряд» — ничего.
 * Бонус+штраф вместе задают силу лексиконного давления: для строк, где слова
 * почти все словарные (школьный диктант), она оправдана; для текста с именами
 * и терминами вне словаря oovPen держат малым или нулём — он, в отличие от
 * бонуса, затирает OOV-написания. oovPen=0 — историческое поведение
 * (обратная совместимость).
 */
export function beamDecode(logits, C, { words, alpha, bonus = 2.5, beam = 64, oovPen = 0 } = {}) {
  if (!alpha) throw new Error("beamDecode: нужен alpha (символы классов)");
  const T = logits.length / C;
  const trie = buildTrie(words || []);
  const SPACE = " ";
  const chOf = (c) => alpha[c - 1];
  const labelCut = 8; // классы ниже frameMax - labelCut в балм не идут

  // запись префикса: key = свёрнутая последовательность символов; b/nb — CTC-prefix
  // вероятности (blank / с окончанием на класс); node — узел тришки текущего слова
  // (null = слово уже вышло за словарь); hits — счёт законченных словарных слов;
  // oov — счёт символов, ушедших в OOV (штраф oovPen).
  const start = { key: "", b: 0, nb: new Map(), node: trie, hits: 0, oov: 0, chars: [] };
  let prefixes = new Map([["", start]]);

  const cloneFrame = (p) => ({ key: p.key, b: LOGF, nb: new Map(), node: p.node, hits: p.hits, oov: p.oov, chars: p.chars });
  const total = (r) => { let s = r.b; for (const lp of r.nb.values()) s = logadd(s, lp); return s; };

  // переход с дописыванием класса c: обновляет тришку, hits и oov
  const advance = (p, c) => {
    const ch = chOf(c);
    if (ch === SPACE) {
      const hit = p.node && p.node !== trie && p.node.end;
      return { node: trie, hits: p.hits + (hit ? 1 : 0), oov: p.oov };
    }
    if (p.node) {
      const nx = p.node.ch.get(ch);
      if (nx) return { node: nx, hits: p.hits, oov: p.oov };
      if (p.node.end) { // слитное написание: слово словаря закончилось, новое начинается
        const nx2 = trie.ch.get(ch);
        if (nx2) return { node: nx2, hits: p.hits + 1, oov: p.oov };
      }
      return { node: null, hits: p.hits, oov: p.oov + 1 };
    }
    return { node: null, hits: p.hits, oov: p.oov + 1 };
  };

  const finalScore = (r) => {
    let s = total(r) + r.hits * bonus - r.oov * oovPen;
    if (r.node && r.node !== trie && r.node.end) s += bonus; // последнее слово без пробела
    return s;
  };

  for (let t = 0; t < T; t++) {
    const base = t * C;
    let fmax = -Infinity;
    for (let c = 0; c < C; c++) if (logits[base + c] > fmax) fmax = logits[base + c];
    const cs = [];
    for (let c = 1; c < C; c++) if (logits[base + c] > fmax - labelCut) cs.push(c);

    const next = new Map();
    const put = (key, make) => {
      let r = next.get(key);
      if (!r) { r = make(); next.set(key, r); }
      return r;
    };

    for (const p of prefixes.values()) {
      const tot = total(p);
      // blank: префикс растёт в тот же key
      if (tot > LOGF) {
        const r = put(p.key, () => cloneFrame(p));
        r.b = logadd(r.b, tot + logits[base]); // эмиссия blank-класса
      }
      for (const c of cs) {
        const lp = logits[base + c];
        if (lp <= LOGF) continue;
        // повтор последнего класса без blank: сворачивается в тот же префикс
        const rep = p.nb.get(c);
        if (rep !== undefined) {
          const r = put(p.key, () => cloneFrame(p));
          r.nb.set(c, logadd(r.nb.get(c) ?? LOGF, rep + lp)); // повтор класса без blank
        }
        // рост префикса символом c: все истории, где последний класс != c
        let from = p.b;
        for (const [d, dlp] of p.nb) if (d !== c) from = logadd(from, dlp);
        if (from > LOGF) {
          const { node, hits, oov } = advance(p, c);
          const key = p.key + chOf(c);
          const r = put(key, () => ({ key, b: LOGF, nb: new Map(), node, hits, oov, chars: p.chars.concat(c) }));
          r.nb.set(c, logadd(r.nb.get(c) ?? LOGF, from + lp)); // эмиссия класса c в этом кадре
        }
      }
    }

    if (next.size > beam) {
      const arr = [...next.values()].sort((a, b) => finalScore(b) - finalScore(a)).slice(0, beam);
      prefixes = new Map(arr.map((r) => [r.key, r]));
    } else {
      prefixes = next;
    }
  }

  let best = null, bestScore = -Infinity;
  for (const r of prefixes.values()) {
    const s = finalScore(r);
    if (s > bestScore && r.chars.length) { bestScore = s; best = r; }
  }
  if (!best) return { ids: [], cands: [], conf: 0 };
  return alignCandidates(logits, C, best.chars);
}

/** Кандидаты для словарной правки: монотонно раскидываем символы по кадрам
 * (каждый символ — кадр с максимумом своего логита после предыдущего кадра),
 * берём топ-3 softmax этих кадров; conf — геосреднее вероятностей символов.
 *
 * Поиск окна ограничен окрестностью ПРОПОРЦИОНАЛЬНОЙ позиции символа: без
 * ограничения частые буквы (о/е) выравнивались на чужой пик логита в другом
 * слове полосы — там вероятность этого класса ~1e-9, и нижняя отсечка 1e-6
 * роняла геосреднее всего балла к нулю (демо: «голографическ» показывал 0%,
 * хотя путь beam верный). Окно щедрые (~2 ширины слота), monotonicity важна
 * для CTC; при вырожденном окне — откат на весь остаток. */
function alignCandidates(logits, C, classSeq) {
  const T = logits.length / C;
  const probs = new Float64Array(C);
  const ids = [];
  const cands = [];
  let logSum = 0;
  let t = 0;
  const L = classSeq.length;
  const slack = Math.max(8, Math.round(T / L));
  for (let i = 0; i < L; i++) {
    const cls = classSeq[i];
    const room = T - (L - i - 1); // каждому оставшемуся символу — кадр
    let lo = Math.max(t, Math.floor((i * T) / L) - slack);
    let hi = Math.min(room, Math.ceil(((i + 1) * T) / L) + slack);
    if (lo >= hi) { lo = t; hi = room; }
    let bf = lo, bv = -Infinity;
    for (let f = lo; f < hi; f++) {
      const v = logits[f * C + cls];
      if (v > bv) { bv = v; bf = f; }
    }
    softmax(logits.subarray(bf * C, bf * C + C), probs);
    ids.push(cls);
    cands.push(topK(probs, C));
    logSum += Math.log(Math.max(probs[cls], 1e-6));
    t = bf + 1;
  }
  return { ids, cands, conf: ids.length ? Math.exp(logSum / ids.length) : 0 };
}
