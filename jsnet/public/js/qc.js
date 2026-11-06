// QC «как это читает модель» — местная замена ONNX-проверке: ближайший сосед
// по фиксразмерному описанию глифа (glyphEmbedding).
// Эталонные описания сохранённых эталонов считает СЕРВЕР тем же кодом
// src/ink-features.js, что и билдер датасета, — страница вердикта меряет ровно
// то представление, которое потом увидит векторная ветка. Обучения нет:
// «модель» = nearest-neighbor по метрике embedDist (инвариантной к повороту/
// сдвигу/масштабу). Пока у автора нет ни одного чистого эталона — QC молчит.
import { buildGlyphVectors, glyphEmbedding, embedDist } from "/src/ink-features.js";

export class NNQC {
  constructor() {
    this.refs = null; // [{id, ch, emb}] — из /api/refs/emb
  }

  // загрузка/обновление эталонных описаний (после сохранения эталонов — force)
  async load(force = false) {
    if (this.refs && !force) return this.refs;
    const res = await fetch("/api/refs/emb").catch(() => null);
    this.refs = res && res.ok ? (await res.json()).refs || [] : this.refs || [];
    return this.refs;
  }

  /**
   * Вердикт по написанию: strokes — как в PenCapture.strokes / inknet-collect.
   * Возвращает null если QC недоступна (нет эталонов) — страница тогда просто
   * сохраняет без проверки; иначе {char, dist} ближайшего эталона.
   */
  read(strokes) {
    if (!this.refs || !this.refs.length) return null;
    let e;
    try { e = glyphEmbedding(buildGlyphVectors(strokes)); } catch { return null; }
    let best = null;
    for (const r of this.refs) {
      const d = embedDist(e, Float64Array.from(r.emb));
      if (!best || d < best.dist) best = { char: r.ch, dist: d };
    }
    return best;
  }
}

// ожидаемое прочтение: билдер датасета приводит все буквы к нижнему регистру,
// поэтому ЗАГЛАВНАЯ «А» засчитывается как прочитанная «а»
export const expectedRead = (ch) => String(ch).toLowerCase();
