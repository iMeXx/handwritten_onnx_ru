// Сбор траектории с canvas через Pointer Events + отрисовка пером (pen.js).
// Настройки (settings.penOnly / settings.pen.* / settings.grid) читаются на каждое
// событие — их можно менять в UI на лету. Сглаживание применяется к СОХРАНЁННЫМ
// точкам: распознаётся ровно то, что видно на холсте, поэтому настройки пера
// реально влияют и на почерк, и на эталоны калибровки.
import { fillPaper, drawStroke, drawSegment, inkPoint } from "./pen.js";

export class PenCapture {
  constructor(canvas, { onStrokeEnd, onStrokeStart, onRejectedInput, onPoint, showGrid, settings } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onStrokeEnd = onStrokeEnd;
    this.onStrokeStart = onStrokeStart;
    this.onRejectedInput = onRejectedInput;
    this.onPoint = onPoint; // живой поток точек (метрики пера online)
    this.showGrid = showGrid || (() => false);
    this.settings = settings || { penOnly: false, grid: false, pen: {} };
    this.strokes = []; // завершённые штрихи: [[{x,y,t,pressure,tiltX,tiltY},...], ...]
    this._redo = []; // убранные undo штрихи — redoLast возвращает их в исходном порядке
    this.points = []; // текущий штрих
    this.drawing = false;
    this._smooth = null; // сглаженные координаты указателя (экспоненциальный фильтр)
    this._setupDpr();

    // Холст растянут фиксированным layout без скроллинга страницы — следим за ним
    // напрямую (поворот планшета, баннер калибровки меняют высоту).
    if (typeof ResizeObserver !== "undefined") {
      let lastW = 0, lastH = 0;
      new ResizeObserver(() => {
        const r = this.canvas.getBoundingClientRect();
        if (Math.abs(r.width - lastW) < 1 && Math.abs(r.height - lastH) < 1) return;
        lastW = r.width; lastH = r.height;
        this._setupDpr();
        this.redraw();
      }).observe(canvas);
    }
    window.addEventListener("resize", () => { this._setupDpr(); this.redraw(); });

    // Drag пером по холсту иначе запускает системное выделение текста страницы
    // (видно «подсветку» при отрыве пера, особенно на планшете).
    canvas.style.userSelect = "none";
    canvas.style.webkitUserSelect = "none";
    canvas.style.webkitTouchCallout = "none";
    canvas.addEventListener("selectstart", (e) => { e.preventDefault(); return false; });
    canvas.addEventListener("dragstart", (e) => e.preventDefault());
    canvas.addEventListener("touchmove", (e) => e.preventDefault(), { passive: false });

    canvas.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      // режим «только перо»: отсекаем ладонь (touch) и мышь
      if (this.settings.penOnly && e.pointerType !== "pen") {
        if (this.onRejectedInput) this.onRejectedInput(e.pointerType);
        return;
      }
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        /* указатель уже отпущен — рисуем без захвата */
      }
      this._redo.length = 0; // новый штрих — цепочка redo обрвана
      this.drawing = true;
      this.points = [];
      this._smooth = null;
      this._addPoint(e);
      // Инкрементальная отрисовка (как brush.js в test-py): на каждое движение
      // рисуем только новый сегмент; полная перерисовка всего штриха на каждый
      // pointermove убивала непрерывное письмо — стоимость росла с его длиной.
      const pen = this.settings.pen || {};
      this._lastInk = inkPoint(pen, this.points[0]);
      this._paintedIdx = 0;
      drawStroke(this.ctx, pen, this.points); // начальная точка-пятно
      if (this.onStrokeStart) this.onStrokeStart();
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this.drawing) return;
      const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      for (const ev of evs.length ? evs : [e]) this._addPoint(ev);
      const pen = this.settings.pen || {};
      while (this._paintedIdx + 1 < this.points.length) {
        const b = inkPoint(pen, this.points[this._paintedIdx + 1]);
        drawSegment(this.ctx, pen, this._lastInk, b);
        this._lastInk = b;
        this._paintedIdx++;
      }
    });
    const finish = (e) => {
      if (!this.drawing) return;
      this.drawing = false;
      this._smooth = null;
      // штрих из одной точки (Android угнал жест.pointercancel'ом) серверный
      // контракт считает некорректным (валидация: ≥2 точек) — не сохраняем,
      // чтобы единичный тап не валил сдачу всей сессии
      if (this.points.length >= 2) this.strokes.push(this.points);
      this.points = [];
      this._lastInk = null;
      if (this.onStrokeEnd) this.onStrokeEnd();
    };
    canvas.addEventListener("pointerup", finish);
    canvas.addEventListener("pointercancel", finish);
  }

  _setupDpr() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  _addPoint(e) {
    // Chromium на Android в getCoalescedEvents подмешивает события с NaN
    // координатами: экспоненциальный фильтр расползается NaN по всему штриху,
    // JSON.stringify превращает NaN в null — и сервер отклоняет сдачу целиком.
    if (!Number.isFinite(e.clientX) || !Number.isFinite(e.clientY)) return;
    const rect = this.canvas.getBoundingClientRect();
    const p = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
      t: performance.now(),
      // для мыши/touch pressure обычно 0 или 0.5 — подставляем середину диапазона
      pressure: e.pressure > 0 ? e.pressure : 0.5,
      tiltX: e.tiltX || 0,
      tiltY: e.tiltY || 0,
    };
    if (this.points.length) {
      const dt = p.t - this.points[this.points.length - 1].t;
      if (dt < 0.5) return; // защита от дублей событий
    }
    // Экспоненциальный фильтр дрожания руки (как в brush.js): 0% — сырые
    // координаты, 100% — почти полная инерция (коэффициент 0.05).
    const sm = Math.min(Math.max(Number(this.settings.pen?.smoothing) || 0, 0), 100);
    const k = 1 - (sm / 100) * 0.95;
    if (!this._smooth) this._smooth = { x: p.x, y: p.y };
    else {
      this._smooth.x += (p.x - this._smooth.x) * k;
      this._smooth.y += (p.y - this._smooth.y) * k;
    }
    this.points.push({ ...p, x: this._smooth.x, y: this._smooth.y });
    if (this.onPoint) this.onPoint(this.points[this.points.length - 1], this.points);
  }

  /** Все штрихи сессии одним трассом: dt между штрихами = реальная пауза. */
  getTrajectory() {
    const all = [];
    for (const s of this.strokes) all.push(...s);
    all.push(...this.points);
    const xs = [], ys = [], dt = [];
    for (let i = 0; i < all.length; i++) {
      xs.push(all[i].x);
      ys.push(all[i].y);
      dt.push(i === 0 ? 16 : all[i].t - all[i - 1].t);
    }
    return { xs, ys, dt };
  }

  clear() {
    this.strokes = [];
    this.points = [];
    this._redo = [];
    this.redraw();
  }

  /** Отменить последний завершённый штрих. true — если было что отменять. */
  undoLast() {
    if (!this.strokes.length) return false;
    this._redo.push(this.strokes.pop());
    this.redraw();
    return true;
  }

  /** Вернуть последний убранный штрих (в исходном порядке). true — если было что возвращать. */
  redoLast() {
    if (!this._redo.length) return false;
    this.strokes.push(this._redo.pop());
    this.redraw();
    return true;
  }

  canRedo() {
    return this._redo.length > 0;
  }

  redraw() {
    const rect = this.canvas.getBoundingClientRect();
    const ctx = this.ctx;
    // Белая «бумага» + клетка-разметка (влияет на почерк; при калибровке
    // показывается всегда, в обычной работе — по настройке grid).
    fillPaper(ctx, rect.width, rect.height, this.showGrid() || !!this.settings.grid);
    const pen = this.settings.pen || {};
    const strokes = this.drawing ? [...this.strokes, this.points] : this.strokes;
    for (const pts of strokes) drawStroke(ctx, pen, pts);
    // ред sync инкрементальной дорисовки: штрих в процессе только что
    // перерисован целиком — продолжать надо с его последней точки
    if (this.drawing && this.points.length) {
      this._lastInk = inkPoint(pen, this.points[this.points.length - 1]);
      this._paintedIdx = this.points.length - 1;
    }
  }
}
