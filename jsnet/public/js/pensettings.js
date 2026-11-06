// Верхняя панель настроек рисования (перо + планшет) — общая для страниц
// «Символы» (index.html) и «Текст по эталону» (text.html). Панель строится из
// PEN_LIMITS/PEN_META/PEN_PRESETS (pen.js), значения пишутся в settings.js —
// на холсте кисть меняется сразу, всё состояние живёт в localStorage.
import { settings, saveSettings, resetSettings } from "./settings.js";
import { PEN_LIMITS, PEN_COLORS, PEN_PRESETS, PEN_META, PEN_UNITS, normalizePen } from "./pen.js";

const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const textOf = (key, v) => `${v}${PEN_UNITS[key] || ""}`;

// Порядок и тип контролов блока «Перо».
const PEN_FIELDS = [
  { key: "R", type: "range" }, { key: "focal", type: "range" }, { key: "angle", type: "range" },
  { key: "pressure", type: "check" }, { key: "tilt", type: "check" },
  { key: "pscale", type: "range" }, { key: "smoothing", type: "range" }, { key: "hardness", type: "range" },
];

/**
 * Монтирует панель в container. capture — перерисовка холста после изменений.
 * Возвращает { render } — render() синхронизирует контролы с settings (для
 * внешних изменений вроде window.inknet.settings({...})).
 */
export function mountDrawingSettings(container, { capture }) {
  container.innerHTML = "";
  const ctl = {}; // key -> { input, val }

  // ── блок «Перо и цвет» ──────────────────────────────────────────
  const penBlock = el("div", "sb-block");
  penBlock.appendChild(el("h3", null, "Перо и цвет"));

  const sw = el("div", "sb-swatches");
  const colorCtls = [];
  for (const c of PEN_COLORS) {
    const b = el("button", "sb-swatch");
    b.type = "button";
    b.dataset.color = c.hex;
    b.style.background = c.hex;
    b.title = c.title;
    b.addEventListener("click", () => commitPen({ color: c.hex }));
    sw.appendChild(b);
    colorCtls.push(b);
  }
  const custom = el("input", "sb-swatch-custom");
  custom.type = "color";
  custom.title = "Свой цвет";
  custom.addEventListener("input", () => commitPen({ color: custom.value }));
  sw.appendChild(custom);
  penBlock.appendChild(sw);

  const grid = el("div", "sb-grid");
  for (const f of PEN_FIELDS) grid.appendChild(makeRow(f));
  penBlock.appendChild(grid);

  const presets = el("div", "sb-presets");
  for (const p of PEN_PRESETS) {
    const b = el("button", "sb-preset", p.name);
    b.type = "button";
    b.addEventListener("click", () => commitPen(p.patch));
    presets.appendChild(b);
  }
  penBlock.appendChild(presets);
  container.appendChild(penBlock);

  // ── блок «Планшет» ──────────────────────────────────────────────
  const tabBlock = el("div", "sb-block");
  tabBlock.appendChild(el("h3", null, "Планшет и холст"));

  const penOnly = el("input");
  penOnly.type = "checkbox";
  penOnly.id = "set-pen-only";
  penOnly.addEventListener("change", () => {
    settings.penOnly = penOnly.checked;
    saveSettings();
  });
  const penOnlyRow = el("label", "row");
  penOnlyRow.append(penOnly, el("span", null, "Только перо (palm rejection)"));
  tabBlock.appendChild(penOnlyRow);

  const gridChk = el("input");
  gridChk.type = "checkbox";
  gridChk.id = "set-grid";
  gridChk.addEventListener("change", () => {
    settings.grid = gridChk.checked;
    saveSettings();
    capture.redraw();
  });
  const gridRow = el("label", "row");
  gridRow.append(gridChk, el("span", null, "Клетка-разметка на холсте"));
  tabBlock.appendChild(gridRow);

  const delay = el("input");
  delay.type = "range";
  delay.min = "0"; delay.max = "2500"; delay.step = "100";
  const delayVal = el("span", "sb-val");
  delay.addEventListener("input", () => {
    settings.autoDelay = Number(delay.value);
    delayVal.textContent = delayText(settings.autoDelay);
    saveSettings();
  });
  const delayRow = el("div", "row");
  delayRow.append(el("label", null, "Авто-распознавание:"), delay, delayVal);
  tabBlock.appendChild(delayRow);

  const hint = el("p", "muted small",
    "Перо: толщина и «вытянутость» пятна, угол, давление и наклон стилуса, сглаживание дрожи " +
    "и жёсткость краёв. Сглаживание и давление применяются к самой траектории — почерк и эталоны " +
    "калибровки получают ровно то, что видно на холсте. «Только перо» игнорирует ладонь и мышь; " +
    "пауза после отрыва пера нужна для многоштриховых символов (п, ш, 4).");
  tabBlock.appendChild(hint);

  const resetBtn = el("button", "link", "Сбросить настройки рисования");
  resetBtn.addEventListener("click", () => {
    resetSettings();
    render();
    capture._setupDpr();
    capture.redraw();
  });
  tabBlock.appendChild(resetBtn);
  container.appendChild(tabBlock);

  function delayText(v) {
    return Number(v) === 0 ? "мгновенно" : `через ${v} мс`;
  }

  function makeRow(field) {
    const row = el("div", "sb-ctrl");
    const label = el("label", null, PEN_META[field.key]);
    label.htmlFor = `set-${field.key}`;
    row.appendChild(label);
    const input = el("input");
    input.id = `set-${field.key}`;
    let val = null;
    if (field.type === "range") {
      const l = PEN_LIMITS[field.key];
      input.type = "range";
      input.min = String(l.min);
      input.max = String(l.max);
      input.step = String(l.step);
      val = el("span", "sb-val");
      input.addEventListener("input", () => {
        val.textContent = textOf(field.key, input.value);
        commitPen({ [field.key]: parseFloat(input.value) });
      });
    } else {
      input.type = "checkbox";
      input.addEventListener("change", () => commitPen({ [field.key]: input.checked }));
    }
    row.appendChild(input);
    if (val) row.appendChild(val);
    ctl[field.key] = { input, val };
    return row;
  }

  // патч пера → настройки; normalizePen клампит значения (2c ≤ 2R и т.п.)
  function commitPen(patch) {
    settings.pen = normalizePen({ ...settings.pen, ...patch });
    saveSettings();
    render();
    capture.redraw();
  }

  // ── синхронизация контролов по текущим настройкам ──────────────
  function render() {
    const pen = settings.pen;
    for (const f of PEN_FIELDS) {
      const c = ctl[f.key];
      if (!c) continue;
      if (f.type === "range") {
        // «вытянутость» не может превышать толщину (2c ≤ 2R) — слайдер тоже.
        if (f.key === "focal") c.input.max = String(Math.min(PEN_LIMITS.focal.max, 2 * pen.R));
        if (f.key === "pscale") c.input.disabled = !pen.pressure;
        c.input.value = String(pen[f.key]);
        c.val.textContent = textOf(f.key, pen[f.key]);
      } else {
        c.input.checked = !!pen[f.key];
      }
    }
    for (const b of colorCtls) {
      b.classList.toggle("active", b.dataset.color === pen.color);
    }
    custom.value = pen.color;
    for (const [i, p] of PEN_PRESETS.entries()) {
      const b = presets.children[i];
      const on = Object.entries(p.patch).every(([k, v]) => pen[k] === v);
      b.classList.toggle("active", on);
    }
    penOnly.checked = !!settings.penOnly;
    gridChk.checked = !!settings.grid;
    delay.value = String(settings.autoDelay);
    delayVal.textContent = delayText(settings.autoDelay);
  }

  render();
  return { render };
}
