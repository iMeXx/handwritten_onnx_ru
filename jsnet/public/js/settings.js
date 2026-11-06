// Настройки рисования (перо/планшет). Общие для всех пользователей,
// хранятся в localStorage; объект settings — живая ссылка, можно читать в любой момент.
// pen.* — нормализованные параметры пера (pen.js), наследуются со страницы /settings
// старого проекта (ключ inknet-pen — отдельный, совместимость с прежними значениями).
import { normalizePen, PEN_DEFAULTS } from "./pen.js";

const KEY = "inknet-settings";

const DEFAULTS = {
  penOnly: false,   // принимать только перо (palm rejection)
  autoDelay: 300,   // мс паузы после отрыва пера до авто-распознавания (0 = сразу)
  grid: true,       // клетка-разметка на холсте («как в тетради», влияет на почерк)
};

function migrate(raw) {
  // Старые плоские настройки (width/pressure/wide): width — базовая толщина линии,
  // в новой модели пера это 2R. wide больше не нужен: холст всегда на всю ширину.
  const pen = raw.pen
    ? { ...raw.pen }
    : {
        ...PEN_DEFAULTS,
        ...(raw.width !== undefined ? { R: Math.max(0.5, Number(raw.width) / 2) } : {}),
        ...(raw.pressure !== undefined ? { pressure: !!raw.pressure } : {}),
      };
  return {
    penOnly: !!raw.penOnly,
    autoDelay: Number.isFinite(Number(raw.autoDelay)) ? Math.max(0, Number(raw.autoDelay)) : DEFAULTS.autoDelay,
    grid: raw.grid === undefined ? DEFAULTS.grid : !!raw.grid,
    pen: normalizePen(pen),
  };
}

function load() {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY));
    return typeof raw === "object" && raw ? migrate(raw) : { ...DEFAULTS, pen: { ...PEN_DEFAULTS } };
  } catch {
    return { ...DEFAULTS, pen: { ...PEN_DEFAULTS } };
  }
}

export const settings = load();

export function saveSettings() {
  localStorage.setItem(KEY, JSON.stringify(settings));
}

export function resetSettings() {
  Object.assign(settings, DEFAULTS);
  settings.pen = { ...PEN_DEFAULTS };
  saveSettings();
}
