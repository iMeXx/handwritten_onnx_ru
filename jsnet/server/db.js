// SQLite через встроенный node:sqlite (Node >= 23). Файл базы: db/app.db
// (можно перенести переменной окружения DB_PATH — так делают тесты, им нужна
// отдельная база, чтобы не засорить рабочую)
import { DatabaseSync } from "node:sqlite";
import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
export const DB_PATH = process.env.DB_PATH || resolve(root, "db", "app.db");
mkdirSync(dirname(DB_PATH), { recursive: true });

export const db = new DatabaseSync(DB_PATH);

db.exec(`
  PRAGMA journal_mode = WAL;
  CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT NOT NULL UNIQUE COLLATE NOCASE,
    pass_hash  TEXT NOT NULL,
    consent    INTEGER NOT NULL DEFAULT 0, -- согласие на обучающий датасет (terms)
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS refs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ch         TEXT NOT NULL,
    xs         TEXT NOT NULL,
    ys         TEXT NOT NULL,
    dt         TEXT NOT NULL,
    qc         TEXT,                        -- что прочитала модель при проверке (''/'мусор'-метки); NULL = проверка пройдена
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS refs_user ON refs(user_id);
  CREATE TABLE IF NOT EXISTS train_samples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ch         TEXT NOT NULL,
    k          INTEGER NOT NULL,           -- номер написания этого символа у автора
    strokes    TEXT NOT NULL,              -- [[{x,y,t},...],...] — штрихи, как в inknet-collect/1
    started_at REAL,                       -- performance.now() на момент задания
    device     TEXT,                       -- ua/dpr/размер холста/перо (для датасета)
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
  );
  CREATE INDEX IF NOT EXISTS train_user ON train_samples(user_id);
  PRAGMA foreign_keys = ON;
`);

// существующие базы, созданные до появления согласия, — добавляем колонку
try {
  db.exec("ALTER TABLE users ADD COLUMN consent INTEGER NOT NULL DEFAULT 0");
} catch { /* колонка уже есть */ }

// базы, созданные до модельной проверки калибровки — добавляем qc-флаг эталона
try {
  db.exec("ALTER TABLE refs ADD COLUMN qc TEXT");
} catch { /* колонка уже есть */ }
