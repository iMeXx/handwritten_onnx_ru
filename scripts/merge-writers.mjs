// Снимок базы написаний + склейка аккаунтов ОДНОГО человека в одного автора.
//
// Зачем: билдер датасета (scripts/build-line-vector-ds.mjs, parity-копия
// inknet-online) группирует написания по username. В базе за одного человека
// заведено несколько аккаунтов (etel1/Etel2, Vika1/Vika2, Dima1/Dima2 — сдали
// с iPad и Android-планшета в один день), поэтому writer-disjoint фолды
// протекают: тест-автор Vika1 учится на Vika2. Менять билдер нельзя (гейт
// make parity), значит группируем авторов ДО билдера — в копии базы.
//
// Исходник не трогается: app.db копируется (вместе с -wal) в снимок, склейка
// идёт в снимке. Список склейки — scripts/writers-merge.json (править по
// факту: пары вычислены по метаданным сдачи и геометрии написаний, сами
// чернила различить авторов не позволяют — glyphEmbedding даёт
// межавторский/внутриавторский разброс 1.00–1.23, см. docs/DATASET-AUDIT-2026-11-05.md).
//
// запуск: make merge   (или node scripts/merge-writers.mjs)
//   DB_SRC=<app.db>      исходная база (по умолчанию jsnet/db/app.db)
//   MERGED_DB=<путь>     куда положить снимок (по умолчанию data/db/app-merged.db)
//   FORCE=1              перезаписать существующий снимок
import { copyFileSync, existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const SRC = resolve(process.env.DB_SRC || resolve(ROOT, "jsnet/db/app.db"));
const DST = resolve(process.env.MERGED_DB || resolve(ROOT, "data/db/app-merged.db"));
const MAP_FILE = resolve(process.env.WRITERS_MAP || resolve(HERE, "writers-merge.json"));
const MIN_WRITER_CHARS = Number(process.env.MIN_WRITER_CHARS || 10);

if (!existsSync(SRC)) { console.error(`нет базы-источника: ${SRC}`); process.exit(1); }
if (!existsSync(MAP_FILE)) { console.error(`нет файла склейки: ${MAP_FILE}`); process.exit(1); }
const map = JSON.parse(readFileSync(MAP_FILE, "utf8"));
mkdirSync(dirname(DST), { recursive: true });
if (existsSync(DST)) unlinkSync(DST); // VACUUM INTO не перезаписывает файл; снимок — производный, пересобираем всегда

// 1) копия базы (вместе с WAL — иначе свежие написания не попадут в снимок)
const tmp = DST + ".src.db";
copyFileSync(SRC, tmp);
for (const ext of ["-wal", "-shm"]) if (existsSync(SRC + ext)) copyFileSync(SRC + ext, tmp + ext);

// 2) VACUUM INTO — консистентный однофайловый снимок без WAL
const srcDb = new DatabaseSync(tmp);
srcDb.prepare("VACUUM INTO ?").all(DST);
srcDb.close();
for (const ext of ["", "-wal", "-shm"]) if (existsSync(tmp + ext)) unlinkSync(tmp + ext);

// 3) склейка
const db = new DatabaseSync(DST);
const counts = (id) => ({
  refs: db.prepare("select count(*) n from refs where user_id=?").get(id).n,
  samples: db.prepare("select count(*) n from train_samples where user_id=?").get(id).n,
});
const findUser = (name) => db.prepare("select id, username, consent, pass_hash from users where username = ?").get(name);

const log = [];
let moved = 0;
for (const [canonical, aliases] of Object.entries(map.authors || {})) {
  const list = [...new Set([canonical, ...(aliases || [])])];
  const found = list.map((n) => findUser(n)).filter(Boolean);
  if (!found.length) { log.push({ canonical, skipped: "нет таких аккаунтов" }); continue; }
  let target = found.find((u) => u.username.toLowerCase() === canonical.toLowerCase());
  if (!target) {
    const srcUser = found[0];
    db.prepare("insert into users (username, pass_hash, consent) values (?, ?, ?)")
      .run(canonical, srcUser.pass_hash, 0);
    target = findUser(canonical);
  }
  // NOCASE уникальность: canonical мог занять место регистраного дубля
  if (target.username !== canonical)
    db.prepare("update users set username = ? where id = ?").run(canonical, target.id);

  const before = counts(target.id);
  let consents = target.consent;
  for (const u of found) {
    if (u.id === target.id) continue;
    const c = counts(u.id);
    db.prepare("update refs set user_id = ? where user_id = ?").run(target.id, u.id);
    db.prepare("update train_samples set user_id = ? where user_id = ?").run(target.id, u.id);
    db.prepare("delete from sessions where user_id = ?").run(u.id);
    db.prepare("delete from users where id = ?").run(u.id);
    consents = Math.max(consents, u.consent);
    moved += c.refs + c.samples;
    log.push({ canonical, merged: u.username, refs: c.refs, samples: c.samples });
  }
  if (consents !== target.consent) db.prepare("update users set consent = ? where id = ?").run(consents, target.id);
  const after = counts(target.id);
  log.push({ canonical, itog: `refs ${before.refs} -> ${after.refs}, samples ${before.samples} -> ${after.samples}` });
}

// 4) контроль: ни одно написание не потерялось
const total = {
  refs: db.prepare("select count(*) n from refs").get().n,
  samples: db.prepare("select count(*) n from train_samples").get().n,
};
const orphan = db.prepare("select count(*) n from train_samples where user_id not in (select id from users)").get().n
  + db.prepare("select count(*) n from refs where user_id not in (select id from users)").get().n;
if (orphan) { console.error(`потеряны написания (сироты): ${orphan}`); db.close(); process.exit(1); }

// 5) кто годен для датасета (тот же гейт покрытия букв, что у билдера)
const vocab = new Set(JSON.parse(readFileSync(resolve(ROOT, "jsnet/public/models/line_vocab.json"), "utf8")));
const authors = [];
for (const u of db.prepare("select id, username, consent from users order by id").all()) {
  const chars = new Set();
  for (const r of db.prepare("select ch from refs where user_id = ? and qc is null").all(u.id))
    if (vocab.has(String(r.ch).toLowerCase())) chars.add(String(r.ch).toLowerCase());
  for (const r of db.prepare("select ch from train_samples where user_id = ?").all(u.id))
    if (vocab.has(String(r.ch).toLowerCase())) chars.add(String(r.ch).toLowerCase());
  const c = counts(u.id);
  authors.push({ username: u.username, consent: u.consent, refs: c.refs, samples: c.samples,
    letters: chars.size, inDataset: chars.size >= MIN_WRITER_CHARS });
}
db.close();

writeFileSync(DST.replace(/\.db$/, "-merge.json"), JSON.stringify({
  source: SRC, mergedDb: DST, map: MAP_FILE, moved, total, authors,
  note: "аккаунты склеены по человеку: writer-disjoint фолды строятся по людям, а не по логинам",
}, null, 2));

console.log(`склейка: ${moved} написаний перенесено, всего refs ${total.refs}, samples ${total.samples}`);
for (const l of log) console.log("  " + JSON.stringify(l));
console.log(`\nавторы снимка (гейт покрытия букв = ${MIN_WRITER_CHARS}):`);
for (const a of authors)
  console.log(`  ${a.username.padEnd(10)} refs ${String(a.refs).padStart(3)} samples ${String(a.samples).padStart(4)} букв ${String(a.letters).padStart(3)} в датасете: ${a.inDataset ? "да" : "нет"}`);
console.log(`\nснимок -> ${DST}\nотчёт  -> ${DST.replace(/\.db$/, "-merge.json")}`);
console.log("датасет по нему: make ds-merged");
