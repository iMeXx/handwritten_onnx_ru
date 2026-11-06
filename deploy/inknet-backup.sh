#!/usr/bin/env bash
# Ежесуточный снимок базы написаний (inknet-backup.timer → этот скрипт, user inknet).
# Копировать app.db в WAL-режиме нельзя (снимок без -wal бывает неполным) —
# только sqlite3 .backup. Храним сжатые снимки 14 дней.
set -euo pipefail

DB="${DB_PATH:-/var/lib/inknet/app.db}"
OUT="${BACKUP_DIR:-/var/backups/inknet}"   # BACKUP_DIR — для проверок на dev-машине
[ -f "$DB" ] || { echo "бэкап: базы нет ($DB) — пропускаем" >&2; exit 0; }

stamp="$(date +%F-%H%M)"
tmp="$OUT/app-$stamp.tmp.db"
sqlite3 "$DB" ".backup '$tmp'"
gzip -9 "$tmp"
mv "$tmp.gz" "$OUT/app-$stamp.db.gz"

find "$OUT" -name 'app-*.db.gz' -mtime +14 -delete
