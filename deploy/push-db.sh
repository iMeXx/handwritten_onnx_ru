#!/usr/bin/env bash
# Перенос базы написаний с dev-машины на сервер (запускается ЛОКАЛЬНО).
# Через git базу не возим (PII), перенос — sqlite3 .backup + scp. Снимок
# безопасен при работающей локальной базе (WAL); на сервере приложение
# кратко останавливается (секунды), чтобы снять старый -wal/-shm рядом с новой
# базой, и ставится правильный владелец.
#
# Запуск из корня проекта:
#   bash deploy/push-db.sh user@host [путь-к-локальной-базе]   (ssh-сервер)
#   bash deploy/push-db.sh local  [путь-к-локальной-базе]      (сервер = эта машина, sudo)
# (обычно make deploy-db / make deploy-db-local)
set -euo pipefail

TARGET="${1:?ssh-цель (user@server) или «local»}"
SRC="${2:-jsnet/db/app.db}"
[ -f "$SRC" ] || { echo "локальной базы нет: $SRC"; exit 1; }
command -v sqlite3 >/dev/null 2>&1 || { echo "нужен sqlite3 CLI (sudo apt install sqlite3)"; exit 1; }

tmp="$(mktemp --suffix=.db)"
trap 'rm -f "$tmp"' EXIT
sqlite3 "$SRC" ".backup '$tmp'"

if [ "$TARGET" = "local" ]; then
  # сервер и есть эта машина: та же последовательность, что удалённая ветка,
  # только локально и через sudo (становится на время остановки сервиса)
  sudo systemctl stop inknet-web
  sudo install -o inknet -g inknet -m 640 "$tmp" /var/lib/inknet/app.db
  sudo rm -f /var/lib/inknet/app.db-wal /var/lib/inknet/app.db-shm
  sudo systemctl start inknet-web
  echo "база обновлена: $(ls -l /var/lib/inknet/app.db)"
  exit 0
fi

scp -q "$tmp" "$TARGET:/var/lib/inknet/app.db.new"
ssh "$TARGET" 'bash -s' <<'REMOTE'
set -eu
[ "$(id -u)" -eq 0 ] || [ -n "${INKNET_PUSH_AS_USER:-}" ] || { echo "нужен root на сервере (или INKNET_PUSH_AS_USER)"; exit 1; }
systemctl stop inknet-web
mv /var/lib/inknet/app.db.new /var/lib/inknet/app.db
rm -f /var/lib/inknet/app.db-wal /var/lib/inknet/app.db-shm
chown inknet:inknet /var/lib/inknet/app.db
chmod 640 /var/lib/inknet/app.db
systemctl start inknet-web
echo "база обновлена: $(ls -l /var/lib/inknet/app.db)"
REMOTE
