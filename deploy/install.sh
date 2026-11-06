#!/usr/bin/env bash
# Инсталляция веб-приложения inknet-hybrid на Ubuntu 22.04/24.04 (root).
# Код в /opt/inknet доставляется отдельно: на этой машине — «make deploy-push-local»,
# на удалённой — «make deploy-push» (rsync по ssh). Скрипт идемпотентен:
# повторный запуск обновляет юниты/nginx, НЕ трогая /etc/inknet/env и базу.
#
# Запуск:
#   sudo bash deploy/install.sh --domain ink.imx.work.gd [--port 4173]
#       [--cert /etc/nginx/ssl/fullchain.cer --key /etc/nginx/ssl/imx.work.gd.key]
#       [--email you@ma.il]
#
# TLS: с --cert/--key используются готовые сертификаты (на этой машине —
# wildcard *.imx.work.gd от /etc/nginx/ssl), certbot не нужен и не ставится;
# без них — certbot --nginx --redirect (нужен проброс порта 80).
#
# Обучение на сервере НЕ делается (решение проекта): torch живёт на dev-машине,
# веса публикуются как код (jsnet/public/models/ + поднимать версию кэша sw.js).
set -euo pipefail

DOMAIN="" PORT="4173" EMAIL="" CERT="" KEY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --email) EMAIL="$2"; shift 2;;
    --cert) CERT="$2"; shift 2;;
    --key) KEY="$2"; shift 2;;
    *) echo "неизвестный аргумент: $1" >&2; exit 1;;
  esac
done
[ -n "$DOMAIN" ] || { echo "нужен --domain (DNS-запись должна смотреть на этот сервер)"; exit 1; }
[ "$(id -u)" -eq 0 ] || { echo "запускать от root (sudo)"; exit 1; }
[ -f /opt/inknet/jsnet/server/index.js ] || { echo "кода нет: сначала «make deploy-push-local» (или deploy-push по ssh)"; exit 1; }
if [ -n "$CERT" ] || [ -n "$KEY" ]; then
  { [ -f "$CERT" ] && [ -f "$KEY" ]; } || { echo "--cert/--key: файлы не найдены ($CERT / $KEY)"; exit 1; }
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates nginx sqlite3 rsync
if [ -z "$CERT" ]; then
  apt-get install -y -qq certbot python3-certbot-nginx
fi

# Node >= 23: встроенный node:sqlite (jsnet/server/db.js). Под sudo/nvm node
# может не решаться — проверяем здесь; Ubuntu 24.04 даёт 18 — ставим
# NodeSource 24 LTS; уже стоящий свежий node (nodejs.org) не трогаем.
need_node=1
if command -v node >/dev/null 2>&1; then
  major="$(node -v | sed 's/^v//;s/\..*//')"
  [ "$major" -ge 23 ] && need_node=0
fi
if [ "$need_node" -eq 1 ]; then
  curl -fsSL https://deb.nodesource.com/setup_24.x | bash -
  apt-get install -y -qq nodejs
fi
echo "node: $(node -v)"

# пользователь-владелец: писать может только в /var/lib/inknet
id -u inknet >/dev/null 2>&1 || useradd --system --home-dir /var/lib/inknet --shell /usr/sbin/nologin inknet
mkdir -p /var/lib/inknet /etc/inknet /var/backups/inknet
chown inknet:inknet /var/lib/inknet /var/backups/inknet
chmod 750 /var/lib/inknet /var/backups/inknet

# /etc/inknet/env: создаём из шаблона (PORT подставляем), существующий НЕ трогаем
if [ ! -f /etc/inknet/env ]; then
  sed "s/^PORT=.*/PORT=$PORT/" /opt/inknet/deploy/env.server.example > /etc/inknet/env
  chown root:inknet /etc/inknet/env; chmod 640 /etc/inknet/env
fi

# systemd: юниты веб-приложения и бэкапа
install -m 644 /opt/inknet/deploy/inknet-web.service /etc/systemd/system/inknet-web.service
install -m 644 /opt/inknet/deploy/inknet-backup.service /etc/systemd/system/inknet-backup.service
install -m 644 /opt/inknet/deploy/inknet-backup.timer /etc/systemd/system/inknet-backup.timer
install -m 755 /opt/inknet/deploy/inknet-backup.sh /usr/local/bin/inknet-backup.sh
systemctl daemon-reload

# nginx: сайт из шаблона (TLS-шаблон — когда даны готовые сертификаты)
if grep -rlw "$DOMAIN" /etc/nginx/sites-enabled/ 2>/dev/null | grep -v "/inknet$" | grep -q .; then
  echo "домен $DOMAIN уже обслуживает другой сайт — не отключаем чужую конфигурацию молча:"
  grep -rlw "$DOMAIN" /etc/nginx/sites-enabled/ 2>/dev/null | grep -v "/inknet$" | sed 's/^/  /'
  echo "решите руками (обычно: взять приложение на поддомен) и повторите"
  exit 1
fi
if [ -n "$CERT" ]; then
  sed -e "s/__DOMAIN__/$DOMAIN/g" -e "s/__PORT__/$PORT/g" \
      -e "s#__CERT__#$CERT#g" -e "s#__KEY__#$KEY#g" \
    /opt/inknet/deploy/nginx-site-tls.conf.template > /etc/nginx/sites-available/inknet
else
  sed -e "s/__DOMAIN__/$DOMAIN/g" -e "s/__PORT__/$PORT/g" \
    /opt/inknet/deploy/nginx-site.conf.template > /etc/nginx/sites-available/inknet
fi
ln -sfn /etc/nginx/sites-available/inknet /etc/nginx/sites-enabled/inknet
nginx -t
systemctl reload nginx

# TLS certbot-режима: выпуск/прикрепление к нашему сайту (идемпотентно)
if [ -z "$CERT" ]; then
  if ! certbot certificates 2>/dev/null | grep -q "Name: $DOMAIN"; then
    [ -n "$EMAIL" ] || echo "внимание: --email не дан, certbot спросит интерактивно"
    certbot --nginx -d "$DOMAIN" --redirect --noninteractive --agree-tos ${EMAIL:+-m "$EMAIL"} ${EMAIL:-}
  fi
fi

systemctl enable --now inknet-web
systemctl enable --now inknet-backup.timer

cat <<EOF

готово:
  приложение: https://$DOMAIN (nginx → 127.0.0.1:$PORT)
  база:       $(grep -m1 '^DB_PATH=' /etc/inknet/env | cut -d= -f2) (бэкап daily, /var/backups/inknet)
  журнал:     journalctl -u inknet-web
  статус:     systemctl status inknet-web
проверьте DNS: $DOMAIN → публичный IP этого сервера, и что 80/443 доходят
сюда (NAT/файрвол) — планшеты открывают https://$DOMAIN.
EOF
