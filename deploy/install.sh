#!/usr/bin/env bash
# Инсталляция веб-приложения inknet-hybrid на Ubuntu 22.04/24.04 (root).
# Код на сервер доставляется отдельно (make deploy-push: rsync в /opt/inknet),
# скрипт идемпотентен: повторный запуск ничего не ломает.
#
# Запуск (обычно через make deploy-install DEPLOY_SSH=... DEPLOY_DOMAIN=...):
#   sudo bash deploy/install.sh --domain imx.work.gd --port 4173 --email you@ma.il [--with-train]
#
# --with-train: дополнительно .venv с torch CPU (этап 3, обучение на сервере).
set -euo pipefail

DOMAIN="" PORT="4173" EMAIL="" WITH_TRAIN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --domain) DOMAIN="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --email) EMAIL="$2"; shift 2;;
    --with-train) WITH_TRAIN=1; shift;;
    *) echo "неизвестный аргумент: $1" >&2; exit 1;;
  esac
done
[ -n "$DOMAIN" ] || { echo "нужен --domain (A-запись должна смотреть на этот сервер)"; exit 1; }
[ "$(id -u)" -eq 0 ] || { echo "запускать от root (sudo)"; exit 1; }
[ -f /opt/inknet/jsnet/server/index.js ] || { echo "кода нет: сначала «make deploy-push» (rsync в /opt/inknet)"; exit 1; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl ca-certificates nginx sqlite3 certbot python3-certbot-nginx python3-venv rsync

# Node >= 23: встроенный node:sqlite (jsnet/server/db.js). Ubuntu 24.04 даёт 18 —
# ставим NodeSource 24 LTS; уже стоящий свежий node (nodejs.org) не трогаем.
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
systemctl enable --now inknet-web
systemctl enable --now inknet-backup.timer

# nginx: site из шаблона + reload
sed -e "s/__DOMAIN__/$DOMAIN/g" -e "s/__PORT__/$PORT/g" \
  /opt/inknet/deploy/nginx-site.conf.template > /etc/nginx/sites-available/inknet
ln -sfn /etc/nginx/sites-available/inknet /etc/nginx/sites-enabled/inknet
nginx -t
systemctl reload nginx

# TLS: повторный выпуск не делаем (certbot продлевает сам по таймеру)
if ! certbot certificates 2>/dev/null | grep -q "Name: $DOMAIN"; then
  [ -n "$EMAIL" ] || echo "внимание: --email не дан, certbot спросит интерактивно"
  certbot --nginx -d "$DOMAIN" --redirect --noninteractive --agree-tos ${EMAIL:+-m "$EMAIL"} ${EMAIL:-}
fi

# этап 3: venv с torch CPU для обучения на сервере
if [ "$WITH_TRAIN" -eq 1 ]; then
  if [ ! -x /opt/inknet/.venv/bin/python ]; then
    cd /opt/inknet && sudo -u inknet env HOME=/var/lib/inknet make venv
  fi
  echo "venv: $(/opt/inknet/.venv/bin/python -c 'import torch; print(torch.__version__)')"
fi

IP_EXT="$(curl -fsS --max-time 5 https://ifconfig.me 2>/dev/null || echo '?')"
cat <<EOF

готово:
  приложение: https://$DOMAIN (nginx → 127.0.0.1:$PORT, TLS certbot)
  база:       $(grep -m1 '^DB_PATH=' /etc/inknet/env | cut -d= -f2) (бэкап daily, /var/backups/inknet)
  журнал:     journalctl -u inknet-web
  статус:     systemctl status inknet-web
проверьте, что внешние порты 80/443 доходят до этого сервера (NAT/файрвол),
и что A-запись $DOMAIN смотрит на $IP_EXT — планшеты открывают https://$DOMAIN.
EOF
