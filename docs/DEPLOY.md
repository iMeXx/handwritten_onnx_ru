# Деплой на Ubuntu (этап 2): веб-приложение jsnet

Веб-приложение (страницы тренировки/сбора/статистики/распознавания) ставится на
чистый Ubuntu 22.04/24.04: **systemd + nginx + Let's Encrypt**, node слушает
только `127.0.0.1`, наружу выходит только HTTPS. npm-зависимостей нет вообще
(чистый `node:http` + встроенный `node:sqlite`, поэтому и нужен **Node ≥ 23** —
ставится NodeSource 24 LTS).

Код доставляется `rsync` (`make deploy-push`) — git-remote для деплоя не нужен;
на сервере код лежит в `/opt/inknet`, база — **вне кодового дерева** в
`/var/lib/inknet/app.db` (PII через git не ездит).

Распознавание выполняется **на планшете** (модель считается в Web Worker,
onnxruntime-web WASM), сервер в инференсе не участвует: он отдаёт статику
(модель ~18 МБ кэшируется service worker'ом cache-first — после первого захода
страница распознавания работает офлайн) и принимает данные (`/api/*` — всегда
живая сеть). Серверного API распознавания нет и не планируется; требования к
серверу отсюда скромные: node + nginx, GPU не нужен (обучение на сервере —
torch CPU, этап 3).

## Предварительные условия

| пункт | что нужно |
|---|---|
| сервер | Ubuntu 22.04/24.04, ssh с sudo (root или NOPASSWD для `deploy/install.sh`) |
| домен | A-запись `imx.work.gd` → публичный IP; проверить: `dig +short imx.work.gd` |
| порты | 80 и 443 проброшены/открыты до сервера (NAT: port-forward на сервер; `ufw allow 80,443`). Внутренний порт приложения (4173) наружу **не** открывается |
| планшеты | та же сеть или Интернет — приложение по `https://imx.work.gd`; первый визит качает модели ~18 МБ |

Пусть `DEPLOY_SSH=user@host` — ssh-цель сервера. Все команды ниже — с dev-машины
из корня проекта.

## Установка

```bash
export DEPLOY_SSH=user@host DEPLOY_DOMAIN=imx.work.gd DEPLOY_EMAIL=you@ma.il

make deploy-push      # rsync исходников в /opt/inknet (без data/work/базы/.env/venv)
make deploy-install   # sudo install.sh: apt, node24, user inknet, systemd, nginx, certbot
make deploy-db        # снимок локальной app.db (sqlite3 .backup) -> /var/lib/inknet
make deploy-status    # systemctl status + таймер бэкапа
make deploy-logs      # журнал сервиса
```

`install.sh` идемпотентен: повторный запуск обновляет код/юниты/nginx, не трогая
`/etc/inknet/env` и базу. Для этапа 3 (обучение на сервере, torch CPU):
`make deploy-install ARGS`-стиль недоступен — вызов напрямую:
`ssh $DEPLOY_SSH "sudo bash /opt/inknet/deploy/install.sh --domain $DEPLOY_DOMAIN --port 4173 --with-train"`.

## Что ставится

| файл на сервере | источник | назначение |
|---|---|---|
| `/etc/systemd/system/inknet-web.service` | `deploy/inknet-web.service` | node под user `inknet`, `ProtectSystem=strict`, запись только в `/var/lib/inknet`; лог в journald |
| `/etc/inknet/env` (root:inknet 640) | `deploy/env.server.example` | `HOST=127.0.0.1`, `PORT`, `COOKIE_SECURE=1`, `DB_PATH` (формат systemd: без инлайн-комментариев) |
| `/etc/nginx/sites-available/inknet` | `deploy/nginx-site.conf.template` | proxy на 127.0.0.1:`PORT`; `client_max_body_size 64m` (большие сдачи с планшета); rate-limit `/api/` (вход по имени без пароля); `/sw.js` без кэша |
| TLS | certbot `--nginx --redirect` | сертификат + редирект 80→443, автопродление таймером certbot |
| `/etc/systemd/system/inknet-backup.{service,timer}` | `deploy/inknet-backup.*` | ежедневно 03:15 `sqlite3 .backup` → `/var/backups/inknet/app-<stamp>.db.gz`, ротация 14 дней (`Persistent=true` — догонит после простоя) |

Локальная разработка не меняется: `make up/down` (nohup) остаются, сервер — systemd.

## Обновление кода/моделей

```bash
make deploy-push          # после коммита — обновить код
ssh $DEPLOY_SSH "sudo systemctl restart inknet-web"
```

Ключ публикации моделей (этап 3, по README проекта — осознанное решение): новый
вес кладётся в `jsnet/public/models/` + поднимается версия кэша моделей в
`jsnet/public/sw.js` (`inknet-models-v2`), иначе service worker на планшетах
останется на старом `.onnx`.

## Бэкап и восстановление

Снимок — только `sqlite3 .backup` (WAL), файлы `app.db*` копировать нельзя.

```bash
# забрать свежий снимок к себе
scp $DEPLOY_SSH:/var/backups/inknet/app-*.db.gz ./backup/

# восстановление (снимок поверх рабочей базы; сервер приложения останавливается)
gunzip -c app-2026-11-06-0315.db.gz > /tmp/app.db
scp /tmp/app.db $DEPLOY_SSH:/var/lib/inknet/app.db.new
ssh $DEPLOY_SSH 'sudo systemctl stop inknet-web && sudo mv /var/lib/inknet/app.db.new /var/lib/inknet/app.db && sudo rm -f /var/lib/inknet/app.db-wal /var/lib/inknet/app.db-shm && sudo chown inknet:inknet /var/lib/inknet/app.db && sudo chmod 640 /var/lib/inknet/app.db && sudo systemctl start inknet-web'
```

## Безопасность (что осознанно принято)

- Вход по имени без пароля — модель сбора почерка в своей сети; наружу выход
  только за TLS (`COOKIE_SECURE=1` в env сервера — cookie по HTTP не уходит),
  rate-limit nginx на `/api/` (8 r/s + burst 20).
- Почерки и имена авторов — PII: база живёт в `/var/lib/inknet` (chmod 750,
  владелец `inknet`), бэкапы сжаты в `/var/backups/inknet`; git-репозиторий
  приватный (`scripts/writers-merge.json` содержит имена).
- Сервис без root, ФС только для чтения, запись только в базу.

## Диагностика

```bash
make deploy-status            # юнит + таймер
make deploy-logs              # journalctl -u inknet-web
ssh $DEPLOY_SSH "sudo nginx -t && sudo certbot certificates"
ssh $DEPLOY_SSH "sudo certbot renew --dry-run"   # продление TLS
dig +short imx.work.gd                            # A-запись = публичный IP сервера
```

Частые грабли: certbot падает, если 80 не проброшен (NAT); старый service worker
на планшете (лечится версией кэша в `sw.js`); «сервер не принял сдачу» при
маленьком `client_max_body_size` (см. `deploy/nginx-site.conf.template`).
