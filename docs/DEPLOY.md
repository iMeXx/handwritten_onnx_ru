# Деплой: веб-приложение jsnet (Ubuntu, systemd + nginx)

Веб-приложение (страницы тренировки/сбора/статистики/распознавания) ставится на
Ubuntu 22.04/24.04: **systemd + nginx**, node слушает только `127.0.0.1`,
наружу выходит только HTTPS. npm-зависимостей нет вообще (чистый `node:http` +
встроенный `node:sqlite`, поэтому и нужен **Node ≥ 23** — ставится NodeSource
24 LTS).

Распознавание выполняется **на планшете** (модель считается в Web Worker,
onnxruntime-web WASM), сервер в инференсе не участвует: он отдаёт статику
(модель ~18 МБ кэшируется service worker'ом cache-first — после первого захода
страница распознавания работает офлайн) и принимает данные (`/api/*` — всегда
живая сеть). Серверного API распознавания нет и не планируется. Обучение на
сервере **не делается** (решение проекта): torch и тренировки живут на
dev-машине, веса публикуются на сервер как код (см. «Обновление кода/моделей»).
Требования к серверу отсюда скромные: node + nginx, GPU не нужен.

Код в `/opt/inknet` доставляется `rsync` — git-remote для деплоя не нужен;
база — **вне кодового дерева** в `/var/lib/inknet/app.db` (PII через git не
ездит).

## Два профиля установки

| | профиль A: отдельный ssh-сервер | профиль B: сервер = эта же машина |
|---|---|---|
| доставка кода | `make deploy-push DEPLOY_SSH=user@host` (rsync по ssh) | `make deploy-push-local` (sudo rsync в `/opt/inknet`) |
| установка | `make deploy-install ...` (sudo по ssh) | `make deploy-install-local` |
| перенос базы | `make deploy-db` | `make deploy-db-local` (sudo) |
| TLS | certbot `--nginx --redirect` | готовые сертификаты: `DEPLOY_CERT=/etc/nginx/ssl/fullchain.cer DEPLOY_KEY=/etc/nginx/ssl/imx.work.gd.key` |

Профиль B актуален для текущей боевой машины: на ней уже стоит nginx и
обслуживает `imx.work.gd` (SPA в `/var/www/tg_apps` + свой `/api/`), порт 22
не слушается (ssh-деплой на себя невозможен), сертификаты лежат в
`/etc/nginx/ssl` и покрывают wildcard **`*.imx.work.gd`** — приложение
заводится на поддомене `ink.imx.work.gd`, существующий сайт не трогается.
`install.sh` при пересечении `server_name` с чужим сайтом останавливается и
ничужую конфигурацию не удаляет.

## Предварительные условия

| пункт | что нужно |
|---|---|
| сервер | Ubuntu 22.04/24.04, sudo (root; для профиля A — ещё и ssh) |
| домен | профиль B: DNS-запись `ink.imx.work.gd` → тот же IP, что `imx.work.gd` (DDNS-панель work.gd); профиль A: A-запись своего домена. Проверка: `dig +short ink.imx.work.gd` |
| порты | 80 и 443 доходят до сервера (NAT: port-forward; `ufw allow 80,443`). Внутренний порт приложения (4173) наружу **не** открывается |
| планшеты | приложение по `https://ink.imx.work.gd`; первый визит качает модели ~18 МБ, дальше страница распознавания офлайн |

## Установка (профиль B, эта машина)

```bash
make deploy-push-local
make deploy-install-local \
  DEPLOY_CERT=/etc/nginx/ssl/fullchain.cer \
  DEPLOY_KEY=/etc/nginx/ssl/imx.work.gd.key
make deploy-db-local        # снимок текущей app.db (sqlite3 .backup) -> /var/lib/inknet
make deploy-status-local
```

Цели `*-local` вызывают `sudo` — терминал спросит пароль. Домен по умолчанию —
`ink.imx.work.gd` (`DEPLOY_DOMAIN=` перебивает).

## Установка (профиль A, ssh-сервер)

```bash
export DEPLOY_SSH=user@host DEPLOY_DOMAIN=app.example.org DEPLOY_EMAIL=you@ma.il

make deploy-push      # rsync исходников в /opt/inknet (без data/work/базы/.env/venv)
make deploy-install   # sudo install.sh: apt, node24, user inknet, systemd, nginx, certbot
make deploy-db        # снимок локальной app.db (sqlite3 .backup) -> /var/lib/inknet
make deploy-status / make deploy-logs
```

`install.sh` идемпотентен: повторный запуск обновляет юниты/nginx, не трогая
`/etc/inknet/env` и базу.

## Что ставится

| файл на сервере | источник | назначение |
|---|---|---|
| `/etc/systemd/system/inknet-web.service` | `deploy/inknet-web.service` | node под user `inknet`, `ProtectSystem=strict`, запись только в `/var/lib/inknet` (база вынесена из кодового дерева, PII); лог journald |
| `/etc/inknet/env` (root:inknet 640) | `deploy/env.server.example` | `HOST=127.0.0.1`, `PORT`, `COOKIE_SECURE=1`, `DB_PATH` (формат systemd: без инлайн-комментариев) |
| `/etc/nginx/sites-available/inknet` | `deploy/nginx-site-tls.conf.template` (готовые сертификаты) или `deploy/nginx-site.conf.template` (certbot) | proxy на 127.0.0.1:`PORT`; `client_max_body_size 64m` (большие сдачи с планшета); rate-limit `/api/` (вход по имени без пароля); `/sw.js` без кэша |
| TLS (профиль A) | certbot `--nginx --redirect` | сертификат + редирект 80→443, автопродление таймером certbot |
| TLS (профиль B) | готовые `DEPLOY_CERT/DEPLOY_KEY` | переиспользование существующего сертификата (wildcard покрывает поддомен); редирект 80→443 в шаблоне; продление — тем же механизмом, что основной сертификат |
| `/etc/systemd/system/inknet-backup.{service,timer}` | `deploy/inknet-backup.*` | ежедневно 03:15 `sqlite3 .backup` → `/var/backups/inknet/app-<stamp>.db.gz`, ротация 14 дней (`Persistent=true` — догонит после простоя) |

Локальная разработка не меняется: `make up/down` (nohup) остаются, сервер — systemd.

## Обновление кода/моделей

```bash
make deploy-push-local                        # профиль B (или deploy-push по ssh)
sudo systemctl restart inknet-web             # (make deploy-status-local для проверки)
```

Ключ публикации моделей (по README проекта — осознанное решение): новый вес
(обучение на dev-машине, экспорт в ONNX) кладётся в `jsnet/public/models/` +
поднимается версия кэша моделей в `jsnet/public/sw.js` (`inknet-jsnet-v1`),
иначе service worker на планшетах останется на старом `.onnx`. После этого —
`deploy-push*` + restart.

## Бэкап и восстановление

Снимок — только `sqlite3 .backup` (WAL), файлы `app.db*` копировать нельзя.

```bash
# забрать свежие снимки к себе
sudo cp -r /var/backups/inknet ./backup/     # профиль B (scp — для ssh-профиля)

# восстановление из снимка (профиль B): снимок поверх рабочей базы
gunzip -c app-2026-11-06-0315.db.gz > /tmp/app.db
bash deploy/push-db.sh local /tmp/app.db     # сервис кратко стоп/старт
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
make deploy-status-local      # юнит + таймер (deploy-logs-local — журнал)
ssh $DEPLOY_SSH "sudo systemctl status inknet-web"     # профиль A
sudo nginx -t
dig +short ink.imx.work.gd                              # DNS = публичный IP сервера
```

Частые грабли: certbot падает, если 80 не проброшен (NAT); старый service
worker на планшете (лечится версией кэша в `sw.js`); «сервер не принял сдачу»
при маленьком `client_max_body_size` (см. шаблоны `deploy/nginx-site*.template`);
на профиле B — не продленный основной сертификат (`/etc/nginx/ssl`): срок
проверяйте `openssl x509 -enddate -in /etc/nginx/ssl/fullchain.cer -noout`.
