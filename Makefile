# inknet-hybrid — проект сбора написаний + датасетный конвейер. Модель
# распознавания — боевая растровая line_int8 (живёт в ../inknet-train,
# обучение/дообучение/экспорт делаются там). Векторная ветка и аблейшн-триада
# закрыты по итогам эксперимента (см. README), обучать здесь больше нечего:
# python-код (train/eval/compare/hybridnet) удалён.
#
# Здесь живут: веб-приложение jsnet (тренировка эталонов, сбор написаний,
# статистика, распознавание строк строковой моделью), скрипт датасета
# (байт-копия inknet-online — контроль make parity), склейка людей
# (merge) и деплой веб-приложения на Ubuntu.
#
# Конфигурация в .env (см. .env.example).
-include .env
export

NODE ?= node
WEB_ONNX ?= jsnet
ONLINE ?= ../inknet-online
DATA_DIR ?= data/line-vectors-m
DATA_DIR_M ?= data/line-vectors-m
# --- person-disjoint разрез (см. docs/DATASET-AUDIT-2026-11-05.md) ---
# в jsnet/db/app.db за одного человека по несколько логинов (etel1/Etel2,
# Vika1/Vika2, Dima1/Dima2 — сдали с iPad и планшета в один день), билдер
# группирует по username, поэтому фолды протекают. Склейка логинов делается в
# СНИМКЕ базы (scripts/merge-writers.mjs), сам билдер не правится — гейт parity
# остаётся зелёным.
MERGED_DB ?= data/db/app-merged.db
WRITERS_MAP ?= scripts/writers-merge.json
# веб-приложение jsnet (make up/down/status/log). Комментарии отдельной строкой: в make хвостовой
# комментарий после присваивания оставляет в значении пробелы — они потом
# всплывают в склеенных строках вроде http://$(WEB_HOST):$(WEB_PORT)
WEB_PORT ?= 3210
WEB_HOST ?= 127.0.0.1
# демон: pid/лог (make up / down / status / log)
WEB_PID ?= jsnet/logs/web.pid
WEB_LOG ?= jsnet/logs/web.log
ARGS ?=

.DEFAULT_GOAL := help

.PHONY: help env guard-merged ds merge clean-merge ds-merged parity up down status log clean \
        deploy-check deploy-push deploy-install deploy-db deploy-logs deploy-status \
        deploy-push-local deploy-install-local deploy-db-local deploy-logs-local deploy-status-local

help: ## список целей
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

env: ## показать эффективную конфигурацию (.env + дефолты)
	@printf 'NODE=%s WEB_ONNX=%s ONLINE=%s\nDATA_DIR=%s DATA_DIR_M=%s\nMERGED_DB=%s WRITERS_MAP=%s\nWEB_PORT=%s WEB_HOST=%s\n' \
	  "$(NODE)" "$(WEB_ONNX)" "$(ONLINE)" "$(DATA_DIR)" "$(DATA_DIR_M)" "$(MERGED_DB)" "$(WRITERS_MAP)" "$(WEB_PORT)" "$(WEB_HOST)"

# --- гейт: цель логин-разреза не должна писать в person-disjoint разрез ---
# .env переводит DATA_DIR на разрез по людям (он единственный валидный,
# правило 7), а ds работает именно с DATA_DIR: без гейта `make ds` пересобрал
# бы датасет по ЛОГИНАМ поверх data/line-vectors-m (один человек заведён
# двумя логинами — фолды протекают). Цель по людям — ds-merged.
guard-merged:
	@bad=""; \
	case "$(DATA_DIR)" in "$(DATA_DIR_M)") bad="DATA_DIR=$(DATA_DIR)";; esac; \
	test -z "$$bad" -o -n "$(FORCE_MERGED)" \
	  || { echo "эта цель пишет в $$bad, а это person-disjoint разрез: датасет соберётся по логинам и затрёт разрез по людям"; \
	       echo "нужно: ds-merged; осознанно пересобрать этим же скриптом: FORCE_MERGED=1"; exit 1; }

ds: guard-merged ## собрать датасет строк из снимка базы jsnet/db -> $(DATA_DIR)/
	$(NODE) scripts/build-line-vector-ds.mjs $(DATA_DIR)

merge: ## снимок базы jsnet/db/app.db + склейка логинов одного человека в одного автора -> $(MERGED_DB)
	@mkdir -p $$(dirname $(MERGED_DB))
	DB_SRC=$(WEB_ONNX)/db/app.db MERGED_DB=$(MERGED_DB) WRITERS_MAP=$(WRITERS_MAP) MIN_WRITER_CHARS=$(MIN_WRITER_CHARS) $(NODE) scripts/merge-writers.mjs

clean-merge: ## удалить снимок базы со склейкой (источник jsnet/db/app.db не трогается)
	rm -f $(MERGED_DB) $$(echo $(MERGED_DB) | sed 's/\.db$$/-merge.json/')

ds-merged: merge ## датасет по person-disjoint разрезу (фолды по людям) -> $(DATA_DIR_M)/
	DB_PATH=$(MERGED_DB) $(NODE) scripts/build-line-vector-ds.mjs $(DATA_DIR_M)

parity: ## контроль, что общие артефакты синхронны с inknet-online (read-only; skip без соседнего репозитория)
	@# всё в одной shell-строке: exit 0 в guard-строке завершила бы только её,
	@# и make пошёл бы дальше сравнить с несуществующим соседом
	@if [ ! -d "$(ONLINE)" ]; then \
	  echo "parity: $(ONLINE) нет — skip (гейт живёт на dev-машине с обоими репозиториями)"; \
	else \
	  cmp -s scripts/build-line-vector-ds.mjs $(ONLINE)/scripts/build-line-vector-ds.mjs \
	    && echo "builder: OK (байт-в-байт как в inknet-online)" \
	    || { echo "builder РАСХОДИТСЯ с $(ONLINE) — синхронизируйте вручную"; exit 1; }; \
	fi

up: ## поднять веб-приложение jsnet фоном (демоном): тренировка/сбор/статистика
	@if [ -f $(WEB_PID) ] && kill -0 $$(cat $(WEB_PID)) 2>/dev/null; then \
	  echo "уже запущено: pid $$(cat $(WEB_PID)) · make status / make down"; \
	else \
	  mkdir -p $$(dirname $(WEB_LOG)); \
	  case "$(WEB_HOST)" in 127.0.0.1|localhost) ;; *) echo "WEB_HOST=$(WEB_HOST): сервер доступен по сети (пишут с планшета), cookie без HTTPS; наружу не выставлять (COOKIE_SECURE=1 + TLS)" ;; esac; \
	  PORT=$(WEB_PORT) HOST=$(WEB_HOST) nohup $(NODE) jsnet/server/index.js >> $(WEB_LOG) 2>&1 & \
	  echo $$! > $(WEB_PID); \
	  sleep 1; \
	  if kill -0 $$(cat $(WEB_PID)) 2>/dev/null; then \
	    echo "веб-приложение поднято фоном · pid $$(cat $(WEB_PID)) · слушает $(WEB_HOST):$(WEB_PORT) · локально http://127.0.0.1:$(WEB_PORT) · лог $(WEB_LOG)"; \
	  else \
	    rm -f $(WEB_PID); \
	    echo "сервер не поднялся, последние строки $(WEB_LOG):" >&2; tail -n 5 $(WEB_LOG) >&2; exit 1; \
	  fi; \
	fi

down: ## остановить фоновое веб-приложение
	@if [ -f $(WEB_PID) ] && kill -0 $$(cat $(WEB_PID)) 2>/dev/null; then \
	  WPID=$$(cat $(WEB_PID)); kill $$WPID; \
	  for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $$WPID 2>/dev/null || break; sleep 0.5; done; \
	  if kill -0 $$WPID 2>/dev/null; then kill -9 $$WPID; echo "остановлено жёстко (pid $$WPID)"; \
	  else echo "остановлено (pid $$WPID)"; fi; \
	else \
	  echo "не запущено"; \
	fi
	@rm -f $(WEB_PID)

status: ## состояние фонового веб-приложения + последние строки лога
	@if [ -f $(WEB_PID) ] && kill -0 $$(cat $(WEB_PID)) 2>/dev/null; then \
	  echo "запущено: pid $$(cat $(WEB_PID)) · адрес и база — в логе ниже (make log)"; \
	else \
	  echo "не запущено (make up — поднять)"; \
	fi
	@test -f $(WEB_LOG) && tail -n 3 $(WEB_LOG) || true

log: ## последние 30 строк лога веб-приложения
	@test -f $(WEB_LOG) && tail -n 30 $(WEB_LOG) || echo "лога ещё нет ($(WEB_LOG))"

clean: ## удалить датасеты (обычный и person-disjoint); веса work-merged/raster НЕ трогает
	@echo "удаляю: $(DATA_DIR) $(DATA_DIR_M)"
	@# work-merged/raster — writer-disjoint бейзлайн растра (сравнение при
	@# дообучении line_int8), в clean не входит; удалять осознанно руками
	rm -rf $(DATA_DIR) $(DATA_DIR_M)

# --- деплой на Ubuntu (этап 2): доставка кода rsync (git-remote и ssh для
# деплоя не обязательны: targets *-local для случая «сервер = эта машина»),
# установка deploy/install.sh. DEPLOY_SSH=user@host для ssh-варианта.
# TLS: DEPLOY_CERT/DEPLOY_KEY — готовые сертификаты (на этой машине wildcard
# *.imx.work.gd в /etc/nginx/ssl); без них install.sh зовёт certbot.
# Подробности — docs/DEPLOY.md (sudo на сервере: root или NOPASSWD) ---
DEPLOY_SSH ?=
DEPLOY_DOMAIN ?= ink.imx.work.gd
DEPLOY_EMAIL ?=
DEPLOY_CERT ?=
DEPLOY_KEY ?=
DEPLOY_TLS := $(if $(DEPLOY_CERT),--cert $(DEPLOY_CERT) --key $(DEPLOY_KEY),)

deploy-check:
	@[ -n "$(DEPLOY_SSH)" ] || { echo "DEPLOY_SSH не задан: make deploy-push DEPLOY_SSH=user@host (или цель *-local)"; exit 1; }

deploy-push: deploy-check ## rsync кода на ssh-сервер в /opt/inknet (исходники: без data/work/базы/venv/.env)
	# --delete без --delete-excluded: исключённое (venv, .env, db) на сервере не удаляется
	rsync -az --delete \
	  --exclude .git --exclude .venv --exclude data --exclude work --exclude 'work-*' \
	  --exclude jsnet/db --exclude jsnet/logs --exclude export --exclude __pycache__ \
	  --exclude .env ./ $(DEPLOY_SSH):/opt/inknet/

deploy-push-local: ## rsync кода на ЭТУ машину в /opt/inknet (sudo; сервер = dev-машина)
	# --delete без --delete-excluded: исключённое (в т.ч. .venv, .env, база) не удаляется
	sudo rsync -az --delete \
	  --exclude .git --exclude .venv --exclude data --exclude work --exclude 'work-*' \
	  --exclude jsnet/db --exclude jsnet/logs --exclude export --exclude __pycache__ \
	  --exclude .env ./ /opt/inknet/

deploy-install: deploy-check ## установка на ssh-сервере: apt, node24, systemd, nginx + TLS (certbot или DEPLOY_CERT/KEY)
	ssh $(DEPLOY_SSH) "sudo bash /opt/inknet/deploy/install.sh --domain $(DEPLOY_DOMAIN) --port $(WEB_PORT) $(if $(DEPLOY_EMAIL),--email $(DEPLOY_EMAIL),)$(DEPLOY_TLS)"

deploy-install-local: ## установка на ЭТОЙ машине: apt, node24, systemd, nginx + TLS (DEPLOY_CERT/KEY — готовые сертификаты)
	sudo bash deploy/install.sh --domain $(DEPLOY_DOMAIN) --port $(WEB_PORT) $(if $(DEPLOY_EMAIL),--email $(DEPLOY_EMAIL),)$(DEPLOY_TLS)

deploy-db: deploy-check ## снимок локальной app.db -> ssh-сервер (sqlite3 .backup; сервис кратко стоп/старт)
	bash deploy/push-db.sh $(DEPLOY_SSH) $(WEB_ONNX)/db/app.db

deploy-db-local: ## снимок локальной app.db на ЭТУ машину (sqlite3 .backup; sudo; сервис кратко стоп/старт)
	bash deploy/push-db.sh local $(WEB_ONNX)/db/app.db

deploy-logs: deploy-check ## последние 40 строк журнала веб-сервиса на ssh-сервере
	ssh $(DEPLOY_SSH) "sudo journalctl -u inknet-web -n 40 --no-pager"

deploy-logs-local: ## журнал веб-сервиса на ЭТОЙ машине
	sudo journalctl -u inknet-web -n 40 --no-pager

deploy-status: deploy-check ## состояние веб-сервиса и таймера бэкапа на ssh-сервере
	ssh $(DEPLOY_SSH) "sudo systemctl status inknet-web --no-pager -n 3; systemctl list-timers inknet-backup.timer --no-pager | head -4"

deploy-status-local: ## состояние веб-сервиса и таймера бэкапа на ЭТОЙ машине
	sudo systemctl status inknet-web --no-pager -n 3; systemctl list-timers inknet-backup.timer --no-pager | head -4
