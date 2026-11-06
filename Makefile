# inknet-hybrid — вариант B: растровая CNN + векторная ветка штрихов, общий
# CTC-хед. САМОСТОЯТЕЛЬНЫЙ проект: свой скрипт датасета (копия inknet-online,
# байт-идентичная — цель parity), свои vendor-порты растра/LineNet, своё
# обучение трёх режимов (raster / vector / fusion) и своя оценка.
#
# Конфигурация в .env (см. .env.example). Параллельно с inknet-online гоняется
# без пересечений: свои DATA_DIR/WORK_DIR, никаких кросс-записей.
-include .env
export

PY ?= ../test-py/venv/bin/python
NODE ?= node
WEB_ONNX ?= jsnet
ONLINE ?= ../inknet-online
DATA_DIR ?= data/line-vectors
WORK_DIR ?= work
MODE ?= fusion
MODES ?= raster vector fusion
EPOCHS ?= 300
BATCH ?= 16
LR ?= 2e-3
PATIENCE ?= 60
SEED ?= 7
RUNS ?= 1
# FOLD нужен export-onnx (один фолд для онзк-веса). Осторожно: из-за `export`
# в начале файла он попадает в окружение всех рецептов, а train.py берёт
# --fold из окружения (FOLD) — без явного --fold обучение молча идёт по
# ОДНОМУ фолду, хотя гейт M2 требует все фолды. Поэтому обучение всегда
# получает TRAIN_FOLD явно.
FOLD ?= 0
TRAIN_FOLD ?= all
# веб-приложение jsnet (make up/down/status/log). Комментарии отдельной строкой: в make хвостовой
# комментарий после присваивания оставляет в значении пробелы — они потом
# всплывают в склеенных строках вроде http://$(WEB_HOST):$(WEB_PORT)
WEB_PORT ?= 3210
WEB_HOST ?= 127.0.0.1
# демон: pid/лог (make up / down / status / log)
WEB_PID ?= jsnet/logs/web.pid
WEB_LOG ?= jsnet/logs/web.log
ARGS ?=
# --- person-disjoint разрез (см. docs/DATASET-AUDIT-2026-11-05.md) ---
# в jsnet/db/app.db за одного человека по несколько логинов (etel1/Etel2,
# Vika1/Vika2, Dima1/Dima2 — сдали с iPad и планшета в один день), билдер
# группирует по username, поэтому фолды протекают. Склейка логинов делается в
# СНИМКЕ базы (scripts/merge-writers.mjs), сам билдер не правится — гейт parity
# остаётся зелёным.
MERGED_DB ?= data/db/app-merged.db
WRITERS_MAP ?= scripts/writers-merge.json
DATA_DIR_M ?= data/line-vectors-m
WORK_DIR_M ?= work-merged
# векторная ветка на LR=2e-3 не обучается вовсе (CER ~0.9, гипотеза-константа;
# при 2e-4 — 0.27 на том же разрезе), поэтому для vector LR отдельный
LR_VECTOR ?= 2e-4
# и на низком LR вектор сходится медленнее (лучшие CER на 300–400 эпохах)
EPOCHS_VECTOR ?= 400
PATIENCE_VECTOR ?= 120
# fusion: 1 = fuse стартует тождественно растровому входу (векторные столбцы весов
# обнулены), то есть гибрид начинается как чистый растр, а вектор добавляется
# постепенно. Лечит CTC-коллапс гибрида на общем LR. Архитектура и ONNX-граф не
# меняются — меняется только инициализация.
FUSE_RASTER_START ?= 0
FUSE_FLAG = $(if $(filter 1 true yes on,$(FUSE_RASTER_START)),--fuse-raster-start,)
# fusion: доля примеров обучения с обнулённым векторным потоком целиком (modality
# dropout). Без него вектор = «отпечаток автора»: гибрид заучивает авторов обучения
# и на новом человеке хуже чистого растра. 0 = выключено.
VEC_DROP ?= 0

TRAIN_ARGS = --data $(DATA_DIR) --out $(WORK_DIR) --epochs $(EPOCHS) \
             --batch $(BATCH) --lr $(LR) --lr-vector $(LR_VECTOR) --patience $(PATIENCE) \
             --seed $(SEED) --runs $(RUNS) --fold $(TRAIN_FOLD) $(FUSE_FLAG) --vec-drop $(VEC_DROP)

.DEFAULT_GOAL := help

.PHONY: help env ds train eval ablation compare smoke parity export-onnx up down status log clean \
        merge ds-merged train-merged eval-merged ablation-merged compare-merged clean-merge smoke-merged

help: ## список целей
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

env: ## показать эффективную конфигурацию (.env + дефолты)
	@printf 'PY=%s NODE=%s WEB_ONNX=%s\nDATA_DIR=%s WORK_DIR=%s MODE=%s\nEPOCHS=%s EPOCHS_VECTOR=%s BATCH=%s LR=%s LR_VECTOR=%s PATIENCE=%s PATIENCE_VECTOR=%s SEED=%s RUNS=%s TRAIN_FOLD=%s FOLD=%s\nMERGED_DB=%s WRITERS_MAP=%s DATA_DIR_M=%s WORK_DIR_M=%s\nWEB_PORT=%s WEB_HOST=%s\n' \
	  "$(PY)" "$(NODE)" "$(WEB_ONNX)" "$(DATA_DIR)" "$(WORK_DIR)" "$(MODE)" "$(EPOCHS)" "$(EPOCHS_VECTOR)" "$(BATCH)" "$(LR)" "$(LR_VECTOR)" "$(PATIENCE)" "$(PATIENCE_VECTOR)" "$(SEED)" "$(RUNS)" "$(TRAIN_FOLD)" "$(FOLD)" "$(MERGED_DB)" "$(WRITERS_MAP)" "$(DATA_DIR_M)" "$(WORK_DIR_M)" "$(WEB_PORT)" "$(WEB_HOST)"

ds: ## собрать общий датасет векторных строк из снимка базы jsnet/db -> $(DATA_DIR)/
	$(NODE) scripts/build-line-vector-ds.mjs $(DATA_DIR)

train: ## обучение одного режима (MODE=fusion|vector|raster) по всем фолдам
	$(PY) train.py $(TRAIN_ARGS) --mode $(MODE) $(ARGS)

eval: ## оценка весов режима MODE + примеры ref->hyp
	$(PY) eval.py --data $(DATA_DIR) --work $(WORK_DIR) --mode $(MODE) $(ARGS)

ablation: ## триада raster+vector+fusion на одних строках/сидах, затем сравнение
	@for m in $(MODES); do \
	  echo "=== ablation: $$m ==="; \
	  $(PY) train.py $(TRAIN_ARGS) --mode $$m $(ARGS) || exit 1; \
	done
	$(PY) compare.py --work $(WORK_DIR)

compare: ## вердикт по work/<mode>/summary.json (гейт M2: -5% относительно raster)
	$(PY) compare.py --work $(WORK_DIR)

smoke: ds train eval ## быстрый прогон конвейера (укоротите эры: ARGS="--epochs 30")

merge: ## снимок базы jsnet/db/app.db + склейка логинов одного человека в одного автора -> $(MERGED_DB)
	@mkdir -p $$(dirname $(MERGED_DB))
	DB_SRC=$(WEB_ONNX)/db/app.db MERGED_DB=$(MERGED_DB) WRITERS_MAP=$(WRITERS_MAP) MIN_WRITER_CHARS=$(MIN_WRITER_CHARS) $(NODE) scripts/merge-writers.mjs

clean-merge: ## удалить снимок базы со склейкой (источник jsnet/db/app.db не трогается)
	rm -f $(MERGED_DB) $$(echo $(MERGED_DB) | sed 's/\.db$$/-merge.json/')

ds-merged: merge ## датасет по person-disjoint разрезу (фолды по людям) -> $(DATA_DIR_M)/
	DB_PATH=$(MERGED_DB) $(NODE) scripts/build-line-vector-ds.mjs $(DATA_DIR_M)

train-merged: ## обучение одного режима (MODE=...) на person-disjoint датасете
	@case "$(MODE)" in \
	  vector) lr=$(LR_VECTOR); ep=$(EPOCHS_VECTOR); pat=$(PATIENCE_VECTOR);; \
	  *) lr=$(LR); ep=$(EPOCHS); pat=$(PATIENCE);; \
	esac; \
	echo "train: $(MODE) (lr=$$lr epochs=$$ep patience=$$pat fold=$(TRAIN_FOLD))"; \
	$(PY) train.py --data $(DATA_DIR_M) --out $(WORK_DIR_M) --epochs $$ep --batch $(BATCH) --lr $$lr --lr-vector $(LR_VECTOR) --patience $$pat --seed $(SEED) --runs $(RUNS) --fold $(TRAIN_FOLD) --mode "$(MODE)" $(FUSE_FLAG) --vec-drop $(VEC_DROP) $(ARGS)

eval-merged: ## оценка весов режима MODE на person-disjoint датасете + примеры
	$(PY) eval.py --data $(DATA_DIR_M) --work $(WORK_DIR_M) --mode $(MODE) $(ARGS)

ablation-merged: ds-merged ## триада raster+vector+fusion по ЛЮДЯМ (LR/эпохи для vector отдельные) + вердикт
	@for m in $(MODES); do \
	  case $$m in \
	    vector) lr=$(LR_VECTOR); ep=$(EPOCHS_VECTOR); pat=$(PATIENCE_VECTOR);; \
	    *) lr=$(LR); ep=$(EPOCHS); pat=$(PATIENCE);; \
	  esac; \
	  echo "=== ablation: $$m (lr=$$lr epochs=$$ep patience=$$pat fold=$(TRAIN_FOLD)) ==="; \
	  $(PY) train.py --data $(DATA_DIR_M) --out $(WORK_DIR_M) --epochs $$ep --batch $(BATCH) --lr $$lr --lr-vector $(LR_VECTOR) --patience $$pat --seed $(SEED) --runs $(RUNS) --fold $(TRAIN_FOLD) --mode $$m $(FUSE_FLAG) --vec-drop $(VEC_DROP) $(ARGS) || exit 1; \
	done
	$(PY) compare.py --work $(WORK_DIR_M)

compare-merged: ## вердикт по $(WORK_DIR_M)/<mode>/summary.json (гейт M2: -5% относительно raster)
	$(PY) compare.py --work $(WORK_DIR_M)

smoke-merged: ds-merged ## быстрая проверка конвейера на людях: триада, 1 фолд, 1 сид
	# эпохи не укорачиваем: CTC-раскатка зависит от числа шагов, а на 600 строках
	# фолда шаг в 2.5 раза короче, чем на 1500 (60 эпох => недотрен, CER ~0.9)
	@for m in $(MODES); do \
	  case $$m in \
	    vector) lr=$(LR_VECTOR); ep=$(EPOCHS_VECTOR); pat=$(PATIENCE_VECTOR);; \
	    *) lr=$(LR); ep=$(EPOCHS); pat=$(PATIENCE);; \
	  esac; \
	  echo "=== smoke: $$m (lr=$$lr, epochs=$$ep) ==="; \
	  $(PY) train.py --data $(DATA_DIR_M) --out $(WORK_DIR_M) --epochs $$ep --batch $(BATCH) --lr $$lr --patience $$pat --seed $(SEED) --runs 1 --fold 0 --mode $$m || exit 1; \
	done
	$(PY) compare.py --work $(WORK_DIR_M)

parity: ## контроль, что общие артефакты синхронны с inknet-online (read-only)
	@cmp -s scripts/build-line-vector-ds.mjs $(ONLINE)/scripts/build-line-vector-ds.mjs \
	  && echo "builder: OK (байт-в-байт как в inknet-online)" \
	  || { echo "builder РАСХОДИТСЯ с $(ONLINE) — синхронизируйте вручную"; exit 1; }
	@cmp -s hybridnet/vendor/raster.py $(ONLINE)/vecnet/vendor/raster.py \
	  && echo "vendor raster: OK" \
	  || { echo "vendor raster РАСХОДИТСЯ с $(ONLINE)"; exit 1; }

export-onnx: ## экспорт HybridNet (fusion, вес фолда FOLD) в ONNX (export/) + std/манифест
	$(PY) export-onnx.py --data $(DATA_DIR) --work $(WORK_DIR) --fold $(FOLD)

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

clean: ## удалить датасеты (обычный и person-disjoint) и веса
	rm -rf $(DATA_DIR) $(WORK_DIR) $(DATA_DIR_M) $(WORK_DIR_M)
