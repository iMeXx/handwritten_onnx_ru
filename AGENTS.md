# AGENTS.md — inknet-hybrid

Instructions for agent assistants. Status and entry point — `README.md`,
data audit — `docs/DATASET-AUDIT-2026-11-05.md`.

## Never launch training runs! If training is needed — tell the user!

## Context

This project is **handwriting collection and the dataset pipeline**. The line
recognition model is the production raster **`line_int8`**; its
training/finetuning/export live in `../inknet-train` (this repo does not train
models). The vector stroke branch and the ablation triad ("Variant B") are
**closed** per the 2026-11 experiment: on the person-disjoint triad raster wins
(raster 0.454 / vector 0.575 / fusion 0.657 CER), per-user logit blending (the
α grid) yields α\* = 0 — the vector branch adds nothing to the production
model. Vector/hybrid code (models, train/eval, export, the worker's hybrid
selector) has been **deleted**; do not retrain this direction without an
explicit request.

What lives here: the web app `jsnet/` (reference training, planned handwriting
collection, dataset statistics, line recognition), the dataset script
(`scripts/build-line-vector-ds.mjs` — a byte-copy from `../inknet-online`,
gated by `make parity`), writer merging (`scripts/merge-writers.mjs` + the
`writers-merge.json` map) and deployment (`deploy/`).

Related repositories (do not edit without an explicit request):

- `../inknet-train` — the production raster model `line_int8`: training,
  finetuning (`--init`, `--refs`, `--inbox`), ONNX export (fp32 → int8
  `quantize_dynamic`), evaluation, model deployment;
- `../inknet-online` — the sibling experiment repo (a vector-native model);
  shared dataset format `inknet-line-vectors/3` and the shared builder script
  (byte-copy, gated by `make parity`);
- `jsnet/` — a self-contained vendor data layer: reference JS extractors
  (`jsnet/src/{ink-features,ink-mlp}.js`), a snapshot of the handwriting
  database (`jsnet/db/app.db`, updated by copying the file from the build
  app), `vocab`/`terms`. The variable `WEB_ONNX` (= jsnet) — the name is fixed
  by the parity-pinned builder. The local web app also lives in
  `jsnet/{server,public}` (`make up`): reference training, planned collection,
  dataset statistics, line recognition (`/lines.html`: ONNX in a Web Worker,
  WASM-only; a single model — the production raster `line_int8`, the worker
  caches the session per url) — all of it writes to the same `app.db`;
  login by author name, no passwords, no registration (the session is the
  `sid` cookie); handwriting QC is nearest-neighbor over `ink-features`
  (not ONNX); the pen is configurable (the "✒ Pen" button),
  `js/{pen,settings,pensettings,capture}.js` are byte-copies — do not edit
  them in place (sync via `cp`); recognition:
  `js/{ctc-decode,line-preprocess,dict}.js` and `js/vendor/onnxruntime/` are
  byte-copies, while `js/{lines,line.worker}.js` are ports with adaptations
  (shell entry point, local ort without WebGPU), the pipeline is 1-to-1.

## Commands

```bash
make env              # effective configuration
make ds               # line dataset (login split) -> data/line-vectors/
make merge            # app.db snapshot -> data/db/app-merged.db, merge logins per person
make ds-merged        # person-disjoint dataset from the snapshot -> data/line-vectors-m/
make parity           # gate: the builder matches inknet-online (skips without the sibling repo)
make up               # jsnet web app as a background daemon: training/collection/stats/recognition
                      # (address in .env: WEB_HOST/WEB_PORT; down / status / log)
make clean            # delete the datasets (does NOT touch work-merged/raster)
make deploy-push      # rsync the code to the ssh server DEPLOY_SSH=user@host into /opt/inknet
make deploy-install   # install the web app on Ubuntu: node24 + systemd + nginx
                      # TLS: certbot or ready-made DEPLOY_CERT/DEPLOY_KEY
make deploy-db        # snapshot of the local app.db -> server (sqlite3 .backup, not cp)
make deploy-status    # systemctl state and the backup timer (deploy-logs — the service log)
                      # server = this machine: *-local targets (sudo, no ssh);
                      # domain/port/certs: DEPLOY_DOMAIN/WEB_PORT/DEPLOY_CERT/DEPLOY_KEY,
                      # details in docs/DEPLOY.md
```

The model (`line_int8`) is managed by the `../inknet-train` targets
(`train/eval/export/deploy`), not from here. The `.env` default is
`DATA_DIR=data/line-vectors-m` (the only valid split): the login split was
deleted as handwriting-leaky (rule 2), so the `ds` target is stopped by
`guard-merged` — it would rebuild the dataset by logins over the person-based
one; the deliberate escape hatch is `FORCE_MERGED=1`. The raster writer-disjoint
baseline weights live in `work-merged/raster/` (the comparison point when
finetuning `line_int8`, do not delete).

## Project rules

1. **Client contracts are untouchable**: blank=0, `line_vocab.json[i-1]`,
   `logits [B,T,C]`; the `/lines.html` worker loads `line_int8` (+`.ort`).
   Extending the alphabet = a synchronized edit of the vocab + the client +
   the trainer (`../inknet-train`).
2. **Evaluate writer-disjoint and per person, not per login**: `app.db` has
   several accounts per person (submissions from two devices), the builder
   groups by `username` → folds leak by handwriting. The grouping is done in
   the database snapshot (`make merge` → `data/db/app-merged.db`, the map
   `scripts/writers-merge.json`); the builder itself must not be touched (the
   `make parity` gate).
3. **Vendor instead of dependencies / byte-copy parity**: the dataset builder
   and `js/{pen,settings,pensettings,capture,ctc-decode,line-preprocess,dict}.js`
   + `js/vendor/onnxruntime/` are byte-copies, synced with `cp` — do not edit
   them in place; `make parity` catches builder drift from inknet-online.
4. **After a new model export** — synchronize
   `jsnet/public/models/line_int8.onnx` + `.ort` and bump `CACHE` in
   `jsnet/public/sw.js` (otherwise cache-first serves the old model).
5. **The model is never quantized for training**: finetuning is fp32 only
   (low LR, monitor CER on the author's own data to catch forgetting), then
   `quantize_dynamic` export and a 100% greedy int8↔fp32 match check — all of
   this lives in `../inknet-train`.

## Known pitfalls

- The parent projects left bug fingerprints: NaN→`null` in JSON, evaluating
  "against yourself", nested spread of typed arrays, mulberry32 correlated
  rows on arithmetic seeds. The full rules are in `../inknet-online/AGENTS.md`;
  when porting code, port the checks too.
- The small data volume gives CER spread ±0.15 per run — never compare models
  from a single training run; `TEST_LINES=120`, not 30.
- The vector branch does not train at all at the shared `LR=2e-3` (CTC collapse
  to a constant, CER ~0.9) — this direction is closed; the number is kept as a
  warning when working with inknet-online.
- `COLLATE NOCASE` in SQLite folds Latin letters only: Cyrillic author names
  «Инна»/«инна» are distinct strings, and one hand becomes two authors, which
  breaks writer-disjoint. Compare names in JS (`findUserByName` in
  `jsnet/server/index.js`), not `WHERE username = ?`.
- The POST body in `jsnet/server/index.js` accumulates `Buffer` chunks and is
  decoded once: string concatenation (`body += chunk`) mangles multibyte UTF-8
  at the chunk boundary, characters turn into U+FFFD, and a large tablet
  session submission is rejected with "invalid handwriting" (the client showed
  "the server did not accept it").
- Chromium on Android mixes pointer events with NaN coordinates into
  `getCoalescedEvents()`: the exponential smoothing filter in `capture.js`
  spreads NaN across the whole stroke, `JSON.stringify` writes `null` instead
  of numbers, and `validStrokes` rejects the ENTIRE `/api/train` batch because
  of one writing (all-or-nothing → "submitted 0, the rest stay in the buffer").
  `pointercancel` (the browser hijacks the gesture) left a 1-point stroke —
  the contract requires ≥2. Guards: coordinate finiteness in `_addPoint` and
  `≥2` in `finish` (`src/capture.js` — a byte-copy, sync with `cp`), batch
  sanitization before submission in `collect.js` (broken writings are removed
  from the buffer with their character list printed, instead of blocking the
  submission).

## Style

- Code comments are written in Russian and explain the "why" (methodology,
  limitations).
- Everything else is in English.
