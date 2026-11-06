#!/usr/bin/env python3
"""Ablation-триада raster / vector / fusion по writer-disjoint фолдам.

Читает work/<mode>/summary.json этого проекта. Критерий гейта M2 (docs/PLAN.md):
fusion лучше raster-only минимум на 5% ОТНОСИТЕЛЬНОГО CER; вес векторной ветки
<= 1 МБ int8; деградация без векторного входа — отдельная проверка eval
MODE=raster.

Запуск: make compare   (после make ablation)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODES = ("raster", "vector", "fusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("WORK_DIR", "work"))
    args = ap.parse_args()

    sums = {}
    for m in MODES:
        p = Path(args.work) / m / "summary.json"
        if p.exists():
            sums[m] = json.load(open(p))
    if not sums:
        raise SystemExit(f"нет summary в {args.work}/<mode>/ — сначала make ablation")

    head = "фолд  тест-автор  " + "  ".join(f"{m:>17s}" for m in MODES if m in sums)
    print(head)
    folds = sorted({f for s in sums.values() for f in [r["fold"] for r in s["results"]]})
    for fold in folds:
        row = [f"{fold:<5d}"]
        writer = ""
        for m in MODES:
            if m not in sums:
                continue
            r = next((x for x in sums[m]["results"] if x["fold"] == fold), None)
            if r:
                writer = r["testWriter"]
                row.append(f"{r['meanCer']:.3f}±{r['stdCer']:.3f}".rjust(17))
            else:
                row.append("—".rjust(17))
        print(" ".join([f"{writer:<10s}"] + row[1:]))
    for m in MODES:
        if m in sums:
            runs = sums[m].get("runs", 1)
            print(f"средн {m:<8s}: {sums[m]['avgCer']:.3f} (runs={runs})")

    if "fusion" in sums and "raster" in sums:
        f, r = sums["fusion"]["avgCer"], sums["raster"]["avgCer"]
        rel = (f - r) / r
        print(f"\nгейт M2: fusion против raster-only: {rel:+.1%} относительно (порог -5%)")
        if rel <= -0.05:
            print("  ПОКРЫТО: fusion уверенно лучше растра — идём в экспорт/интеграцию")
        elif rel < 0:
            print("  ШУМ/МАЛО: улучшение есть, но ниже порога — нужны данные (M0) и ещё сиды")
        else:
            print("  НЕ ПОКРЫТО: fusion не лучше растра — векторная ветка на этих данных бесполезна")
    if "vector" in sums and "raster" in sums:
        print(f"справка: vector против raster-only: {(sums['vector']['avgCer'] - sums['raster']['avgCer']) / sums['raster']['avgCer']:+.1%}")


if __name__ == "__main__":
    main()
