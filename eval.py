#!/usr/bin/env python3
"""Оценка обученной модели (любой режим) + примеры ref->hyp.
Запуск: make eval [MODE=fusion|vector|raster]"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hybridnet.hdset import PairDataset, collate, load_pairs
from hybridnet.metrics import cer
from train import evaluate, load_vocab, make_net

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.environ.get("DATA_DIR", "data/line-vectors"))
ap.add_argument("--work", default=os.environ.get("WORK_DIR", "work"))
ap.add_argument("--mode", choices=["raster", "vector", "fusion"], default=os.environ.get("MODE", "fusion"))
ap.add_argument("--fold", default="all", help="номер фолда или all")
args = ap.parse_args()

manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
vocab = load_vocab()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for fr in manifest["folds"]:
    if args.fold != "all" and int(args.fold) != fr["fold"]:
        continue
    ckpt = Path(args.work) / args.mode / f"fold{fr['fold']}" / "model-best.pt"
    if not ckpt.exists():
        print(f"{args.mode} fold{fr['fold']}: нет {ckpt} — сначала make train MODE={args.mode}")
        continue
    with_raster = args.mode != "vector"
    te_s, doc = load_pairs(args.data, fr["fold"], "test", with_raster)
    net = make_net(args.mode, len(vocab) + 1).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    dl = DataLoader(PairDataset(te_s), batch_size=8, collate_fn=collate)
    c, acc, pairs = evaluate(net, args.mode, dl, vocab, device)
    rows.append({"fold": fr["fold"], "writer": fr["testWriter"], "cer": c, "acc": acc})
    print(f"{args.mode} fold{fr['fold']} тест {fr['testWriter']}: cer {c:.3f} acc {acc:.3f}")
    for ref, hyp in pairs[:6]:
        print(f"   {'OK' if ref == hyp else '  '} {ref:24s} -> {hyp}")
if rows:
    print(f"\n{args.mode}: средний writer-disjoint CER {np.mean([r['cer'] for r in rows]):.3f}")
