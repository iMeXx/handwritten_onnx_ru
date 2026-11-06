#!/usr/bin/env python3
"""Обучение модели варианта B: один скрипт на три ablation-режима.

--mode raster   чистый растр (vendor LineNet — архитектура боевой модели)
--mode vector   чистый вектор (VecLineNet, архитектура inknet-online)
--mode fusion   HybridNet: растр + векторная ветка, общий CTC-хед

Все три режима видят ОДНИ строки, ОДНИ фолды (writer-disjoint), те же
гиперпараметры и метрики — отличие только во входе/архитектуре. Итоговая
триада сравнения: make ablation && make compare.

Запуск:
    make train                        # режим из .env (MODE=fusion)
    make train ARGS="--mode raster"
    make train ARGS="--runs 3 --epochs 300"
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hybridnet.hdset import PairDataset, collate, load_pairs
from hybridnet.hmodel import HybridNet, LineNet
from hybridnet.metrics import cer, greedy_ids
from hybridnet.vmodel import VecLineNet

HERE = Path(__file__).resolve().parent
VOCAB_PATH = HERE / "jsnet/public/models/line_vocab.json"


def load_vocab() -> list[str]:
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8"))


def text_of(ids: list[int], vocab: list[str]) -> str:
    return "".join(vocab[i - 1] for i in ids if i > 0)


def make_net(mode: str, n_classes: int, vec_drop: float = 0.0):
    if mode == "raster":
        return LineNet(n_classes)          # vendor-копия боевой архитектуры
    if mode == "vector":
        return VecLineNet(d_in=10, n_classes=n_classes)  # как в inknet-online
    return HybridNet(n_classes, vec_drop=vec_drop)


def model_inputs(mode: str, batch: dict, device: str | torch.device):
    if mode == "raster":
        return (batch["image"].to(device),)
    if mode == "vector":
        return (batch["feats"].to(device),)
    return (batch["image"].to(device), batch["feats"].to(device),
            batch["vec_lens"].to(device), batch["img_lens"].to(device))


def frame_lens(mode: str, batch: dict) -> torch.Tensor:
    return batch["vec_lens"] if mode == "vector" else batch["img_lens"]


def build_net(mode, net, batch, device):
    """forward net c единым вызовом для всех режимов."""
    return net(*model_inputs(mode, batch, device))


@torch.no_grad()
def evaluate(net, mode, loader, vocab, device):
    net.eval()
    refs, hyps = [], []
    for batch in loader:
        logits = build_net(mode, net, batch, device).cpu().numpy()
        lens = frame_lens(mode, batch).tolist()
        targets = batch["targets"].tolist()
        tlens = batch["classes_lens"].tolist()
        off = 0
        for b in range(logits.shape[0]):
            n = tlens[b]
            refs.append(text_of(targets[off:off + n], vocab))
            off += n
            hyps.append(text_of(greedy_ids(logits[b, : lens[b]]), vocab))
    acc = float(np.mean([r == h for r, h in zip(refs, hyps)])) if refs else 0.0
    return cer(refs, hyps), acc, list(zip(refs, hyps))


def train_fold(args, fold_info, vocab):
    fold = fold_info["fold"]
    with_raster = args.mode != "vector"
    tr_s, _ = load_pairs(args.data, fold, "train", with_raster)
    te_s, _ = load_pairs(args.data, fold, "test", with_raster)
    n_classes = len(vocab) + 1

    seed = getattr(args, "_seed_run", args.seed)
    torch.manual_seed(seed + fold)
    np.random.seed(seed + fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = make_net(args.mode, n_classes, vec_drop=getattr(args, "vec_drop", 0.0)).to(device)
    if args.mode == "fusion" and args.fuse_raster_start:
        n = net.init_as_raster()
        print(f"  {args.mode} fold{fold}: fuse инициализирован тождественно растровому "
              f"входу (векторные столбцы обнулены, {n} весов)")
    dl = DataLoader(PairDataset(tr_s), batch_size=args.batch, shuffle=True, collate_fn=collate)
    va_dl = DataLoader(PairDataset(te_s), batch_size=8, collate_fn=collate)
    loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    # Векторная ветка на общем LR=2e-3 не обучается вовсе (CTC-коллапс к гипотезе-
    # константе, CER ~0.9 даже на своих авторах), а растру этот LR нужен для
    # сходимости. Поэтому в fusion векторная ветка (vproj/vtcn) крутится отдельной
    # группой параметров со своим --lr-vector; raster/общий хед остаются на --lr.
    vgroups = (net.vproj, net.vtcn) if args.mode == "fusion" else ()
    if vgroups and args.lr_vector:
        vpar = [p for m in vgroups for p in m.parameters()]
        seen_ids = {id(p) for p in vpar}
        rest = [p for p in net.parameters() if id(p) not in seen_ids]
        opt = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                                 {"params": vpar, "lr": args.lr_vector}], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.mode == "fusion":   # фиксируем конфигурацию гибрида в логе (её ловит ablation-отчёт)
        print(f"  {args.mode} fold{fold}: lr={args.lr} lr_vector={args.lr_vector} "
              f"vec_drop={args.vec_drop} fuse_raster_start={int(args.fuse_raster_start)}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = Path(args.out) / args.mode / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    history, best_cer, best_ep, bad = [], 9.0, -1, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        tot = seen = 0
        for batch in dl:
            lp = net.log_probs(*model_inputs(args.mode, batch, device))
            lens = frame_lens(args.mode, batch).to(device)
            targets = batch["targets"].to(device)
            tlens = batch["classes_lens"].to(device)
            loss = loss_fn(lp, targets, lens, tlens)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * batch["vec_lens"].shape[0]
            seen += int(batch["vec_lens"].shape[0])
        sched.step()
        cer_v, acc_v, _ = evaluate(net, args.mode, va_dl, vocab, device)
        history.append({"epoch": epoch, "loss": tot / max(1, seen), "cer_test": cer_v, "acc_test": acc_v})
        if epoch % 10 == 0 or cer_v < best_cer:
            print(f"  {args.mode} fold{fold} ep{epoch:4d} loss {tot / max(1, seen):.3f} "
                  f"cer(test-writer {fold_info['testWriter']}) {cer_v:.3f} acc {acc_v:.3f}")
        if cer_v < best_cer - 1e-4:
            best_cer, best_ep, bad = cer_v, epoch, 0
            torch.save(net.state_dict(), out / "model-best.pt")
        else:
            bad += 1
        if bad >= args.patience:
            print(f"  {args.mode} fold{fold} ранний стоп (ep {best_ep} лучший)")
            break
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    print(f"{args.mode} fold{fold}: лучший cer {best_cer:.3f} (ep {best_ep}, {time.time() - t0:.0f}s)")
    return {"fold": fold, "testWriter": fold_info["testWriter"], "bestCer": best_cer, "bestEpoch": best_ep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", "data/line-vectors"))
    ap.add_argument("--out", default=os.environ.get("WORK_DIR", "work"))
    ap.add_argument("--mode", choices=["raster", "vector", "fusion"],
                    default=os.environ.get("MODE", "fusion"))
    ap.add_argument("--fold", default=os.environ.get("FOLD", "all"))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 300)))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("BATCH", 16)))
    ap.add_argument("--lr", type=float, default=float(os.environ.get("LR", 2e-3)))
    ap.add_argument("--lr-vector", dest="lr_vector", type=float,
                    default=float(os.environ.get("LR_VECTOR", 0)),
                    help="отдельный LR для векторной ветки в fusion (0 = общая группа)")
    ap.add_argument("--fuse-raster-start", dest="fuse_raster_start", action="store_true",
                    default=os.environ.get("FUSE_RASTER_START", "0") not in ("0", "", "false", "no"),
                    help="fusion: fuse стартует тождественным растровому входу (векторные столбцы весов = 0)")
    ap.add_argument("--vec-drop", dest="vec_drop", type=float,
                    default=float(os.environ.get("VEC_DROP", 0)),
                    help="fusion: доля примеров обучения с полностью обнулённым векторным потоком (modality dropout)")
    ap.add_argument("--patience", type=int, default=int(os.environ.get("PATIENCE", 60)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 7)))
    ap.add_argument("--runs", type=int, default=int(os.environ.get("RUNS", 1)),
                    help="повторов с разными сидами (шум малых данных ~±0.15 CER)")
    args = ap.parse_args()

    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") not in ("inknet-line-vectors/2", "inknet-line-vectors/3"):
        raise SystemExit(f"format {manifest.get('format')!r} — собрать датасет: make ds")
    vocab = load_vocab()
    per_fold: dict[int, dict] = {}
    for run in range(args.runs):
        args._seed_run = args.seed + 100 * run
        for fr in manifest["folds"]:
            if args.fold != "all" and int(args.fold) != fr["fold"]:
                continue
            print(f"=== {args.mode} fold {fr['fold']} run {run + 1}/{args.runs}: train {fr['trainLines']} строк у "
                  f"{', '.join(fr['trainWriters'])}, тест {fr['testLines']} у {fr['testWriter']} (seed {args._seed_run}) ===")
            r = train_fold(args, fr, vocab)
            e = per_fold.setdefault(fr["fold"], {"fold": fr["fold"], "testWriter": fr["testWriter"], "cers": []})
            e["cers"].append(r["bestCer"])
    for e in per_fold.values():
        e["meanCer"] = float(np.mean(e["cers"]))
        e["stdCer"] = float(np.std(e["cers"]))
    results = sorted(per_fold.values(), key=lambda e: e["fold"])
    summary = {"mode": args.mode, "runs": args.runs, "seed": args.seed, "results": results,
               "avgCer": float(np.mean([e["meanCer"] for e in results])) if results else None}
    mdir = Path(args.out) / args.mode
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if results:
        print(f"\n{args.mode} writer-disjoint ({args.runs} сид(ов)): " +
              ", ".join(f"фолд{e['fold']} {e['meanCer']:.3f}±{e['stdCer']:.3f}" for e in results) +
              f" | средний CER {summary['avgCer']:.3f}")


if __name__ == "__main__":
    main()
