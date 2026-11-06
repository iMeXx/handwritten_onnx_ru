#!/usr/bin/env python3
"""Экспорт HybridNet (режим fusion) в ONNX (M3/M4) для веб-клиента.

Граф — точное повторение hmodel.forward для B=1 (клиентский путь):
  image [1,1,64,W] -> CNN+proj -> [1,T,256],  T = W//4 (два maxpool)
  feats [1,F,10]   -> векторная ветка TCN -> [1,F,vh]
  выравнивание: линейная интерполяция [1,vh,F] -> [1,vh,T] (align_corners=False)
  — та же интерполяция по времени, что при обучении (правило parity в AGENTS.md:
  B=1 без паддингов, поэтому vec_lens=F и img_lens=T в графе не нужны явно);
  fuse -> BiLSTM -> head -> logits [1,T,C], blank=0, класс i>0 = vocab[i-1].

feats ожидаются УЖЕ z-scored (mean/sd фолда применяет воркер тем же кодом
applyStandardize из src/ink-mlp.js — так паддинг-семантика «нулевой вектор»
совпадает с обучением, и graceful-деградация при пустом векторе = 1 кадр нулей).

Проверки на реальном тест-фолде (перед копированием в клиент):
  • torch vs onnxruntime: max |Δlogits| (иначе экспорт рассинхронизирован);
  • CER оннкс-графа == CER torch-модели на test-разрезе фолда;
  • пустой векторный вход (1 кадр нулей) — конечные logits, декод не падает.

Выход: export/line_hybrid.onnx + line_hybrid_std.json (mean/sd фолда — воркер
обязан применить ровно их) + hybrid_manifest.json (происхождение весов и
метрики). В прод-клиент модель НЕ копируется: интеграция в веб-клиента —
отдельное осознанное решение (боевая модель — растровая line_int8).

Запуск: make export-onnx            (FOLD из .env, по умолчанию 0)
        .venv/bin/python export-onnx.py --fold 0   (venv — «make venv»)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hybridnet.hdset import load_pairs
from hybridnet.hmodel import HybridNet
from hybridnet.metrics import cer, greedy_ids

HERE = Path(__file__).resolve().parent
VOCAB_PATH = HERE / "jsnet/public/models/line_vocab.json"


class ServeHybrid(nn.Module):
    """B=1 версия HybridNet без циклов по батчу — экспорт-двойник hmodel.forward.

    Интерполяция кадров — ровно та же операция (linear, align_corners=False),
    что в обучении; при F=1 даёт константный вектор по всем T кадрам, что
    эквивалентно «нулевому векторному входу» при нулевых feats (деградация).
    """

    def __init__(self, net: HybridNet):
        super().__init__()
        self.net = net

    def forward(self, image: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        net = self.net
        f = net.cnn(image)                                    # [1,128,16,T]
        b, c, h, t = f.shape
        f = torch.relu(net.proj(f.permute(0, 3, 1, 2).reshape(b, t, c * h)))  # [1,T,256]
        v = net.vector_frames(feats)                          # [1,F,vh]
        v = F.interpolate(v.transpose(1, 2), size=t, mode="linear",
                          align_corners=False).transpose(1, 2)  # [1,T,vh]
        out, _ = net.lstm(net.fuse(torch.cat([f, v], dim=-1)))
        return net.head(out)                                  # [1,T,C]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("DATA_DIR", "data/line-vectors"))
    ap.add_argument("--work", default=os.environ.get("WORK_DIR", "work"))
    ap.add_argument("--fold", type=int, default=int(os.environ.get("FOLD", "0")),
                    help="фолд leave-one-writer-out, чьи веса публикуем")
    ap.add_argument("--out", default=str(HERE / "export/line_hybrid.onnx"))
    ap.add_argument("--opset", type=int, default=17,
                    help="17 — поддерживается и onnxruntime-web 1.31 (wasm+webgpu)")
    args = ap.parse_args()

    manifest = json.loads((Path(args.data) / "manifest.json").read_text(encoding="utf-8"))
    fr = next((f for f in manifest["folds"] if f["fold"] == args.fold), None)
    if fr is None:
        raise SystemExit(f"фолд {args.fold} не найден в manifest.json (есть: "
                         f"{[f['fold'] for f in manifest['folds']]})")
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    n_classes = len(vocab) + 1

    ckpt = Path(args.work) / "fusion" / f"fold{args.fold}" / "model-best.pt"
    net = HybridNet(n_classes)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    serve = ServeHybrid(net).eval()

    # ————— проверка на реальном test-разрезе фолда (feats уже z-scored из датасета) —————
    samples, _ = load_pairs(args.data, args.fold, "test", with_raster=True)
    refs = [s.word for s in samples]
    hyps_t, hyps_o = [], []
    max_diff = 0.0
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_grad_enabled(False)  # экспорт-прогон: градиенты не нужны и ломают .numpy()

    # экспорт с фиксированными примерами размеров (динамические оси ниже)
    s0 = samples[0]
    img0 = torch.from_numpy(s0.ink).unsqueeze(0).unsqueeze(0)   # [1,1,64,W]
    fe0 = torch.from_numpy(s0.feats).unsqueeze(0)               # [1,F,10]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        serve, (img0, fe0), str(out_path),
        input_names=["image", "feats"], output_names=["logits"],
        dynamic_axes={"image": {3: "width"}, "feats": {1: "frames"}, "logits": {1: "steps"}},
        opset_version=args.opset,
        dynamo=False,  # torch>=2.9 экспорт через dynamo ломается на LSTM с динамической длиной;
                       # легаси-TracedScript путь даёт ONNX LSTM op, его поднимает onnxruntime-web
    )
    import onnxruntime as ort
    so = ort.SessionOptions()
    sess = ort.InferenceSession(str(out_path), so, providers=["CPUExecutionProvider"])

    # сравнение логитов и greedy-декодов по каждому образцу (T у torch и ort совпадает)
    for s in samples:
        img = s.ink[np.newaxis, np.newaxis]                     # [1,1,64,W]
        feats = s.feats[np.newaxis]                             # [1,F,10]
        t_l = serve(torch.from_numpy(img), torch.from_numpy(feats))[0].numpy()
        o_l = sess.run(["logits"], {"image": img, "feats": feats})[0][0]
        max_diff = max(max_diff, float(np.abs(t_l - o_l).max()))
        hyps_t.append("".join(vocab[i - 1] for i in greedy_ids(t_l) if i > 0))
        hyps_o.append("".join(vocab[i - 1] for i in greedy_ids(o_l) if i > 0))
    cer_t, cer_o = cer(refs, hyps_t), cer(refs, hyps_o)

    # graceful деградация: пустой вектор = 1 кадр нулей (воркер шлёт именно так)
    deg = sess.run(["logits"], {"image": img0.numpy(), "feats": np.zeros((1, 1, 10), np.float32)})[0]
    if not np.isfinite(deg).all():
        raise SystemExit("пустой векторный вход дал неконечные logits — интеграция невозможна")

    # std фолда — воркер обязан применять ровно эти mean/sd (applyStandardize)
    std_path = out_path.with_name(out_path.stem + "_std.json")
    std_path.write_text(json.dumps({
        "format": "inknet-hybrid-std/1",
        "fold": args.fold,
        "mean": fr["mean"], "sd": fr["sd"],
        "note": "z-score кадров ink-features (STRIDE=10): client применяет applyStandardize(mean,sd) ПЕРЕД подачей в feats; пустой вектор = 1 кадр нулей ПОСЛЕ z-score",
    }, indent=2), encoding="utf-8")
    mani_path = out_path.with_name(out_path.stem + "_manifest.json")
    mani_path.write_text(json.dumps({
        "model": out_path.name,
        "format": "inknet-hybrid/1",
        "inputs": {"image": "[1,1,64,W] float32 ink 0..1",
                   "feats": "[1,F,10] float32 z-scored ink-features (STRIDE=10)",
                   "align": "линейная интерполяция F->T=W//4 внутри графа (align_corners=False)"},
        "output": "logits [1,T,C], blank=0, класс i>0 = line_vocab.json[i-1]",
        "opset": args.opset,
        "weights": str(ckpt),
        "fold": args.fold, "testWriter": fr["testWriter"],
        "trainWriters": fr["trainWriters"],
        "cerWriterDisjoint": cer_o,
        "maxLogitDiffTorchOnnx": max_diff,
    }, indent=2), encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"ONNX: {out_path} ({size_kb:.0f} кБ, opset {args.opset})")
    print(f"torch vs ort: max|Δlogits| = {max_diff:.2e}")
    print(f"CER writer-disjoint (фолд {args.fold}, тест {fr['testWriter']}): torch {cer_t:.3f} == ort {cer_o:.3f}")
    print(f"деградация при пустом feats: logits конечны ({deg.shape})")
    print(f"std -> {std_path.name}, манифест -> {mani_path.name}")
    if max_diff > 1e-4:
        raise SystemExit("РАСХОЖДЕНИЕ torch vs ONNX — экспорт непроверен, в прод не выпускать")
    if abs(cer_t - cer_o) > 1e-9 and "".join(hyps_t) != "".join(hyps_o):
        raise SystemExit("декод torch != декод ort — экспорт непроверен")


if __name__ == "__main__":
    main()
