# Парный датасет inknet-line-vectors/2+/3: одна и та же синтезированная строка
# в ДВУХ представлениях — кадры ink-features (feats) и растр штрихов (ink),
# растеризованный vendor-портом клиента. Ablation-режимы растер/вектор/гибрид
# обучаются на ровно этих же строках и фолдах.
#
# Растеризация выполняется ОДИН раз при загрузке и кэшируется в памяти: строки
# детерминированы (перо PEN_DEFAULTS, фиксированные штрихи из JSON), поэтому
# кэш эквивалентен офлайн-препроцессингу и не вносит стохастичность в сравнение.
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hybridnet.vendor.raster import Pen, rasterize_strokes, strip_to_tensor

PEN = Pen()  # PEN_DEFAULTS: R=1, hardness=100 — как в растровом baseline inknet-online


@dataclass
class PairSample:
    classes: list[int]
    word: str
    writer: str
    feats: np.ndarray                 # [nFrames, 10] z-scored
    ink: np.ndarray | None = None     # float32 [64, dw] или None (vector-режим)


def load_pairs(data_dir: str | Path, fold: int, split: str, with_raster: bool) -> tuple[list[PairSample], dict]:
    path = Path(data_dir) / f"fold{fold}.{split}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") not in ("inknet-line-vectors/2", "inknet-line-vectors/3"):
        raise ValueError(f"{path}: нужен формат inknet-line-vectors/2+/3 (со strokes)")
    stride = doc["stride"]
    samples: list[PairSample] = []
    skipped = 0
    for s in doc["samples"]:
        n_fr = s["nFrames"]
        if n_fr < 2 * len(s["classes"]) - 1:  # CTC: кадров вектора должно хватать на метку
            skipped += 1
            continue
        ink = None
        if with_raster:
            strokes = [[{"x": p[0], "y": p[1], "t": p[2]} for p in st] for st in s["strokes"]]
            ras = rasterize_strokes(strokes, PEN)
            t = None
            if ras is not None:
                cov, _, _ = ras
                t = strip_to_tensor(cov, PEN)
            if t is None or t[1] // 4 < 2 * len(s["classes"]) - 1:
                skipped += 1
                continue
            ink = t[0]
        feats = np.asarray(s["feats"], dtype=np.float32).reshape(n_fr, stride)
        samples.append(PairSample(classes=list(s["classes"]), word=s["word"],
                                  writer=s["writer"], feats=feats, ink=ink))
    if skipped:
        print(f"  pairs {split} fold{fold}: пропущено (CTC-длина/пустой растр): {skipped}")
    return samples, doc


def collate(batch: list[PairSample]) -> dict:
    """Пакет: image right-pad 0, feats right-pad 0; длины — отдельные тензоры.
    lens для CTCLoss: img_lens = dw//4 (растр/fusion), vec_lens = nFrames (vector)."""
    images = [torch.from_numpy(s.ink) for s in batch if s.ink is not None]
    feats = [torch.from_numpy(s.feats) for s in batch]
    img_lens = torch.tensor([(s.ink.shape[1] // 4 if s.ink is not None else s.feats.shape[0]) for s in batch], dtype=torch.long)
    vec_lens = torch.tensor([s.feats.shape[0] for s in batch], dtype=torch.long)
    max_w = max((s.ink.shape[1] for s in batch if s.ink is not None), default=0)
    padded_img = torch.zeros(len(batch), 1, 64, max_w, dtype=torch.float32) if max_w else None
    for i, x in enumerate(images):
        padded_img[i, 0, :, : x.shape[1]] = x
    max_f = int(vec_lens.max())
    padded_feats = torch.zeros(len(batch), max_f, feats[0].shape[1], dtype=torch.float32)
    for i, x in enumerate(feats):
        padded_feats[i, : x.shape[0]] = x
    targets = torch.tensor([c for s in batch for c in s.classes], dtype=torch.long)
    return {"image": padded_img, "img_lens": img_lens, "feats": padded_feats,
            "vec_lens": vec_lens, "targets": targets,
            "classes_lens": torch.tensor([len(s.classes) for s in batch], dtype=torch.long)}


class PairDataset(Dataset):
    def __init__(self, samples: list[PairSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PairSample:
        return self.samples[idx]
