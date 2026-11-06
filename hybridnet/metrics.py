# Жадный CTC-декод и метрики (совместимы с blank=0 контрактом клиента).
from __future__ import annotations

import numpy as np


def greedy_ids(logits_row: np.ndarray) -> list[int]:
    """argmax по классам -> collapse повторов -> убрать blank (как greedy_decode в ctc-decode.js)."""
    out: list[int] = []
    prev = 0
    for c in np.argmax(logits_row, axis=-1).tolist():
        if c != 0 and c != prev:
            out.append(c)
        prev = c
    return out


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    dist = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    chars = max(1, sum(len(r) for r in refs))
    return dist / chars
