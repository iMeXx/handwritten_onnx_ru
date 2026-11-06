# VecLineNet: векторная строчная CTC-модель (M1 baseline inknet-online).
#
# Интерфейс совпадает с контрактом веб-клиента (ctc-decode.js):
# класс 0 = CTC blank, класс i>0 = line_vocab.json[i-1]; выход — логиты
# [B, T, C] row-major. Вход вместо растра — последовательность z-scored
# кадров ink-features (stride=10 из jsnet/src/ink-features.js, vendor-копия).
#
# Архитектура: проекция 10->H, каузальный TCN (разреженные свёртки, чтобы
# модель можно было гонять инкрементально в live-режиме), 2x BiLSTM, head.
from __future__ import annotations

import torch
import torch.nn as nn


def _causal_pad(length: int, dilation: int) -> int:
    return dilation * (length - 1)


class CausalBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, kernel_size=3, padding=_causal_pad(3, dilation), dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,C,T]
        return self.drop(self.act(self.conv(x)[..., : x.shape[-1]]))  # обрезать правый causal-pad


class VecLineNet(nn.Module):
    def __init__(self, d_in: int = 10, n_classes: int = 72, hid: int = 96,
                 lstm_hidden: int = 96, dilations=(1, 2, 4, 8, 8, 4, 2, 1), dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, hid), nn.GELU())
        self.tcn = nn.Sequential(*(CausalBlock(hid, d, dropout) for d in dilations))
        self.lstm = nn.LSTM(hid, lstm_hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,F,D] -> logits [B,F,C] (число кадров не меняется)."""
        h = self.proj(x)
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.head(h)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """[F, B, C] — log_softmax по классам, вход nn.CTCLoss."""
        return torch.log_softmax(self.forward(x), dim=-1).transpose(0, 1)
