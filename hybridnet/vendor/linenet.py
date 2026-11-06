# Модель: CRNN-типа CNN + 2x BiLSTM + head, CTC. Интерфейсы подогнаны под
# веб-клиента: вход `image` [B,1,64,W] (инк 0..1), выход `logits`
# [B,T,C] row-major (кадры по времени; layout logits[t*C+c] из ctc-decode.js),
# класс 0 = blank, класс i>0 = line_vocab.json[i-1].
#
# T = W // 4 (два maxpool stride 2) — жадный и beam-декодер клиента T не
# используют вовсе, только C; но W >= 16 обязательно (фильтр полос в воркере).
# Размер специально ~3.3M параметров: fp32 ~13 МБ (эталон line.onnx 14.3 МБ),
# int8 ~3 МБ (эталон line_int8.onnx 3.6 МБ), чтобы web-доставка не откатилась.
from __future__ import annotations

import torch
import torch.nn as nn


class LineNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                       # 64 x W -> 32 x W/2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                       # -> 16 x W/4
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        feat = 128 * 16
        self.proj = nn.Linear(feat, 256)
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        # BiLSTM: выход 2*hidden = 512
        self.head = nn.Linear(2 * 256, n_classes)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image [B,1,64,W] -> logits [B,T,C], T=W//4."""
        f = self.cnn(image)                         # [B,128,16,T]
        b, c, h, t = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, t, c * h)
        f = torch.relu(self.proj(f))
        out, _ = self.lstm(f)
        return self.head(out)                       # [B,T,C]

    def log_probs(self, image: torch.Tensor) -> torch.Tensor:
        """[T,B,C] — log_softmax по классам, что нужно nn.CTCLoss."""
        return torch.log_softmax(self.forward(image), dim=-1).transpose(0, 1)
