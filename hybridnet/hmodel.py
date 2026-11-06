# HybridNet — модель варианта B: растровая CNN + векторная ветка штрихов,
# общий BiLSTM+CTC хед. Контракт клиента сохранён: blank=0, класс i>0 =
# line_vocab.json[i-1], logits [B,T,C] row-major.
#
# Выравнивание представлений: растр даёт T_r = W//4 кадров (два maxpool),
# векторная ветка — T_v кадров ink-features. Числа не совпадают (разная
# параметризация одной строки), поэтому векторные признаки перед fuse
# ПЕРЕДИНТЕРПОЛИРУЮТСЯ к T_r по времени (линейно, per-sample: кадры батча
# паддингов в интерполяцию не попадают). Это же выравнирование повторит
# клиентский ONNX-граф при экспорте.
#
# Честность ablation: raster-only веса = эталонная LineNet из vendor (тот же
# класс, те же параметры), vector-only = VecLineNet из vmodel (архитектура
# inknet-online). Fusion добавляет к растровому стеку только векторную ветку
# (~0.35M) и расширяет fuse-проецию.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybridnet.vendor.linenet import LineNet
from hybridnet.vmodel import CausalBlock

__all__ = ["LineNet", "HybridNet"]


class HybridNet(nn.Module):
    def __init__(self, n_classes: int, d_in: int = 10, vh: int = 96,
                 dilations=(1, 2, 4, 8, 8, 4, 2, 1), dropout: float = 0.2,
                 vec_drop: float = 0.0):
        super().__init__()
        # vec_drop — доля примеров обучения, у которых векторный поток целиком
        # обнуляется (modality dropout). Без него векторные признаки (скорость,
        # направление, ритм почерка) работают как «отпечаток автора»: модель
        # заучивает 4 авторов обучения и на новом человеке гибрид резко хуже
        # чистого растра. С ним часть шагов CTC проходит без вектора, и растровый
        # путь обязан оставаться сильным. На inference (и в ONNX-графе) dropout
        # не действует — контракт клиента не меняется.
        self.vec_drop = float(vec_drop)
        # растровая часть — точь-в-точь conv+proj у LineNet (паритет raster-only)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.proj = nn.Linear(128 * 16, 256)
        # векторная ветка: та же проекция/TCN, что у VecLineNet, без хвоста
        self.vproj = nn.Sequential(nn.Linear(d_in, vh), nn.GELU())
        self.vtcn = nn.Sequential(*(CausalBlock(vh, d, dropout) for d in dilations))
        # fuse: склейка [raster 256, vector vh] -> 256
        self.fuse = nn.Sequential(nn.Linear(256 + vh, 256), nn.ReLU())
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.head = nn.Linear(2 * 256, n_classes)

    def vector_frames(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B,F,D] -> [B,F,vh] (последовательность без изменения длины)."""
        h = self.vproj(feats)
        return self.vtcn(h.transpose(1, 2)).transpose(1, 2)

    def init_as_raster(self) -> int:
        """Старт гибрида = ровно растровая модель. В fuse растровые столбцы весов —
        единичная матрица, векторные — нули, bias — ноль. Тогда
        relu(fuse(cat(raster, vector))) == relu(raster), то же самое, что у LineNet
        перед BiLSTM, а вектор постепенно добавляется к представлению (столбцы
        fuse для него становятся ненулевыми с первого же шага, градиенты на ветку
        идут через активации вектора). Смысл: общий хед не получает шум от
        необученной векторной ветки — именно из-за этого гибрид на LR=2e-3
        проваливался в CTC-коллапс (выход — одна пустая гипотеза). Архитектура и
        ONNX-граф не меняются, это только init."""
        rd = self.proj.out_features           # растровая часть склейки = 256
        lin = self.fuse[0]
        with torch.no_grad():
            lin.weight[:, :rd].copy_(torch.eye(rd, dtype=lin.weight.dtype, device=lin.weight.device))
            lin.weight[:, rd:].zero_()
            lin.bias.zero_()
        return lin.weight[:, rd:].numel()

    def forward(self, image: torch.Tensor, feats: torch.Tensor,
                vec_lens: torch.Tensor, img_lens: torch.Tensor) -> torch.Tensor:
        """image [B,1,64,W], feats [B,F,D] одной и той же строки.

        Возвращает logits [B,T,C], T = W//4 (как у боевой LineNet).
        """
        f = self.cnn(image)                                 # [B,128,16,T]
        b, c, h, t = f.shape
        f = torch.relu(self.proj(f.permute(0, 3, 1, 2).reshape(b, t, c * h)))  # [B,T,256]
        v = self.vector_frames(feats)                       # [B,F,vh]
        if self.training and self.vec_drop > 0:
            # целиком обнуляем векторный поток у части примеров (см. vec_drop в __init__)
            keep = (torch.rand(b, 1, 1, device=v.device) >= self.vec_drop).to(v.dtype)
            v = v * keep
        # выравнивание по времени: интерполяция только реальных кадров каждого
        # примера (пересечение с img_lens — CTC-длины растровой ветки)
        aligned = []
        for i in range(b):
            tl = max(1, int(img_lens[i]))
            vi = v[i, : max(1, int(vec_lens[i]))].transpose(0, 1).unsqueeze(0)  # [1,vh,F_i]
            vi = F.interpolate(vi, size=tl, mode="linear", align_corners=False)
            aligned.append(vi[0].transpose(0, 1))           # [T_i,vh]
        # padding-хвосты растровых кадров (после img_lens) остаются с нулевым вектором
        vpad = torch.zeros(b, t, aligned[0].shape[1], dtype=v.dtype, device=v.device)
        for i, a in enumerate(aligned):
            vpad[i, : a.shape[0]] = a
        out, _ = self.lstm(self.fuse(torch.cat([f, vpad], dim=-1)))
        return self.head(out)

    def log_probs(self, image: torch.Tensor, feats: torch.Tensor,
                  vec_lens: torch.Tensor, img_lens: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.forward(image, feats, vec_lens, img_lens), dim=-1).transpose(0, 1)
