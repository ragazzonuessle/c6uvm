from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from c6_uvm.record.types import FloatArray
from c6_uvm.statute.config import ModelConfig, TrainConfig


class SubtypeHead(nn.Module):
    def __init__(self, rna_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rna_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x).squeeze(-1)
        return out


class SubtypeAgent:
    def __init__(self, cfg: ModelConfig) -> None:
        self._cfg = cfg
        self._head = SubtypeHead(cfg.rna_dim, cfg.subtype_hidden)
        self._fitted = False

    @property
    def head(self) -> SubtypeHead:
        return self._head

    def fit(
        self,
        rna: FloatArray,
        gep_class: NDArray[np.int64],
        train_cfg: TrainConfig,
    ) -> list[float]:
        x = torch.as_tensor(rna, dtype=torch.float32)
        y = torch.as_tensor((gep_class == 2).astype(np.float32), dtype=torch.float32)
        opt = torch.optim.Adam(
            self._head.parameters(), lr=train_cfg.subtype_lr, weight_decay=train_cfg.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        history: list[float] = []
        self._head.train()
        for _ in range(train_cfg.subtype_steps):
            opt.zero_grad()
            logits = self._head(x)
            loss = loss_fn(logits, y)
            torch.autograd.backward(loss)
            opt.step()
            history.append(float(loss.detach()))
        self._fitted = True
        return history

    def predict_proba(self, rna: FloatArray) -> FloatArray:
        self._head.eval()
        with torch.no_grad():
            x = torch.as_tensor(rna, dtype=torch.float32)
            probs = torch.sigmoid(self._head(x))
        return probs.numpy().astype(np.float64)

    def prob_class2(self, rna: FloatArray) -> float:
        single = rna.reshape(1, -1)
        return float(self.predict_proba(single)[0])

    def emit_class(self, rna: FloatArray) -> int:
        return 2 if self.prob_class2(rna) >= 0.5 else 1
