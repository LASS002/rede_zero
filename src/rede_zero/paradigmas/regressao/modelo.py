"""Modelo de regressão de deslocamento (Paradigma B - SCR-07)."""

from typing import Any, Mapping
import numpy as np
import torch
from torch import Tensor, nn

from rede_zero.data.readers.base import BBox
from rede_zero.paradigmas.base import ParadigmaBase


class ModeloRegressao(ParadigmaBase):
    """Paradigma de regressão de deslocamento baseado em patch 128x128 expandido 2x."""

    nome = "regressao"
    versao = "0.1.0"

    def __init__(self, in_channels: int = 3, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        # Backbone com 3 blocos conv [32, 64, 128]
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32 -> 16
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        feat_dim = 128 * 4 * 4  # 2048

        # MLP head -> (dx, dy, sw, sh) + confianca
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, treino_batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        x = treino_batch["patch"]
        features = self.backbone(x)
        pooled = self.pool(features)
        flat = torch.flatten(pooled, 1)
        saida = self.head(flat)  # (B, 5)

        delta = saida[:, :4]
        conf_logits = saida[:, 4:5]
        confianca = torch.sigmoid(conf_logits)

        return {
            "predicoes": saida,
            "delta": delta,
            "conf_logits": conf_logits,
            "confianca": confianca,
        }

    def perda_fn(self, saidas: Mapping[str, Tensor], alvos: Mapping[str, Tensor]) -> Tensor:
        from rede_zero.treino.losses import perda_regressao

        return perda_regressao(saidas, alvos)

    def prever_roi(self, patch: np.ndarray | Tensor, contexto: Mapping[str, Any]) -> BBox:
        self.eval()
        if isinstance(patch, np.ndarray):
            t_patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            t_patch = patch
            if t_patch.dim() == 3:
                t_patch = t_patch.unsqueeze(0)

        device = next(self.parameters()).device
        t_patch = t_patch.to(device)

        with torch.no_grad():
            saidas = self.forward({"patch": t_patch})

        delta = saidas["delta"][0].cpu().numpy()
        conf = float(saidas["confianca"][0].item())
        dx, dy, sw, sh = float(delta[0]), float(delta[1]), float(delta[2]), float(delta[3])

        roi_ant = contexto.get("roi_anterior") or contexto.get("roi")
        if roi_ant is None:
            return BBox(x=0.0, y=0.0, w=0.0, h=0.0, conf=conf)

        patch_w = roi_ant.w * 2.0
        patch_h = roi_ant.h * 2.0
        novo_cx = roi_ant.center[0] + dx * patch_w
        novo_cy = roi_ant.center[1] + dy * patch_h
        novo_w = roi_ant.w * float(np.exp(np.clip(sw, -2.0, 2.0)))
        novo_h = roi_ant.h * float(np.exp(np.clip(sh, -2.0, 2.0)))
        novo_x = novo_cx - novo_w / 2.0
        novo_y = novo_cy - novo_h / 2.0

        return BBox(
            x=novo_x,
            y=novo_y,
            w=max(1.0, novo_w),
            h=max(1.0, novo_h),
            id=roi_ant.id,
            label=roi_ant.label,
            conf=conf,
        )
