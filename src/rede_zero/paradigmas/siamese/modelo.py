"""Modelo completo do paradigma Siamese."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from rede_zero.data.readers.base import BBox
from rede_zero.paradigmas.base import ParadigmaBase
from rede_zero.paradigmas.siamese.backbone import BackboneSiamese
from rede_zero.paradigmas.siamese.head import CabecaSiamese


class ModeloSiamese(ParadigmaBase):
    """Rastreador Siamese com backbone compartilhado e correlação depthwise."""

    nome = "siamese"
    versao = "1.0.0"

    def __init__(self, canais_backbone: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.backbone = BackboneSiamese(canais_backbone)
        self.head = CabecaSiamese(self.backbone.canais_saida)

    def forward(self, treino_batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        try:
            template = treino_batch["template"]
            busca = treino_batch["busca"]
        except KeyError as erro:
            raise KeyError("O lote Siamese requer 'template' e 'busca'.") from erro
        return self.head(self.backbone(template), self.backbone(busca))

    def perda_fn(
        self, saidas: Mapping[str, Tensor], alvos: Mapping[str, Tensor]
    ) -> Tensor:
        raise NotImplementedError("A perda Siamese será implementada na SCR-10.")

    @staticmethod
    def _para_lote(imagem: np.ndarray | Tensor, device: torch.device) -> Tensor:
        tensor = torch.as_tensor(imagem, device=device)
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1) if tensor.shape[-1] == 3 else tensor
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("A imagem deve ter formato RGB (H, W, 3) ou (B, 3, H, W).")
        tensor = tensor.float()
        return tensor / 255.0 if tensor.max() > 1 else tensor

    @torch.no_grad()
    def prever_roi(
        self, patch: np.ndarray | Tensor, contexto: Mapping[str, Any]
    ) -> BBox:
        roi = contexto.get("roi")
        template = contexto.get("template")
        if not isinstance(roi, BBox) or template is None:
            raise ValueError("O contexto Siamese requer 'roi' (BBox) e 'template'.")

        device = next(self.parameters()).device
        saidas = self.forward(
            {"template": self._para_lote(template, device), "busca": self._para_lote(patch, device)}
        )
        score_map = saidas["score_map"].flatten(1).softmax(dim=1).reshape_as(saidas["score_map"])
        _, _, altura, largura = score_map.shape
        grade_y, grade_x = torch.meshgrid(
            torch.arange(altura, device=device), torch.arange(largura, device=device), indexing="ij"
        )
        x = (score_map[0, 0] * grade_x).sum().item()
        y = (score_map[0, 0] * grade_y).sum().item()
        largura_busca, altura_busca = contexto.get("tamanho_busca", (roi.w * 4, roi.h * 4))
        offset_x, offset_y = saidas["offset"][0].tolist()
        dx = ((x - (largura - 1) / 2) / max((largura - 1) / 2, 1) + offset_x) * largura_busca / 2
        dy = ((y - (altura - 1) / 2) / max((altura - 1) / 2, 1) + offset_y) * altura_busca / 2
        return BBox(roi.x + dx, roi.y + dy, roi.w, roi.h, roi.id, roi.label, roi.conf)
