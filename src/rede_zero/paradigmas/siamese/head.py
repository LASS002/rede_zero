"""Head de correlação e deslocamento para o Siamese."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class CabecaSiamese(nn.Module):
    """Produz score map por correlação depthwise e offset normalizado."""

    def __init__(self, canais: int) -> None:
        super().__init__()
        self.offset = nn.Sequential(
            nn.Linear(canais * 2, canais),
            nn.ReLU(inplace=True),
            nn.Linear(canais, 2),
            nn.Tanh(),
        )

    @staticmethod
    def _correlacao_depthwise(template: Tensor, busca: Tensor) -> Tensor:
        lote, canais, altura, largura = busca.shape
        if template.shape[:2] != (lote, canais):
            raise ValueError("Template e busca devem ter o mesmo lote e canais.")
        if template.shape[2] > altura or template.shape[3] > largura:
            raise ValueError("O mapa do template deve caber no mapa de busca.")

        busca_agrupada = busca.reshape(1, lote * canais, altura, largura)
        kernels = template.reshape(lote * canais, 1, *template.shape[2:])
        correlacao = functional.conv2d(busca_agrupada, kernels, groups=lote * canais)
        return correlacao.reshape(lote, canais, *correlacao.shape[2:]).sum(dim=1, keepdim=True)

    def forward(self, template: Tensor, busca: Tensor) -> dict[str, Tensor]:
        score_map = self._correlacao_depthwise(template, busca) / math.sqrt(template.shape[1])
        descritores = torch.cat(
            (template.mean(dim=(2, 3)), busca.mean(dim=(2, 3))), dim=1
        )
        return {"score_map": score_map, "offset": self.offset(descritores)}
