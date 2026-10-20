"""Funções de perda dos paradigmas de rastreamento."""

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as functional


def alvo_gaussiano(
    deslocamento: Tensor, tamanho: tuple[int, int], sigma: float = 1.0
) -> Tensor:
    """Gera rótulos Gaussianos a partir de offsets normalizados em [-0.5, 0.5]."""
    if deslocamento.ndim != 2 or deslocamento.shape[1] != 2:
        raise ValueError("O deslocamento deve ter formato (B, 2).")
    altura, largura = tamanho
    if altura < 1 or largura < 1 or sigma <= 0:
        raise ValueError("Tamanho e sigma devem ser positivos.")

    grade_y, grade_x = torch.meshgrid(
        torch.arange(altura, device=deslocamento.device, dtype=deslocamento.dtype),
        torch.arange(largura, device=deslocamento.device, dtype=deslocamento.dtype),
        indexing="ij",
    )
    centro_x = (deslocamento[:, 0].clamp(-0.5, 0.5) + 0.5) * (largura - 1)
    centro_y = (deslocamento[:, 1].clamp(-0.5, 0.5) + 0.5) * (altura - 1)
    distancia_quadrada = (grade_x - centro_x[:, None, None]) ** 2 + (
        grade_y - centro_y[:, None, None]
    ) ** 2
    return torch.exp(-distancia_quadrada / (2 * sigma**2)).unsqueeze(1)


def perda_siamese(
    saidas: Mapping[str, Tensor], alvos: Mapping[str, Tensor], peso_offset: float = 1.0
) -> Tensor:
    """Combina BCE no score map Gaussiano e L1 no deslocamento previsto."""
    score_map = saidas["score_map"]
    offset = saidas["offset"]
    alvo_offset = alvos["alvo_siamese"].to(device=offset.device, dtype=offset.dtype)
    alvo_score = alvo_gaussiano(alvo_offset, score_map.shape[-2:]).to(dtype=score_map.dtype)
    perda_score = functional.binary_cross_entropy_with_logits(score_map, alvo_score)
    perda_offset = functional.l1_loss(offset, alvo_offset)
    return perda_score + peso_offset * perda_offset


def perda_regressao(saidas: Mapping[str, Tensor], alvos: Mapping[str, Tensor]) -> Tensor:
    """Combina smooth-L1 dos deltas e BCE da confiança logística."""
    delta = alvos["alvo_regressao"].to(
        device=saidas["delta"].device, dtype=saidas["delta"].dtype
    )
    confianca = alvos["confianca"].to(
        device=saidas["conf_logits"].device, dtype=saidas["conf_logits"].dtype
    )
    return functional.smooth_l1_loss(saidas["delta"], delta) + functional.binary_cross_entropy_with_logits(
        saidas["conf_logits"], confianca
    )
