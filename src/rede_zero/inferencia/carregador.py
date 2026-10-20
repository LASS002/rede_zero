"""Carregador de modelos para inferência (SCR-16)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from rede_zero.paradigmas.base import ParadigmaBase, get_paradigma, listar_paradigmas


def carregar_modelo_para_inferencia(
    paradigma_nome: str,
    ckpt_path: str | Path,
    device: torch.device | str | None = None,
    **fabrica_kwargs: Any,
) -> ParadigmaBase:
    """Instancia o paradigma via factory e carrega pesos do checkpoint.

    Args:
        paradigma_nome: Nome do paradigma ("siamese", "regressao", "detector").
        ckpt_path: Caminho para o arquivo .pt (checkpoint salvo pelo Treinador).
        device: Dispositivo alvo (cuda/cpu). Default: cuda se disponível.
        **fabrica_kwargs: Argumentos extras passados à fábrica do paradigma.

    Returns:
        Modelo em modo eval() no dispositivo solicitado.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo = get_paradigma(paradigma_nome, **fabrica_kwargs)
    modelo.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict")
    if state is None:
        # Tentar carregar state_dict direto (formato antigo)
        state = ckpt

    modelo.load_state_dict(state, strict=False)
    modelo.eval()
    return modelo


def obter_dispositivo_preferido() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validar_paradigma_para_inferencia(nome: str) -> bool:
    """Verifica se o paradigma está disponível para inferência (não pendente)."""
    try:
        get_paradigma(nome)
        return True
    except NotImplementedError:
        return False


def listar_paradigmas_disponiveis() -> dict[str, str]:
    """Retorna dicionário {nome: status} para exibição ao usuário."""
    resultado = {}
    for nome in listar_paradigmas():
        status = "disponível" if validar_paradigma_para_inferencia(nome) else "pendente (SCR-18)"
        resultado[nome] = status
    return resultado