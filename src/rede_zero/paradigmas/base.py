"""Contrato comum e registry dos paradigmas de rastreamento."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from rede_zero.data.readers.base import BBox


class ParadigmaBase(nn.Module, ABC):
    """Interface que toda rede de rastreamento deve implementar."""

    nome: str
    versao: str

    @abstractmethod
    def forward(self, treino_batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Executa o forward de um lote já preparado para treinamento."""

    @abstractmethod
    def prever_roi(
        self, patch: np.ndarray | Tensor, contexto: Mapping[str, Any]
    ) -> BBox:
        """Prevê a próxima ROI para um único objeto independente."""

    @abstractmethod
    def perda_fn(
        self, saidas: Mapping[str, Tensor], alvos: Mapping[str, Tensor]
    ) -> Tensor:
        """Calcula a perda do paradigma para saídas e alvos do lote."""


class _ParadigmaContrato(ParadigmaBase):
    """Implementação temporária até a rede específica ser adicionada."""

    def __init__(self, nome: str) -> None:
        super().__init__()
        self.nome = nome
        self.versao = "0.0.0"

    def forward(self, treino_batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        raise NotImplementedError(
            f"O modelo do paradigma '{self.nome}' ainda não foi implementado."
        )

    def prever_roi(
        self, patch: np.ndarray | Tensor, contexto: Mapping[str, Any]
    ) -> BBox:
        raise NotImplementedError(
            f"O modelo do paradigma '{self.nome}' ainda não foi implementado."
        )

    def perda_fn(
        self, saidas: Mapping[str, Tensor], alvos: Mapping[str, Tensor]
    ) -> Tensor:
        raise NotImplementedError(
            f"A perda do paradigma '{self.nome}' ainda não foi implementada."
        )


FabricaParadigma = Callable[..., ParadigmaBase]
_REGISTRY: dict[str, FabricaParadigma] = {}


def registrar_paradigma(nome: str, fabrica: FabricaParadigma) -> None:
    """Registra uma fábrica sob um nome único normalizado."""
    nome_normalizado = nome.strip().lower()
    if not nome_normalizado:
        raise ValueError("O nome do paradigma não pode ser vazio.")
    if nome_normalizado in _REGISTRY:
        raise ValueError(f"O paradigma '{nome_normalizado}' já está registrado.")
    _REGISTRY[nome_normalizado] = fabrica


def get_paradigma(nome: str, **kwargs: Any) -> ParadigmaBase:
    """Instancia um paradigma registrado pelo seu nome."""
    nome_normalizado = nome.strip().lower()
    try:
        fabrica = _REGISTRY[nome_normalizado]
    except KeyError as erro:
        opcoes = ", ".join(listar_paradigmas())
        raise ValueError(
            f"Paradigma desconhecido: '{nome}'. Opções: {opcoes}."
        ) from erro
    return fabrica(**kwargs)


def listar_paradigmas() -> tuple[str, ...]:
    """Retorna os nomes de todos os paradigmas registrados, em ordem estável."""
    return tuple(_REGISTRY)


def _detector_pendente(**_: Any) -> ParadigmaBase:
    raise NotImplementedError(
        "O paradigma 'detector' está pendente; sua implementação é a SCR-18."
    )


registrar_paradigma("siamese", lambda **_: _ParadigmaContrato("siamese"))
registrar_paradigma("regressao", lambda **_: _ParadigmaContrato("regressao"))
registrar_paradigma("detector", _detector_pendente)
