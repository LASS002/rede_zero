"""Contrato comum e registry dos paradigmas de rastreamento."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, NoReturn

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
        """Executa o forward de um lote preparado para treinamento."""

    @abstractmethod
    def prever_roi(
        self, patch: np.ndarray | Tensor, contexto: Mapping[str, Any]
    ) -> BBox:
        """Prevê a próxima ROI de um único objeto independente."""

    @abstractmethod
    def perda_fn(
        self, saidas: Mapping[str, Tensor], alvos: Mapping[str, Tensor]
    ) -> Tensor:
        """Calcula a perda para as saídas e alvos do lote."""


class _ParadigmaContrato(ParadigmaBase):
    """Implementação temporária até a rede específica ser adicionada."""

    def __init__(self, nome: str) -> None:
        super().__init__()
        self.nome = nome
        self.versao = "0.0.0"

    def _pendente(self, componente: str) -> NoReturn:
        raise NotImplementedError(
            f"O {componente} do paradigma '{self.nome}' ainda não foi implementado."
        )

    def forward(self, treino_batch: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        self._pendente("modelo")

    def prever_roi(
        self, patch: np.ndarray | Tensor, contexto: Mapping[str, Any]
    ) -> BBox:
        self._pendente("modelo")

    def perda_fn(
        self, saidas: Mapping[str, Tensor], alvos: Mapping[str, Tensor]
    ) -> Tensor:
        self._pendente("perda")


FabricaParadigma = Callable[..., ParadigmaBase]
_REGISTRY: dict[str, FabricaParadigma] = {}


def registrar_paradigma(nome: str, fabrica: FabricaParadigma) -> None:
    nome_normalizado = nome.strip().lower()
    if not nome_normalizado:
        raise ValueError("O nome do paradigma não pode ser vazio.")
    if nome_normalizado in _REGISTRY:
        raise ValueError(f"O paradigma '{nome_normalizado}' já está registrado.")
    _REGISTRY[nome_normalizado] = fabrica


def get_paradigma(nome: str, **kwargs: Any) -> ParadigmaBase:
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
    return tuple(_REGISTRY)


def _detector_pendente(**_: Any) -> ParadigmaBase:
    raise NotImplementedError(
        "O paradigma 'detector' está pendente; sua implementação é a SCR-18."
    )


def _fabrica_siamese(**kwargs: Any) -> ParadigmaBase:
    from rede_zero.paradigmas.siamese import ModeloSiamese

    return ModeloSiamese(**kwargs)


registrar_paradigma("siamese", _fabrica_siamese)
registrar_paradigma("regressao", lambda **_: _ParadigmaContrato("regressao"))
registrar_paradigma("detector", _detector_pendente)
