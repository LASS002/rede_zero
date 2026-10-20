"""Paradigma C — Detector + associação (PENDENTE, implementação é a SCR-18).

Contrato resumido (detalhes em ``docs/paradigmas.md``):

- Por frame, o modelo detecta todas as bboxes (classe única "object") e as associa
  aos trackers existentes por IoU/greedy matching — sem NMS conjunto entre objetos,
  pois as ROIs são independentes e não se sobrepõem (até 8 objetos).
- ``forward(treino_batch)``: devolve ``bboxes`` e ``scores`` normalizados para o batch.
- ``prever_roi(patch, contexto) -> BBox``: herda do contrato comum; no detector o
  "patch" é o frame completo e o contexto carrega os IDs dos trackers ativos.
- ``perda_fn(saidas, alvos)``: classificação + regressão de bbox (definir na SCR-18).
- Registro no registry: nome ``detector`` — hoje o factory lança ``NotImplementedError``
  e a classe ``Detector`` abaixo é o ponto de partida para a SCR-18.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn

import numpy as np
from torch import Tensor

from rede_zero.data.readers.base import BBox
from rede_zero.paradigmas.base import ParadigmaBase

MENSAGEM_PENDENTE = "O paradigma 'detector' está pendente; sua implementação é a SCR-18."


class Detector(ParadigmaBase):
    """Placeholder do detector+associação — implementar na SCR-18."""

    def __init__(self, **_: Any) -> None:
        super().__init__()
        self.nome = "detector"
        self.versao = "0.0.0"

    def _pendente(self, componente: str) -> NoReturn:
        raise NotImplementedError(
            f"O {componente} do paradigma 'detector' está pendente; "
            "sua implementação é a SCR-18."
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


def fabrica_detector(**_: Any) -> ParadigmaBase:
    """Factory registrado no registry — lança enquanto a SCR-18 não entregar o modelo."""
    raise NotImplementedError(MENSAGEM_PENDENTE)
