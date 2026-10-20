"""Testes do pipeline de inferência (SCR-17)."""

import numpy as np

from rede_zero.data.readers import BBox
from rede_zero.inferencia.pipeline import executar_pipeline


class LeitorFalso:
    """Leitor mínimo de dois frames RGB para o pipeline."""

    def __init__(self) -> None:
        self.frames = [
            np.zeros((40, 40, 3), dtype=np.uint8),
            np.full((40, 40, 3), 255, dtype=np.uint8),
        ]

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def get_frame(self, indice: int) -> np.ndarray:
        return self.frames[indice]


class ModeloFalso:
    """Modelo determinístico que sempre devolve a mesma ROI válida."""

    nome = "regressao"

    def __init__(self) -> None:
        self.patches: list[np.ndarray] = []

    def prever_roi(self, patch: np.ndarray, contexto: dict[str, object]) -> BBox:
        self.patches.append(patch)
        return BBox(12, 14, 8, 6, id=3, label="alvo", conf=0.9)


def test_pipeline_atualiza_roi_final_em_dois_frames():
    modelo = ModeloFalso()
    resultado = executar_pipeline(
        LeitorFalso(), [BBox(2, 4, 8, 6, id=3, label="alvo")], modelo
    )

    assert len(resultado.rois_por_frame) == 2
    assert resultado.rois_por_frame[0][0].to_tlwh() == (2, 4, 8, 6)
    assert resultado.rois_por_frame[-1][0].to_tlwh() == (12, 14, 8, 6)
    assert resultado.objetos_perdidos == []
    assert len(modelo.patches) == 1
    assert modelo.patches[0].shape == (128, 128, 3)
