"""Testes da seleção interativa de ROIs (SCR-15) — GUI simulada."""

import numpy as np
import pytest

from rede_zero.data.readers.base import BBox
from rede_zero.inferencia.selecao_roi import carregar_primeiro_frame, selecionar_rois


def _frame_fake(*_a, **_k):
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def test_selecionar_rois_ate_oito(monkeypatch):
    sequencia = [(10, 20, 30, 40), (100, 200, 50, 60), (0, 0, 0, 0)]
    iterador = iter(sequencia)
    monkeypatch.setattr(
        "rede_zero.inferencia.selecao_roi.carregar_primeiro_frame", _frame_fake
    )
    monkeypatch.setattr(
        "rede_zero.inferencia.selecao_roi.cv2.selectROI",
        lambda *_a, **_k: next(iterador),
    )
    monkeypatch.setattr(
        "rede_zero.inferencia.selecao_roi.cv2.destroyAllWindows", lambda *_a: None
    )
    rois = selecionar_rois("fake.mp4")
    assert len(rois) == 2
    assert all(isinstance(r, BBox) for r in rois)
    assert [r.id for r in rois] == [0, 1]
    assert rois[0].to_tlwh() == (10.0, 20.0, 30.0, 40.0)


def test_selecionar_rois_sem_selecao_erro(monkeypatch):
    monkeypatch.setattr(
        "rede_zero.inferencia.selecao_roi.carregar_primeiro_frame", _frame_fake
    )
    monkeypatch.setattr(
        "rede_zero.inferencia.selecao_roi.cv2.selectROI", lambda *_a, **_k: (0, 0, 0, 0)
    )
    monkeypatch.setattr(
        "rede_zero.inferencia.selecao_roi.cv2.destroyAllWindows", lambda *_a: None
    )
    with pytest.raises(ValueError):
        selecionar_rois("fake.mp4")


def test_carregar_primeiro_frame_video_invalido(tmp_path):
    with pytest.raises(IOError):
        carregar_primeiro_frame(tmp_path / "nao_existe.mp4")
