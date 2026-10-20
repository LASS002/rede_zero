"""Testes de perdas e métricas de rastreamento (SCR-10)."""

import pytest
import torch

from rede_zero.data.readers import BBox
from rede_zero.paradigmas.siamese import ModeloSiamese
from rede_zero.treino.losses import alvo_gaussiano, perda_regressao, perda_siamese
from rede_zero.treino.metricas import curva_precisao, curva_success, iou, resumo_rastreamento


def test_alvo_gaussiano_tem_pico_no_deslocamento_central():
    alvo = alvo_gaussiano(torch.zeros(1, 2), (5, 5))
    assert alvo.shape == (1, 1, 5, 5)
    assert alvo[0, 0, 2, 2] == pytest.approx(1.0)
    assert alvo[0, 0, 0, 0] < alvo[0, 0, 2, 2]


def test_perdas_siamese_e_regressao_propagam_gradiente():
    score_map = torch.zeros(2, 1, 9, 9, requires_grad=True)
    offset = torch.zeros(2, 2, requires_grad=True)
    perda_sia = perda_siamese(
        {"score_map": score_map, "offset": offset},
        {"alvo_siamese": torch.tensor([[0.0, 0.0], [0.25, -0.25]])},
    )
    perda_sia.backward()
    assert score_map.grad is not None
    assert offset.grad is not None
    assert torch.isfinite(
        ModeloSiamese().perda_fn(
            {"score_map": score_map.detach(), "offset": offset.detach()},
            {"alvo_siamese": torch.zeros(2, 2)},
        )
    )

    delta = torch.zeros(2, 4, requires_grad=True)
    logits = torch.zeros(2, 1, requires_grad=True)
    perda_reg = perda_regressao(
        {"delta": delta, "conf_logits": logits},
        {"alvo_regressao": torch.ones(2, 4), "confianca": torch.ones(2, 1)},
    )
    perda_reg.backward()
    assert delta.grad is not None
    assert logits.grad is not None


def test_iou_e_curvas_com_casos_conhecidos():
    referencia = BBox(0, 0, 10, 10)
    sobreposta = BBox(5, 0, 10, 10)
    distante = BBox(30, 0, 10, 10)
    assert iou(referencia, referencia) == pytest.approx(1.0)
    assert iou(referencia, sobreposta) == pytest.approx(1 / 3)
    assert iou(referencia, distante) == 0.0

    success = curva_success([1.0, 0.5, 0.0], limiares=[0.5, 0.75])
    assert success == {0.5: pytest.approx(2 / 3), 0.75: pytest.approx(1 / 3)}

    precisao = curva_precisao([referencia, distante], [referencia, referencia], [5, 30])
    assert precisao == {5.0: pytest.approx(0.5), 30.0: pytest.approx(1.0)}
    resumo = resumo_rastreamento([referencia, distante], [referencia, referencia])
    assert resumo["iou_medio"] == pytest.approx(0.5)
    assert resumo["precisao_20px"] == pytest.approx(0.5)
