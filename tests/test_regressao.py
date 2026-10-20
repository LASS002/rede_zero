"""Testes do paradigma de regressão (SCR-07)."""

import numpy as np
import pytest
import torch

from rede_zero.data.readers import BBox
from rede_zero.paradigmas import get_paradigma
from rede_zero.paradigmas.regressao import ModeloRegressao


def test_instanciacao_regressao():
    modelo_direto = ModeloRegressao()
    assert modelo_direto.nome == "regressao"

    modelo_registry = get_paradigma("regressao")
    assert isinstance(modelo_registry, ModeloRegressao)


def test_forward_batch_shape():
    modelo = ModeloRegressao()
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 128)
    treino_batch = {"patch": x}

    saidas = modelo(treino_batch)
    assert "predicoes" in saidas
    assert "delta" in saidas
    assert "confianca" in saidas
    assert "conf_logits" in saidas

    assert saidas["predicoes"].shape == (batch_size, 5)
    assert saidas["delta"].shape == (batch_size, 4)
    assert saidas["confianca"].shape == (batch_size, 1)

    # Confiança deve estar no intervalo [0, 1] devido à sigmóide
    assert (saidas["confianca"] >= 0.0).all() and (saidas["confianca"] <= 1.0).all()


def test_perda_fn_e_backward():
    modelo = ModeloRegressao()
    batch_size = 2
    x = torch.randn(batch_size, 3, 128, 128)
    treino_batch = {"patch": x}
    alvos = {
        "alvo_regressao": torch.zeros(batch_size, 4),
        "confianca": torch.ones(batch_size, 1),
    }

    saidas = modelo(treino_batch)
    perda = modelo.perda_fn(saidas, alvos)

    assert isinstance(perda, torch.Tensor)
    assert perda.dim() == 0  # escalar
    assert perda.item() > 0

    perda.backward()
    for param in modelo.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_prever_roi():
    modelo = ModeloRegressao()
    modelo.eval()

    patch_np = np.zeros((128, 128, 3), dtype=np.uint8)
    roi_ant = BBox(x=100.0, y=100.0, w=50.0, h=50.0, id=1, label="objeto")
    contexto = {"roi_anterior": roi_ant}

    nova_roi = modelo.prever_roi(patch_np, contexto)
    assert isinstance(nova_roi, BBox)
    assert nova_roi.id == 1
    assert nova_roi.label == "objeto"
    assert nova_roi.w > 0
    assert nova_roi.h > 0
    assert 0.0 <= nova_roi.conf <= 1.0
