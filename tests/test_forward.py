"""Testes de forward e independência multiobjeto (até 8 objetos) (SCR-09)."""

import numpy as np
import pytest
import torch

from rede_zero.data.readers import BBox
from rede_zero.paradigmas import get_paradigma


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_forward_shapes_siamese(batch_size: int):
    modelo = get_paradigma("siamese")
    modelo.eval()

    template = torch.randn(batch_size, 3, 128, 128)
    busca = torch.randn(batch_size, 3, 256, 256)

    with torch.no_grad():
        saidas = modelo({"template": template, "busca": busca})

    assert "score_map" in saidas
    assert "offset" in saidas
    assert saidas["score_map"].shape == (batch_size, 1, 9, 9)
    assert saidas["offset"].shape == (batch_size, 2)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_forward_shapes_regressao(batch_size: int):
    modelo = get_paradigma("regressao")
    modelo.eval()

    patch = torch.randn(batch_size, 3, 128, 128)

    with torch.no_grad():
        saidas = modelo({"patch": patch})

    assert "predicoes" in saidas
    assert "delta" in saidas
    assert "confianca" in saidas
    assert "conf_logits" in saidas

    assert saidas["predicoes"].shape == (batch_size, 5)
    assert saidas["delta"].shape == (batch_size, 4)
    assert saidas["confianca"].shape == (batch_size, 1)


def test_oito_objetos_independentes_regressao():
    """Valida que até 8 objetos são processados de forma estritamente independente sem estado compartilhado."""
    modelo = get_paradigma("regressao")
    modelo.eval()

    n_objetos = 8
    patches = torch.randn(n_objetos, 3, 128, 128)

    # 1. Forward em batch dos 8 objetos simultaneamente
    with torch.no_grad():
        saida_batch = modelo({"patch": patches})
    pred_batch = saida_batch["predicoes"]

    # 2. Forward individual de cada objeto
    for i in range(n_objetos):
        patch_indiv = patches[i : i + 1]
        with torch.no_grad():
            saida_indiv = modelo({"patch": patch_indiv})
        pred_indiv = saida_indiv["predicoes"]

        # Saída do objeto i no batch deve ser idêntica ao forward isolado
        assert torch.allclose(pred_batch[i : i + 1], pred_indiv, atol=1e-5), (
            f"Objeto {i} apresentou interferência entre chamadas!"
        )

    # 3. Inferência de ROI para 8 objetos independentes
    rois = [
        BBox(x=10.0 * i, y=20.0 * i, w=30.0, h=30.0, id=i, label=f"obj_{i}")
        for i in range(n_objetos)
    ]
    novas_rois = []
    for i, roi in enumerate(rois):
        patch_np = (patches[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        nova_roi = modelo.prever_roi(patch_np, {"roi_anterior": roi})
        novas_rois.append(nova_roi)
        assert nova_roi.id == roi.id
        assert nova_roi.label == roi.label

    assert len(novas_rois) == n_objetos


def test_oito_objetos_independentes_siamese():
    """Valida independência para o paradigma Siamese com 8 pares distintos."""
    modelo = get_paradigma("siamese")
    modelo.eval()

    n_objetos = 8
    templates = torch.randn(n_objetos, 3, 128, 128)
    buscas = torch.randn(n_objetos, 3, 256, 256)

    with torch.no_grad():
        saida_batch = modelo({"template": templates, "busca": buscas})

    for i in range(n_objetos):
        with torch.no_grad():
            saida_indiv = modelo({"template": templates[i : i + 1], "busca": buscas[i : i + 1]})

        assert torch.allclose(
            saida_batch["score_map"][i : i + 1], saida_indiv["score_map"], atol=1e-5
        )
        assert torch.allclose(
            saida_batch["offset"][i : i + 1], saida_indiv["offset"], atol=1e-5
        )
