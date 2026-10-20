"""Testes do registry de paradigmas de rastreamento (SCR-09)."""

import pytest

from rede_zero.paradigmas import (
    ParadigmaBase,
    get_paradigma,
    listar_paradigmas,
)
from rede_zero.paradigmas.regressao import ModeloRegressao
from rede_zero.paradigmas.siamese import ModeloSiamese


def test_listar_paradigmas():
    nomes = listar_paradigmas()
    assert isinstance(nomes, tuple)
    assert "siamese" in nomes
    assert "regressao" in nomes
    assert "detector" in nomes


def test_get_paradigma_siamese():
    modelo = get_paradigma("siamese")
    assert isinstance(modelo, ParadigmaBase)
    assert isinstance(modelo, ModeloSiamese)
    assert modelo.nome == "siamese"


def test_get_paradigma_regressao():
    modelo = get_paradigma("regressao")
    assert isinstance(modelo, ParadigmaBase)
    assert isinstance(modelo, ModeloRegressao)
    assert modelo.nome == "regressao"


def test_get_paradigma_detector_lanca_not_implemented():
    with pytest.raises(NotImplementedError) as exc_info:
        get_paradigma("detector")
    assert "detector" in str(exc_info.value).lower()
    assert "scr-18" in str(exc_info.value).lower()


def test_get_paradigma_nome_invalido():
    with pytest.raises(ValueError) as exc_info:
        get_paradigma("inexistente")
    msg = str(exc_info.value)
    assert "inexistente" in msg
    assert "siamese" in msg
    assert "regressao" in msg


def test_get_paradigma_case_insensitivity():
    m1 = get_paradigma("SIAMESE")
    m2 = get_paradigma("Regressao")
    assert m1.nome == "siamese"
    assert m2.nome == "regressao"
