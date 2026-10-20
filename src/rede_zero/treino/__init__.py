"""Perdas, métricas e rotinas de treinamento."""

from rede_zero.treino.losses import perda_regressao, perda_siamese
from rede_zero.treino.metricas import curva_precisao, curva_success, iou, resumo_rastreamento
from rede_zero.treino.trainer import Treinador, carregar_config

__all__ = [
    "Treinador",
    "carregar_config",
    "perda_regressao",
    "perda_siamese",
    "iou",
    "curva_success",
    "curva_precisao",
    "resumo_rastreamento",
]
