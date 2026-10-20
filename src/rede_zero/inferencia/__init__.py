"""Componentes de inferência independentes do pacote de treinamento."""

from rede_zero.inferencia.pipeline import (
    ObjetoPerdido,
    ResultadoPipeline,
    executar_pipeline,
)

__all__ = ["ObjetoPerdido", "ResultadoPipeline", "executar_pipeline"]
