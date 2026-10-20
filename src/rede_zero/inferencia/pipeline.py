"""Loop de inferência multi-ROI, sem GUI e sem dependência de treinamento."""

from __future__ import annotations

from dataclasses import dataclass

from rede_zero.data.dataset_treino import recortar_com_padding
from rede_zero.data.readers.base import BBox, ReaderBase
from rede_zero.paradigmas.base import ParadigmaBase


@dataclass(frozen=True)
class ObjetoPerdido:
    """Registro de uma ROI removida por confiança abaixo do limiar."""

    frame: int
    indice_roi: int
    confianca: float


@dataclass
class ResultadoPipeline:
    """ROIs ativas por frame e eventos de perda de rastreamento."""

    rois_por_frame: list[list[BBox]]
    objetos_perdidos: list[ObjetoPerdido]


@dataclass
class _RastreioAtivo:
    indice_roi: int
    roi: BBox
    template: object | None = None


def executar_pipeline(
    reader: ReaderBase,
    rois_iniciais: list[BBox],
    paradigma: ParadigmaBase,
    limiar_confianca: float = 0.5,
) -> ResultadoPipeline:
    """Rastreia até oito ROIs, independentemente, ao longo dos frames do leitor."""
    if len(rois_iniciais) > 8:
        raise ValueError("O pipeline aceita no máximo oito ROIs iniciais.")
    if not 0.0 <= limiar_confianca <= 1.0:
        raise ValueError("O limiar de confiança deve estar entre 0 e 1.")
    if reader.n_frames == 0:
        return ResultadoPipeline([], [])

    frame_inicial = reader.get_frame(0)
    ativos = [
        _RastreioAtivo(indice_roi, roi, _criar_template(frame_inicial, roi, paradigma))
        for indice_roi, roi in enumerate(rois_iniciais)
    ]
    rois_por_frame = [[rastreio.roi for rastreio in ativos]]
    objetos_perdidos: list[ObjetoPerdido] = []

    for indice_frame in range(1, reader.n_frames):
        frame = reader.get_frame(indice_frame)
        proximos_ativos: list[_RastreioAtivo] = []
        for rastreio in ativos:
            patch, contexto = _preparar_entrada(frame, rastreio, paradigma)
            nova_roi = paradigma.prever_roi(patch, contexto)
            if nova_roi.conf < limiar_confianca:
                objetos_perdidos.append(
                    ObjetoPerdido(indice_frame, rastreio.indice_roi, nova_roi.conf)
                )
                continue
            proximos_ativos.append(
                _RastreioAtivo(rastreio.indice_roi, nova_roi, rastreio.template)
            )
        ativos = proximos_ativos
        rois_por_frame.append([rastreio.roi for rastreio in ativos])

    return ResultadoPipeline(rois_por_frame, objetos_perdidos)


def _criar_template(frame: object, roi: BBox, paradigma: ParadigmaBase) -> object | None:
    if paradigma.nome != "siamese":
        return None
    return recortar_com_padding(frame, *roi.center, roi.w * 2, roi.h * 2, out_size=128)


def _preparar_entrada(
    frame: object, rastreio: _RastreioAtivo, paradigma: ParadigmaBase
) -> tuple[object, dict[str, object]]:
    roi = rastreio.roi
    if paradigma.nome == "siamese":
        tamanho_busca = (roi.w * 4, roi.h * 4)
        patch = recortar_com_padding(frame, *roi.center, *tamanho_busca, out_size=256)
        return patch, {"roi": roi, "template": rastreio.template, "tamanho_busca": tamanho_busca}
    if paradigma.nome == "regressao":
        patch = recortar_com_padding(frame, *roi.center, roi.w * 2, roi.h * 2, out_size=128)
        return patch, {"roi": roi, "roi_anterior": roi}
    return frame, {"roi": roi}
