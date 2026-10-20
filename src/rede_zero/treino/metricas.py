"""Métricas de rastreamento por frame."""

from collections.abc import Iterable, Sequence

from rede_zero.data.readers.base import BBox


def iou(caixa_a: BBox, caixa_b: BBox) -> float:
    """Calcula a interseção sobre união de duas caixas no formato TLWH."""
    esquerda = max(caixa_a.x1, caixa_b.x1)
    topo = max(caixa_a.y1, caixa_b.y1)
    direita = min(caixa_a.x2, caixa_b.x2)
    baixo = min(caixa_a.y2, caixa_b.y2)
    intersecao = max(0.0, direita - esquerda) * max(0.0, baixo - topo)
    uniao = caixa_a.w * caixa_a.h + caixa_b.w * caixa_b.h - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


def iou_por_frame(predicoes: Sequence[BBox], alvos: Sequence[BBox]) -> list[float]:
    """Retorna o IoU de cada par predição/alvo correspondente."""
    _validar_pares(predicoes, alvos)
    return [iou(predicao, alvo) for predicao, alvo in zip(predicoes, alvos)]


def curva_success(
    valores_iou: Iterable[float], limiares: Iterable[float] | None = None
) -> dict[float, float]:
    """Calcula taxa de frames com IoU maior ou igual a cada limiar."""
    valores = list(valores_iou)
    pontos = list(limiares) if limiares is not None else [indice / 20 for indice in range(21)]
    if not valores:
        return {float(limiar): 0.0 for limiar in pontos}
    return {
        float(limiar): sum(valor >= limiar for valor in valores) / len(valores)
        for limiar in pontos
    }


def distancia_centros(caixa_a: BBox, caixa_b: BBox) -> float:
    """Calcula a distância euclidiana em pixels entre centros de duas caixas."""
    centro_a, centro_b = caixa_a.center, caixa_b.center
    return ((centro_a[0] - centro_b[0]) ** 2 + (centro_a[1] - centro_b[1]) ** 2) ** 0.5


def curva_precisao(
    predicoes: Sequence[BBox], alvos: Sequence[BBox], limiares: Iterable[float] | None = None
) -> dict[float, float]:
    """Calcula taxa de frames dentro de cada limiar de erro de centro."""
    _validar_pares(predicoes, alvos)
    pontos = list(limiares) if limiares is not None else list(range(1, 51))
    distancias = [distancia_centros(predicao, alvo) for predicao, alvo in zip(predicoes, alvos)]
    if not distancias:
        return {float(limiar): 0.0 for limiar in pontos}
    return {
        float(limiar): sum(distancia <= limiar for distancia in distancias) / len(distancias)
        for limiar in pontos
    }


def resumo_rastreamento(predicoes: Sequence[BBox], alvos: Sequence[BBox]) -> dict[str, float]:
    """Resume IoU médio, AUC da success e precisão a 20 pixels."""
    valores_iou = iou_por_frame(predicoes, alvos)
    success = curva_success(valores_iou)
    precisao = curva_precisao(predicoes, alvos)
    return {
        "iou_medio": sum(valores_iou) / len(valores_iou) if valores_iou else 0.0,
        "success_auc": sum(success.values()) / len(success) if success else 0.0,
        "precisao_20px": precisao[20.0],
    }


def _validar_pares(predicoes: Sequence[BBox], alvos: Sequence[BBox]) -> None:
    if len(predicoes) != len(alvos):
        raise ValueError("Predições e alvos devem ter o mesmo número de frames.")
