"""Seleção interativa de ROIs no primeiro frame do vídeo (SCR-15).

O usuário delimita até ``max_objetos`` ROIs (default 8, sem sobreposição) em uma
janela OpenCV; as ROIs retornam como ``BBox`` em pixels, prontas para o pipeline
de inferência (SCR-14/16).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rede_zero.data.readers.base import BBox

MAX_OBJETOS_PADRAO = 8
_JANELA = "Selecione o objeto (ENTER=confirmar, c/ESC=encerrar)"


def carregar_primeiro_frame(caminho_video: str | Path) -> np.ndarray:
    """Lê o primeiro frame do vídeo como RGB uint8 (H, W, 3)."""
    cap = cv2.VideoCapture(str(caminho_video))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise IOError(f"Não foi possível ler o vídeo: {caminho_video}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def selecionar_rois(
    caminho_video: str | Path, max_objetos: int = MAX_OBJETOS_PADRAO
) -> list[BBox]:
    """Janela interativa no frame 0 do vídeo para delimitar as ROIs iniciais.

    Confirma cada ROI com ENTER; encerra com seleção vazia, ESC/'c' ou ao atingir
    ``max_objetos``. Retorna ``list[BBox]`` com ids ``0..n-1`` em pixels.
    """
    if max_objetos < 1:
        raise ValueError("max_objetos deve ser >= 1")

    frame = carregar_primeiro_frame(caminho_video)
    vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    rois: list[BBox] = []
    try:
        while len(rois) < max_objetos:
            x, y, w, h = cv2.selectROI(_JANELA, vis, showCrosshair=True)
            if w <= 0 or h <= 0:
                break
            roi = BBox(x=float(x), y=float(y), w=float(w), h=float(h), id=len(rois))
            rois.append(roi)
            _desenhar(vis, roi)
    finally:
        cv2.destroyAllWindows()
    if not rois:
        raise ValueError("Nenhuma ROI foi selecionada.")
    return rois


def _desenhar(frame_bgr: np.ndarray, roi: BBox) -> None:
    x1, y1, x2, y2 = (int(v) for v in roi.to_xyxy())
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame_bgr, f"id {roi.id}", (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )
