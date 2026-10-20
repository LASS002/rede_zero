#!/usr/bin/env python3
"""CLI de inferência (SCR-16): --video, --paradigma, --ckpt, --out, --roi, --listar-paradigmas."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from rede_zero.data.readers.base import BBox
from rede_zero.data.readers.video import VideoReader
from rede_zero.inferencia.carregador import (
    carregar_modelo_para_inferencia,
    listar_paradigmas_disponiveis,
    obter_dispositivo_preferido,
)
from rede_zero.inferencia.pipeline import executar_pipeline, ResultadoPipeline
from rede_zero.inferencia.selecao_roi import selecionar_rois


def _parse_roi(s: str) -> BBox:
    """Parse 'x,y,w,h' -> BBox."""
    try:
        x, y, w, h = map(float, s.split(","))
        return BBox(x=x, y=y, w=w, h=h, id=0)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"ROI inválida: '{s}' (use x,y,w,h)") from e


def _criar_video_writer(caminho_saida: Path, fps: float, tamanho: tuple[int, int]) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(caminho_saida), fourcc, fps, tamanho)
    if not writer.isOpened():
        raise RuntimeError(f"Falha ao criar VideoWriter para '{caminho_saida}'")
    return writer


def _desenhar_rois(frame_bgr: np.ndarray, rois: list[BBox], perdidos: list[int]) -> None:
    for roi in rois:
        x1, y1, x2, y2 = (int(v) for v in roi.to_xyxy())
        cor = (0, 0, 255) if roi.id in perdidos else (0, 255, 0)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(
            frame_bgr, f"id {roi.id}", (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 1, cv2.LINE_AA,
        )


def _extrair_fps_tamanho(video_path: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return fps, w, h


def main() -> int:
    parser = argparse.ArgumentParser(description="Executar inferência de rastreamento em vídeo.")
    parser.add_argument("--video", help="Caminho do arquivo de vídeo de entrada.")
    parser.add_argument("--paradigma", choices=["siamese", "regressao", "detector"],
                        help="Paradigma de rastreamento a usar.")
    parser.add_argument("--ckpt", help="Caminho do checkpoint (.pt) do modelo treinado.")
    parser.add_argument("--out", help="Caminho do vídeo de saída anotado (.mp4).")
    parser.add_argument("--roi", type=lambda s: _parse_roi(s), action="append",
                        help="ROI inicial no formato 'x,y,w,h' (pode repetir até 8). Se omitido, abre GUI.")
    parser.add_argument("--limiar", type=float, default=0.5,
                        help="Limiar de confiança para considerar objeto perdido (default: 0.5).")
    parser.add_argument("--listar-paradigmas", action="store_true",
                        help="Lista os paradigmas disponíveis e sai.")
    args = parser.parse_args()

    if args.listar_paradigmas:
        for nome, status in listar_paradigmas_disponiveis().items():
            print(f"  {nome}: {status}")
        return 0

    # Validar argumentos obrigatórios (exceto --listar-paradigmas)
    for arg_name in ["video", "paradigma", "ckpt", "out"]:
        if not getattr(args, arg_name):
            parser.error(f"o argumento --{arg_name} é obrigatório")

    if args.listar_paradigmas:
        for nome, status in listar_paradigmas_disponiveis().items():
            print(f"  {nome}: {status}")
        return 0

    # Validar paradigma
    if args.paradigma == "detector":
        print("Erro: paradigma 'detector' ainda não implementado (SCR-18).", file=sys.stderr)
        return 1

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Erro: vídeo não encontrado: {video_path}", file=sys.stderr)
        return 1

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Erro: checkpoint não encontrado: {ckpt_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = obter_dispositivo_preferido()
    print(f"Dispositivo: {device} | Paradigma: {args.paradigma} | Checkpoint: {ckpt_path}")

    # Carregar modelo
    try:
        modelo = carregar_modelo_para_inferencia(args.paradigma, ckpt_path, device)
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}", file=sys.stderr)
        return 1

    # Preparar ROIs iniciais
    if args.roi:
        if len(args.roi) > 8:
            print("Erro: máximo 8 ROIs.", file=sys.stderr)
            return 1
        rois_iniciais = list(args.roi)
        for i, roi in enumerate(rois_iniciais):
            roi.id = i  # garantir IDs sequenciais
    else:
        try:
            rois_iniciais = selecionar_rois(video_path)
        except Exception as e:
            print(f"Erro na seleção de ROIs: {e}", file=sys.stderr)
            return 1

    print(f"ROIs iniciais: {len(rois_iniciais)}")

    # Abrir vídeo para propriedades e pipeline
    try:
        reader = VideoReader(video_path)
    except Exception as e:
        print(f"Erro ao abrir vídeo: {e}", file=sys.stderr)
        return 1

    # Executar pipeline completo para obter ROIs por frame
    try:
        resultado: ResultadoPipeline = executar_pipeline(
            reader=reader,
            rois_iniciais=rois_iniciais,
            paradigma=modelo,
            limiar_confianca=args.limiar,
        )
    except Exception as e:
        print(f"Erro durante pipeline: {e}", file=sys.stderr)
        reader.cap.release()
        return 1

    # Renderizar vídeo de saída
    fps, w, h = _extrair_fps_tamanho(video_path)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    if not writer.isOpened():
        print(f"Erro: não foi possível criar writer para {out_path}", file=sys.stderr)
        reader.cap.release()
        return 1

    try:
        # Frame 0
        frame_0 = reader.get_frame(0)
        frame_bgr = cv2.cvtColor(frame_0, cv2.COLOR_RGB2BGR)
        _desenhar_rois(frame_bgr, resultado.rois_por_frame[0], [])
        writer.write(frame_bgr)

        # Frames subsequentes
        perdidos_ate_agora = set()
        for idx_frame in range(1, reader.n_frames):
            frame = reader.get_frame(idx_frame)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            rois_frame = resultado.rois_por_frame[idx_frame]
            # Objetos perdidos neste frame
            perdidos_agora = {p.indice_roi for p in resultado.objetos_perdidos if p.frame == idx_frame}
            perdidos_ate_agora.update(perdidos_agora)

            _desenhar_rois(frame_bgr, rois_frame, list(perdidos_ate_agora))
            writer.write(frame_bgr)

        print(f"Vídeo salvo em: {out_path}")
    finally:
        reader.cap.release()
        writer.release()

    return 0


if __name__ == "__main__":
    sys.exit(main())