"""Rotinas de avaliação de rastreamento com cálculo de métricas (SCR-13)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

from rede_zero.data.readers import BBox, OTBReader, VideoReader
from rede_zero.inferencia.pipeline import executar_pipeline
from rede_zero.paradigmas import get_paradigma
from rede_zero.treino.metricas import resumo_rastreamento


def carregar_modelo(ckpt_path: Path, paradigma_arg: Optional[str] = None, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    paradigma_nome = paradigma_arg or ckpt.get("paradigma")
    if not paradigma_nome:
        raise ValueError("Paradigma não identificado no checkpoint. Especifique via --paradigma.")
    modelo = get_paradigma(paradigma_nome)
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.to(device)
    modelo.eval()
    return modelo, paradigma_nome


def obter_reader(video_ou_dir: Path, gt_path: Optional[Path] = None):
    if video_ou_dir.is_file() and video_ou_dir.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        if gt_path is None:
            candidato_csv = video_ou_dir.with_suffix(".csv")
            if candidato_csv.is_file():
                gt_path = candidato_csv
        return VideoReader(video_ou_dir, gt_path)
    return OTBReader(video_ou_dir, gt_path)


def avaliar_rastreamento(
    reader, modelo, limiar_confianca: float = 0.5
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if reader.n_frames == 0:
        return {"erro": "Sequência sem frames."}, []

    rois_iniciais = reader.get_gt(0)[:8]
    if not rois_iniciais:
        return {"erro": "Nenhum objeto com ground-truth no frame inicial."}, []

    resultado = executar_pipeline(reader, rois_iniciais, modelo, limiar_confianca)

    todas_preds: List[BBox] = []
    todos_alvos: List[BBox] = []
    detalhes_por_frame: List[Dict[str, Any]] = []

    for t in range(reader.n_frames):
        gts = {b.id: b for b in reader.get_gt(t)}
        preds = {b.id: b for b in resultado.rois_por_frame[t]} if t < len(resultado.rois_por_frame) else {}

        for init_box in rois_iniciais:
            obj_id = init_box.id
            if obj_id in gts:
                gt_box = gts[obj_id]
                pred_box = preds.get(obj_id, BBox(x=-9999, y=-9999, w=0, h=0, conf=0.0))
                todas_preds.append(pred_box)
                todos_alvos.append(gt_box)
                detalhes_por_frame.append({
                    "frame": t,
                    "id": obj_id,
                    "pred": pred_box.to_tlwh(),
                    "gt": gt_box.to_tlwh(),
                    "conf": pred_box.conf,
                })

    resumo = resumo_rastreamento(todas_preds, todos_alvos)
    resumo.update({
        "n_frames": reader.n_frames,
        "n_objetos": len(rois_iniciais),
        "objetos_perdidos": len(resultado.objetos_perdidos),
        "amostras_avaliadas": len(todas_preds),
    })
    return resumo, detalhes_por_frame
