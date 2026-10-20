"""Varredura e levantamento estatístico de datasets para rastreamento (SCR-02)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from rede_zero.data.readers.otb import OTBReader
from rede_zero.data.readers.video import VideoReader


@dataclass
class SequenciaStats:
    nome: str
    formato: str
    n_frames: int
    n_bboxes: int
    ids_unicos: List[int]
    larguras: List[float] = field(default_factory=list)
    alturas: List[float] = field(default_factory=list)
    anomalias: List[str] = field(default_factory=list)


def inventariar_sequencia_otb(seq_dir: Path) -> SequenciaStats:
    reader = OTBReader(seq_dir)
    stats = SequenciaStats(
        nome=seq_dir.name,
        formato="Formato-A (OTB/Frames)",
        n_frames=reader.n_frames,
        n_bboxes=0,
        ids_unicos=[],
    )
    ids_set = set()
    for i in range(reader.n_frames):
        bboxes = reader.get_gt(i)
        stats.n_bboxes += len(bboxes)
        for b in bboxes:
            if b.id != -1:
                ids_set.add(b.id)
            if b.w <= 0 or b.h <= 0:
                stats.anomalias.append(f"Frame {i}: bbox com dimensão inválida (w={b.w}, h={b.h})")
            stats.larguras.append(b.w)
            stats.alturas.append(b.h)
    stats.ids_unicos = sorted(list(ids_set))
    return stats


def inventariar_video_tracking(video_path: Path, gt_path: Optional[Path]) -> SequenciaStats:
    reader = VideoReader(video_path, gt_path)
    stats = SequenciaStats(
        nome=video_path.name,
        formato="Formato-B (Vídeo+CSV)",
        n_frames=reader.n_frames,
        n_bboxes=0,
        ids_unicos=[],
    )
    ids_set = set()
    for i in range(reader.n_frames):
        bboxes = reader.get_gt(i)
        stats.n_bboxes += len(bboxes)
        for b in bboxes:
            if b.id != -1:
                ids_set.add(b.id)
            if b.w <= 0 or b.h <= 0:
                stats.anomalias.append(f"Frame {i}: bbox com dimensão inválida (w={b.w}, h={b.h})")
            stats.larguras.append(b.w)
            stats.alturas.append(b.h)
    stats.ids_unicos = sorted(list(ids_set))
    return stats


def varrer_dataset(raiz: Path | str) -> List[SequenciaStats]:
    raiz_path = Path(raiz)
    if not raiz_path.exists():
        return []

    resultados: List[SequenciaStats] = []

    # 1. Busca por Formato-A: subdiretórios com 'images' ou imagens diretas
    for img_dir in sorted(raiz_path.rglob("images")):
        if img_dir.is_dir():
            seq_dir = img_dir.parent
            try:
                stats = inventariar_sequencia_otb(seq_dir)
                if stats.n_frames > 0:
                    resultados.append(stats)
            except Exception as e:
                pass

    # 2. Busca por Formato-B: vídeos .mp4 acompanhados de csvs de tracking
    for video in sorted(raiz_path.rglob("*.mp4")):
        parent = video.parent
        # procura CSVs de tracking no mesmo diretório ou em subpastas vailatracker
        csv_dirs = [parent] + [p for p in parent.rglob("*") if p.is_dir() and "tracker" in p.name.lower()]
        target_gt = None
        for d in csv_dirs:
            if list(d.glob("object_id_*.csv")):
                target_gt = d
                break
            if (d / "all_id_detection.csv").exists():
                target_gt = d / "all_id_detection.csv"
                break
        try:
            stats = inventariar_video_tracking(video, target_gt)
            resultados.append(stats)
        except Exception:
            pass

    return resultados


def gerar_relatorio_md(sequencias: List[SequenciaStats], destino: Optional[Path | str] = None) -> str:
    total_seqs = len(sequencias)
    total_frames = sum(s.n_frames for s in sequencias)
    total_bboxes = sum(s.n_bboxes for s in sequencias)
    todas_w = [w for s in sequencias for w in s.larguras]
    todas_h = [h for s in sequencias for h in s.alturas]

    w_media = float(np.mean(todas_w)) if todas_w else 0.0
    h_media = float(np.mean(todas_h)) if todas_h else 0.0
    anomalias_totais = sum(len(s.anomalias) for s in sequencias)

    linhas = [
        "# Relatório de Inventário do Dataset",
        "",
        "## Resumo Geral",
        f"- **Sequências encontradas**: {total_seqs}",
        f"- **Total de frames**: {total_frames}",
        f"- **Total de bounding boxes anotadas**: {total_bboxes}",
        f"- **Tamanho médio de bbox**: {w_media:.1f} x {h_media:.1f} px",
        f"- **Inconsistências/anomalias detectadas**: {anomalias_totais}",
        "",
        "## Detalhamento por Sequência",
        "",
        "| Sequência | Formato | Frames | BBoxes | Objetos Únicos | WxH Médio | Anomalias |",
        "|---|---|---|---|---|---|---|",
    ]

    for s in sequencias:
        sw = f"{np.mean(s.larguras):.1f}" if s.larguras else "0.0"
        sh = f"{np.mean(s.alturas):.1f}" if s.alturas else "0.0"
        linhas.append(
            f"| `{s.nome}` | {s.formato} | {s.n_frames} | {s.n_bboxes} | "
            f"{len(s.ids_unicos)} | {sw}x{sh} px | {len(s.anomalias)} |"
        )

    linhas.append("")
    conteudo = "\n".join(linhas)

    if destino is not None:
        p = Path(destino)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(conteudo, encoding="utf-8")

    return conteudo
