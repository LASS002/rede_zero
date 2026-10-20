"""Testes do inventário de dataset (SCR-02)."""

import csv
import json
from pathlib import Path
import cv2
import numpy as np

from rede_zero.data.inventario import gerar_relatorio_md, varrer_dataset


def test_inventario_dataset_sintetico(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Cria sequência formato OTB
    seq_otb = raw_dir / "seq_otb"
    img_dir = seq_otb / "images"
    img_dir.mkdir(parents=True)
    cv2.imwrite(str(img_dir / "frame_000000.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
    cv2.imwrite(str(img_dir / "frame_000001.jpg"), np.zeros((50, 50, 3), dtype=np.uint8))
    (seq_otb / "gt.txt").write_text("5,5,10,15\n6,6,10,15\n", encoding="utf-8")

    # Cria vídeo com tracking csv
    seq_video_dir = raw_dir / "video_seq"
    seq_video_dir.mkdir(parents=True)
    video_path = seq_video_dir / "video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 10.0, (80, 60))
    for _ in range(3):
        vw.write(np.zeros((60, 80, 3), dtype=np.uint8))
    vw.release()

    csv_path = seq_video_dir / "object_id_01.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Tracker ID", "Label", "X_min", "Y_min", "X_max", "Y_max"])
        w.writerow([0, 1, "object", 10, 10, 20, 20])
        w.writerow([1, 1, "object", 12, 12, 22, 22])
        w.writerow([2, 1, "object", 14, 14, 24, 24])

    seqs = varrer_dataset(raw_dir)
    assert len(seqs) == 2

    nomes = {s.nome for s in seqs}
    assert "seq_otb" in nomes
    assert "video.mp4" in nomes

    total_frames = sum(s.n_frames for s in seqs)
    assert total_frames == 5  # 2 + 3

    out_file = tmp_path / "relatorio.md"
    md = gerar_relatorio_md(seqs, out_file)
    assert out_file.exists()
    assert "# Relatório de Inventário do Dataset" in md
    assert "Sequências encontradas**: 2" in md
