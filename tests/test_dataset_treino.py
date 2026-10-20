"""Testes do dataset de treino (SCR-03)."""

from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from rede_zero.data.dataset_treino import TrackingDataset, recortar_com_padding
from rede_zero.data.readers import BBox, OTBReader


def test_recortar_com_padding():
    img = np.ones((50, 50, 3), dtype=np.uint8) * 100
    # Recorte central
    crop = recortar_com_padding(img, cx=25.0, cy=25.0, w=20.0, h=20.0, out_size=32)
    assert crop.shape == (32, 32, 3)

    # Recorte na borda com padding
    crop_borda = recortar_com_padding(img, cx=0.0, cy=0.0, w=20.0, h=20.0, out_size=32)
    assert crop_borda.shape == (32, 32, 3)


def test_tracking_dataset_shapes(tmp_path: Path):
    seq_dir = tmp_path / "seq"
    images_dir = seq_dir / "images"
    images_dir.mkdir(parents=True)

    # 3 frames com movimento de um objeto
    for i in range(3):
        frame = np.full((100, 100, 3), 50 + i * 20, dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"frame_{i:06d}.jpg"), frame)

    gt_file = seq_dir / "gt.txt"
    gt_file.write_text("10,10,20,20,1\n12,14,20,20,1\n15,18,20,20,1\n", encoding="utf-8")

    reader = OTBReader(seq_dir)
    dataset = TrackingDataset([reader], num_amostras=8, template_size=128, busca_size=256, patch_size=128)

    assert len(dataset) == 8
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))

    assert "template" in batch
    assert "busca" in batch
    assert "patch" in batch
    assert "alvo_siamese" in batch
    assert "alvo_regressao" in batch
    assert "confianca" in batch

    assert batch["template"].shape == (4, 3, 128, 128)
    assert batch["busca"].shape == (4, 3, 256, 256)
    assert batch["patch"].shape == (4, 3, 128, 128)
    assert batch["alvo_siamese"].shape == (4, 2)
    assert batch["alvo_regressao"].shape == (4, 4)
    assert batch["confianca"].shape == (4, 1)

    assert batch["template"].dtype == torch.float32
    assert batch["busca"].dtype == torch.float32
    assert batch["patch"].dtype == torch.float32
