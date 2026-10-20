"""Testes do loop de treino, checkpointing e logging (SCR-11)."""

import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset

from rede_zero.treino import Treinador, carregar_config


class _SyntheticSiameseDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "template": torch.randn(3, 128, 128),
            "busca": torch.randn(3, 256, 256),
            "alvo_siamese": torch.tensor([0.1, -0.1], dtype=torch.float32),
        }


class _SyntheticRegressaoDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "patch": torch.randn(3, 128, 128),
            "alvo_regressao": torch.tensor([0.1, -0.1, 0.05, 0.05], dtype=torch.float32),
            "confianca": torch.tensor([1.0], dtype=torch.float32),
        }


def test_carregar_config(tmp_path: Path):
    base_file = tmp_path / "base.yaml"
    base_file.write_text("experimento: base\nsemente: 42\ntreino:\n  lr: 0.001\n  amp: false\n")

    child_file = tmp_path / "child.yaml"
    child_file.write_text(
        "_base_: base.yaml\nexperimento: exp1\nparadigma: regressao\ntreino:\n  epochs: 5\n"
    )

    cfg = carregar_config(child_file)
    assert cfg["experimento"] == "exp1"
    assert cfg["semente"] == 42
    assert cfg["paradigma"] == "regressao"
    assert cfg["treino"]["lr"] == 0.001
    assert cfg["treino"]["epochs"] == 5


def test_treino_uma_epoca_siamese_salva_ckpt_e_log(tmp_path: Path):
    exp_dir = tmp_path / "exp_siamese"
    cfg = {
        "experimento": "exp_siamese",
        "paradigma": "siamese",
        "device": "cpu",
        "semente": 123,
        "treino": {
            "epochs": 1,
            "lr": 1e-3,
            "amp": False,
            "ckpt_dir": str(exp_dir / "checkpoints"),
        },
    }

    loader = DataLoader(_SyntheticSiameseDataset(), batch_size=2)
    treinador = Treinador(config=cfg, loader_treino=loader)
    treinador.log_csv = exp_dir / "train_log.csv"

    resultado = treinador.treinar(epochs=1)
    assert resultado["ult_epoch"] == 1
    assert resultado["melhor_loss"] > 0

    # Verifica checkpoint
    ckpt_last = exp_dir / "checkpoints" / "last.pt"
    ckpt_best = exp_dir / "checkpoints" / "best.pt"
    assert ckpt_last.is_file()
    assert ckpt_best.is_file()

    # Verifica log CSV
    assert treinador.log_csv.is_file()
    with open(treinador.log_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["epoch", "loss", "lr", "tempo_s"]
    assert rows[1][0] == "1"


def test_treino_regressao_e_resume(tmp_path: Path):
    exp_dir = tmp_path / "exp_reg"
    cfg = {
        "experimento": "exp_reg",
        "paradigma": "regressao",
        "device": "cpu",
        "semente": 42,
        "treino": {
            "epochs": 2,
            "lr": 5e-4,
            "amp": False,
            "ckpt_dir": str(exp_dir / "checkpoints"),
        },
    }

    loader = DataLoader(_SyntheticRegressaoDataset(), batch_size=2)
    treinador = Treinador(config=cfg, loader_treino=loader)
    treinador.log_csv = exp_dir / "train_log.csv"

    # Treina época 1
    treinador.treinar(epochs=1)
    ckpt_path = exp_dir / "checkpoints" / "last.pt"
    assert ckpt_path.is_file()

    # Resuma para época 2 com novo treinador
    treinador_resumido = Treinador(config=cfg, loader_treino=loader, resume=ckpt_path)
    treinador_resumido.log_csv = exp_dir / "train_log.csv"
    assert treinador_resumido.inicio_epoch == 2

    resultado2 = treinador_resumido.treinar(epochs=2)
    assert resultado2["ult_epoch"] == 2

    # Verifica se CSV agora contém épocas 1 e 2
    with open(treinador_resumido.log_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3  # header + ep1 + ep2
    assert rows[1][0] == "1"
    assert rows[2][0] == "2"
