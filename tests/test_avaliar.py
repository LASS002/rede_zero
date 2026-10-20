"""Testes da CLI e rotinas de avaliação de rastreamento (SCR-13)."""

import json
from pathlib import Path
import subprocess
import sys
import torch

from rede_zero.data.readers import BBox
from rede_zero.data.sintetico import gerar_video_sintetico
from rede_zero.paradigmas import get_paradigma
from rede_zero.treino import avaliar_rastreamento, carregar_modelo, obter_reader


def _criar_checkpoint_fake(destino: Path, paradigma: str = "regressao") -> Path:
    modelo = get_paradigma(paradigma)
    destino.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 1,
            "loss": 0.05,
            "paradigma": paradigma,
            "versao": "0.1.0",
            "model_state_dict": modelo.state_dict(),
        },
        destino,
    )
    return destino


def test_carregar_modelo_e_reader(tmp_path: Path):
    ckpt_path = _criar_checkpoint_fake(tmp_path / "ckpt.pt", "regressao")
    modelo, nome = carregar_modelo(ckpt_path, device="cpu")
    assert nome == "regressao"
    assert modelo.nome == "regressao"


def test_avaliar_rastreamento_video_sintetico(tmp_path: Path):
    # Gera vídeo sintético curto
    out_dir = tmp_path / "sintetico"
    video_path, gt_path = gerar_video_sintetico(out_dir, n_frames=5, n_objetos=2, semente=42)

    reader = obter_reader(Path(video_path), Path(gt_path))
    assert reader.n_frames == 5

    modelo = get_paradigma("regressao")
    modelo.eval()

    resumo, detalhes = avaliar_rastreamento(reader, modelo, limiar_confianca=0.0)
    assert "iou_medio" in resumo
    assert "success_auc" in resumo
    assert "precisao_20px" in resumo
    assert resumo["n_frames"] == 5
    assert resumo["n_objetos"] == 2
    assert len(detalhes) > 0


def test_cli_avaliar_subprocess(tmp_path: Path):
    out_dir = tmp_path / "sintetico"
    video_path, gt_path = gerar_video_sintetico(out_dir, n_frames=4, n_objetos=1, semente=42)

    ckpt_path = _criar_checkpoint_fake(tmp_path / "runs" / "exp" / "checkpoints" / "best.pt", "regressao")
    out_json = tmp_path / "runs" / "exp" / "metricas.json"
    out_csv = tmp_path / "runs" / "exp" / "metricas_detalhes.csv"

    cmd = [
        sys.executable,
        "scripts/avaliar.py",
        "--ckpt", str(ckpt_path),
        "--video", str(video_path),
        "--gt", str(gt_path),
        "--out", str(out_json),
        "--csv-out", str(out_csv),
        "--device", "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Erro na CLI: {res.stderr}"
    assert out_json.is_file()
    assert out_csv.is_file()

    with open(out_json, "r", encoding="utf-8") as f:
        dados = json.load(f)

    assert dados["paradigma"] == "regressao"
    assert "iou_medio" in dados
    assert "success_auc" in dados
    assert "precisao_20px" in dados
