#!/usr/bin/env python3
"""CLI de treino (SCR-12): --config, --resume, --paradigma, --dry-run."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rede_zero.data.dataset_treino import TrackingDataset
from rede_zero.data.readers.base import ReaderBase
from rede_zero.data.readers.otb import OTBReader
from rede_zero.data.readers.video import VideoReader
from rede_zero.treino.trainer import Treinador, carregar_config


def carregar_readers(raiz: Path) -> list[ReaderBase]:
    """Forma-A: subpastas com images/ (OTB). Forma-B: .mp4 com object_id_*.csv vizinhos."""
    readers: list[ReaderBase] = []
    for img_dir in sorted(raiz.rglob("images")):
        if img_dir.is_dir():
            try:
                reader = OTBReader(img_dir.parent)
                if reader.n_frames > 0:
                    readers.append(reader)
            except Exception:
                pass
    for video in sorted(raiz.rglob("*.mp4")):
        gt_dir = next(
            (d for d in [video.parent, *video.parent.rglob("*")]
             if d.is_dir() and list(d.glob("object_id_*.csv"))),
            None,
        )
        if gt_dir is None:
            continue
        try:
            readers.append(VideoReader(video, gt_dir))
        except Exception:
            pass
    return readers


def main() -> int:
    parser = argparse.ArgumentParser(description="Treinar um paradigma de rastreamento.")
    parser.add_argument("--config", required=True, help="Caminho do YAML de configuração.")
    parser.add_argument("--paradigma", help="Sobrepõe o paradigma da config (siamese|regressao).")
    parser.add_argument("--resume", help="Caminho de ckpt ou 'last' para retomar.")
    parser.add_argument("--epochs", type=int, help="Sobrepõe o nº de epochs da config.")
    parser.add_argument("--num-amostras", type=int, help="Sobrepõe o nº de amostras do dataset.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Monta tudo e imprime o resumo sem treinar.")
    args = parser.parse_args()

    cfg = carregar_config(args.config)
    if args.paradigma:
        cfg["paradigma"] = args.paradigma
    if args.epochs:
        cfg.setdefault("treino", {})["epochs"] = args.epochs

    raiz = Path(cfg.get("dados", {}).get("raiz", "data/real"))
    readers = carregar_readers(raiz)
    if not readers:
        print(f"Erro: nenhuma sequência encontrada em '{raiz}'.", file=sys.stderr)
        return 1

    imgs = cfg.get("imagens", {})
    dataset = TrackingDataset(
        readers,
        template_size=int(imgs.get("template", 128)),
        busca_size=int(imgs.get("busca", 256)),
        patch_size=int(imgs.get("patch", 128)),
        num_amostras=int(args.num_amostras or 1000),
    )
    treino_cfg = cfg.get("treino", {})
    loader = DataLoader(
        dataset,
        batch_size=int(treino_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(treino_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    treinador = Treinador(cfg, loader_treino=loader, resume=args.resume)
    print(f"Paradigma: {cfg.get('paradigma')} | device: {treinador.device} | "
          f"readers: {len(readers)} | amostras: {len(dataset)} | "
          f"epochs: {treinador.total_epochs}")

    if args.dry_run:
        print("Dry-run OK — nada foi treinado.")
        return 0

    resultado = treinador.treinar()
    print(f"Concluído: {resultado}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
