#!/usr/bin/env python3
"""CLI para avaliar modelos de rastreamento em sequências anotadas (SCR-13)."""

import argparse
import csv
import json
from pathlib import Path
import sys
import torch

from rede_zero.treino.avaliador import avaliar_rastreamento, carregar_modelo, obter_reader


def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliação de modelos de rastreamento.")
    parser.add_argument("--ckpt", type=str, required=True, help="Caminho do checkpoint .pt.")
    parser.add_argument("--video", type=str, required=True, help="Vídeo (.mp4) ou pasta com frames.")
    parser.add_argument("--gt", type=str, default=None, help="Caminho do arquivo/pasta de ground-truth.")
    parser.add_argument("--paradigma", type=str, default=None, help="siamese ou regressao.")
    parser.add_argument("--limiar-confianca", type=float, default=0.5, help="Limiar para perda de objeto.")
    parser.add_argument("--out", type=str, default=None, help="Caminho do JSON de saída.")
    parser.add_argument("--csv-out", type=str, default=None, help="Caminho para exportar CSV por frame.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"Erro: Checkpoint não encontrado: {ckpt_path}", file=sys.stderr)
        return 1

    modelo, paradigma_nome = carregar_modelo(ckpt_path, args.paradigma, args.device)
    reader = obter_reader(Path(args.video), Path(args.gt) if args.gt else None)

    resumo, detalhes = avaliar_rastreamento(reader, modelo, args.limiar_confianca)
    resumo["paradigma"] = paradigma_nome
    resumo["ckpt"] = str(ckpt_path)

    out_path = Path(args.out) if args.out else ckpt_path.parent.parent / "metricas.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2)

    if args.csv_out and detalhes:
        csv_p = Path(args.csv_out)
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame", "id", "pred", "gt", "conf"])
            w.writeheader()
            w.writerows(detalhes)

    print(f"\n--- Resultados da Avaliação ({paradigma_nome}) ---")
    for k, v in resumo.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nMétricas salvas em: {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
