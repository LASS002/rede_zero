#!/usr/bin/env python3
"""CLI para varrer e inventariar datasets de rastreamento (SCR-02)."""

import argparse
from pathlib import Path
import sys

from rede_zero.data.inventario import gerar_relatorio_md, varrer_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventário estatístico do dataset de rastreamento.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/labiocom01/Documents/yolo_cfc_retrain/data/raw",
        help="Diretório raiz dos dados a inventariar (padrão: data/raw real).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/inventario/relatorio.md",
        help="Caminho do relatório Markdown a ser gerado (padrão: data/inventario/relatorio.md).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Erro: diretório '{data_dir}' não encontrado.", file=sys.stderr)
        return 1

    print(f"Varrendo diretório: {data_dir}...")
    seqs = varrer_dataset(data_dir)
    print(f"Sequências identificadas: {len(seqs)}")

    output_path = Path(args.output)
    md = gerar_relatorio_md(seqs, output_path)
    print(f"Relatório gerado com sucesso em: {output_path}")
    print(f"\n{md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
