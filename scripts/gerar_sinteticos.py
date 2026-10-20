#!/usr/bin/env python3
"""CLI para gerar vídeos sintéticos de rastreamento (SCR-04)."""

import argparse

from rede_zero.data.sintetico import gerar_video_sintetico


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera objetos coloridos em movimento sobre fundo verde variável."
    )
    parser.add_argument("--output", default="data/sinteticos/exp1", help="Pasta da sequência gerada.")
    parser.add_argument("--frames", type=int, default=90, help="Frames a gerar (máximo 90).")
    parser.add_argument("--objetos", type=int, default=3, help="Objetos a gerar (1 a 8).")
    parser.add_argument("--seed", type=int, default=42, help="Semente para geração reproduzível.")
    args = parser.parse_args()
    video, anotacoes = gerar_video_sintetico(
        args.output, n_frames=args.frames, n_objetos=args.objetos, semente=args.seed
    )
    print(f"Vídeo: {video}\nAnotações: {anotacoes}")


if __name__ == "__main__":
    main()
