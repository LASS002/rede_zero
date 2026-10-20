"""Geração de vídeos curtos de objetos coloridos em fundos verdes variáveis."""

import csv
from pathlib import Path
import random

import cv2
import numpy as np


def gerar_video_sintetico(
    destino: str | Path,
    n_frames: int = 90,
    n_objetos: int = 3,
    largura: int = 320,
    altura: int = 240,
    fps: int = 15,
    semente: int = 42,
) -> tuple[Path, Path]:
    """Salva vídeo MP4 e CSV de tracking e retorna seus caminhos."""
    if not 1 <= n_frames <= 90:
        raise ValueError("n_frames deve estar entre 1 e 90.")
    if not 1 <= n_objetos <= 8:
        raise ValueError("n_objetos deve estar entre 1 e 8.")
    if largura < 96 or altura < 96 or fps < 1:
        raise ValueError("Dimensões mínimas são 96x96 e fps deve ser positivo.")

    pasta = Path(destino)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_video, caminho_csv = pasta / "video.mp4", pasta / "video.csv"
    rng = random.Random(semente)
    gerador_ruido = np.random.default_rng(semente)
    writer = cv2.VideoWriter(
        str(caminho_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Não foi possível criar o vídeo: {caminho_video}")

    raio_max = max(5, min(9, largura // 40, altura // 30))
    objetos = _criar_objetos(n_objetos, largura, altura, raio_max, rng)
    eixo_x, eixo_y = np.meshgrid(np.linspace(0, 1, largura), np.linspace(0, 1, altura))
    mapa_sombra = cv2.resize(
        gerador_ruido.integers(0, 256, (5, 7), dtype=np.uint8),
        (largura, altura), interpolation=cv2.INTER_CUBIC,
    )
    with caminho_csv.open("w", newline="", encoding="utf-8") as arquivo_csv:
        csv_writer = csv.writer(arquivo_csv)
        csv_writer.writerow(
            ["Frame", "Tracker ID", "Label", "X_min", "Y_min", "X_max", "Y_max", "Confidence"]
        )
        try:
            for indice_frame in range(n_frames):
                frame = _fundo_verde(eixo_x, eixo_y, indice_frame, mapa_sombra)
                _mover_objetos(objetos, largura, altura, rng)
                for objeto in objetos:
                    x, y, raio = objeto["x"], objeto["y"], objeto["raio"]
                    _desenhar_objeto(
                        frame, x, y, raio, objeto["cores"], indice_frame, objeto["fase"]
                    )
                    semi_lado = int(np.ceil(1.7 * raio + max(2, raio // 3)))
                    x1, y1 = max(0, x - semi_lado), max(0, y - semi_lado)
                    x2, y2 = min(largura, x + semi_lado + 1), min(altura, y + semi_lado + 1)
                    csv_writer.writerow(
                        [indice_frame, objeto["id"], "objeto", x1, y1, x2, y2, 1.0]
                    )
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    return caminho_video, caminho_csv


def _criar_objetos(n: int, largura: int, altura: int, raio: int, rng: random.Random) -> list[dict]:
    objetos = []
    for indice in range(n):
        tamanho = rng.randint(max(4, raio - 2), raio)
        margem = int(np.ceil(1.7 * tamanho + 5))
        for _ in range(500):
            x, y = rng.randint(margem, largura - margem), rng.randint(margem, altura - margem)
            separa = lambda obj: (1.7 * (tamanho + obj["raio"]) + 6) ** 2
            if all((x - obj["x"]) ** 2 + (y - obj["y"]) ** 2 > separa(obj) for obj in objetos):
                break
        else:
            raise ValueError("Não há espaço para posicionar objetos sem sobreposição.")
        cores = [tuple(rng.randint(90, 255) for _ in range(3)) for _ in range(8)]
        objetos.append({"id": indice + 1, "x": x, "y": y, "raio": tamanho,
                        "vx": rng.choice((-2, -1, 1, 2)), "vy": rng.choice((-2, -1, 1, 2)),
                        "cores": cores, "fase": rng.randrange(90)})
    return objetos


def _mover_objetos(objetos: list[dict], largura: int, altura: int, rng: random.Random) -> None:
    anteriores = [(obj["x"], obj["y"]) for obj in objetos]
    for obj in objetos:
        obj["vx"] = max(-3, min(3, obj["vx"] + rng.choice((-1, 0, 0, 1))))
        obj["vy"] = max(-3, min(3, obj["vy"] + rng.choice((-1, 0, 0, 1))))
        margem = int(np.ceil(1.7 * obj["raio"] + 5))
        for eixo, limite in (("x", largura), ("y", altura)):
            posicao = obj[eixo] + obj["v" + eixo]
            if posicao < margem or posicao >= limite - margem:
                obj["v" + eixo] *= -1
            obj[eixo] = max(margem, min(limite - margem - 1, posicao))
    for indice, atual in enumerate(objetos):
        for indice_outro, outro in enumerate(objetos[indice + 1:], indice + 1):
            distancia = (atual["x"] - outro["x"]) ** 2 + (atual["y"] - outro["y"]) ** 2
            separacao = 1.7 * (atual["raio"] + outro["raio"]) + 6
            if distancia < separacao**2:
                atual["x"], atual["y"] = anteriores[indice]
                outro["x"], outro["y"] = anteriores[indice_outro]
                atual["vx"], outro["vx"] = -atual["vx"], -outro["vx"]
                atual["vy"], outro["vy"] = -atual["vy"], -outro["vy"]


def _fundo_verde(
    x: np.ndarray, y: np.ndarray, frame: int, mapa_sombra: np.ndarray
) -> np.ndarray:
    altura, largura = x.shape
    deslocado = np.roll(mapa_sombra, shift=(frame // 8, frame // 11), axis=(0, 1))
    sombras = cv2.GaussianBlur(deslocado, (0, 0), 32).astype(np.float32) / 255
    verde = 105 + 30 * np.sin(2 * np.pi * x + frame * 0.08) + 18 * np.cos(3 * np.pi * y)
    verde = np.clip(verde * (0.68 + 0.62 * sombras), 45, 190).astype(np.uint8)
    fundo = np.empty((altura, largura, 3), dtype=np.uint8)
    fundo[:, :, 0], fundo[:, :, 1], fundo[:, :, 2] = verde // 5, verde, verde // 4
    return fundo


def _desenhar_objeto(
    frame: np.ndarray, x: int, y: int, raio: int,
    cores: list[tuple[int, int, int]], indice: int, fase: int,
) -> None:
    if ((indice + fase) // 12) % 2:
        for raio_angulo in range(0, 360, 60):
            angulo = np.deg2rad(raio_angulo + fase * 3)
            ponta = (int(x + np.cos(angulo) * raio * 1.7), int(y + np.sin(angulo) * raio * 1.7))
            cv2.line(frame, (x, y), ponta, cores[(raio_angulo // 60) % len(cores)], max(2, raio // 3))
    cv2.circle(frame, (x, y), raio, cores[indice % len(cores)], -1)
    passo = max(2, raio // 2)
    for py in range(y - raio, y + raio + 1, passo):
        for px in range(x - raio, x + raio + 1, passo):
            if (px - x) ** 2 + (py - y) ** 2 <= raio**2:
                cor = cores[(px + py + indice) % len(cores)]
                cv2.rectangle(frame, (px, py), (px + passo - 1, py + passo - 1), cor, -1)
