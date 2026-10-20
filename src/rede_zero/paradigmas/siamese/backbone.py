"""Backbone compartilhado do paradigma Siamese."""

from collections.abc import Sequence

from torch import Tensor, nn


class BackboneSiamese(nn.Module):
    """Quatro blocos Conv2d, BatchNorm, ReLU e MaxPool compartilhados."""

    def __init__(self, canais: Sequence[int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        if len(canais) != 4:
            raise ValueError("O backbone Siamese requer exatamente quatro blocos.")

        entrada = 3
        blocos: list[nn.Module] = []
        for saida in canais:
            blocos.append(
                nn.Sequential(
                    nn.Conv2d(entrada, saida, kernel_size=3, padding=1),
                    nn.BatchNorm2d(saida),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
            entrada = saida
        self.blocos = nn.Sequential(*blocos)
        self.canais_saida = entrada

    def forward(self, imagem: Tensor) -> Tensor:
        return self.blocos(imagem)
