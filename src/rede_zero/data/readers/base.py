"""Contrato base para leitores de sequências/vídeos de rastreamento."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class BBox:
    """Bounding box em coordenadas de pixel (top-left x, y, width, height)."""
    x: float
    y: float
    w: float
    h: float
    id: int = -1
    label: str = "object"
    conf: float = 1.0

    @property
    def x1(self) -> float:
        return self.x

    @property
    def y1(self) -> float:
        return self.y

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def to_tlwh(self) -> Tuple[float, float, float, float]:
        return self.x, self.y, self.w, self.h

    def to_xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2


class ReaderBase(ABC):
    """Interface abstrata para acesso sequencial e aleatório a frames e ground-truth."""

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Número total de frames na sequência."""
        pass

    def __len__(self) -> int:
        return self.n_frames

    @abstractmethod
    def get_frame(self, idx: int) -> np.ndarray:
        """Retorna o frame `idx` como array NumPy uint8 em RGB (H, W, 3)."""
        pass

    @abstractmethod
    def get_gt(self, idx: int) -> List[BBox]:
        """Retorna as anotações de ground-truth para o frame `idx`."""
        pass
