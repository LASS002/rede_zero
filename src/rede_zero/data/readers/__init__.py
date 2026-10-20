"""Módulo de leitores de dados para rastreamento."""

from rede_zero.data.readers.base import BBox, ReaderBase
from rede_zero.data.readers.otb import OTBReader
from rede_zero.data.readers.video import VideoReader

__all__ = ["BBox", "ReaderBase", "OTBReader", "VideoReader"]
