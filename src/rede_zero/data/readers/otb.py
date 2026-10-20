"""Leitor de sequências de imagens e anotações estilo OTB / Formato-A."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from rede_zero.data.readers.base import BBox, ReaderBase


class OTBReader(ReaderBase):
    """Leitor de sequências de frames em diretório com ground-truth (txt ou json)."""

    def __init__(self, seq_dir: str | Path, gt_path: Optional[str | Path] = None):
        self.seq_dir = Path(seq_dir)
        self.frame_paths = self._find_frames()
        self.gt_by_frame: Dict[int, List[BBox]] = {}

        if gt_path is not None:
            self._load_gt_file(Path(gt_path))
        else:
            self._auto_load_gt()

    def _find_frames(self) -> List[Path]:
        img_dirs = [self.seq_dir / "images", self.seq_dir / "img", self.seq_dir]
        for d in img_dirs:
            if d.exists() and d.is_dir():
                frames = sorted(
                    [p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
                )
                if frames:
                    return frames
        return []

    def _auto_load_gt(self) -> None:
        for name in ("gt.txt", "groundtruth_rect.txt", "groundtruth.txt"):
            p = self.seq_dir / name
            if p.exists():
                self._load_gt_file(p)
                return

        labels_dir = self.seq_dir / "labels"
        if labels_dir.exists() and labels_dir.is_dir():
            self._load_labels_dir(labels_dir)

    def _load_gt_file(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.replace("\t", ",").replace(" ", ",").split(",") if p.strip()]
            if not parts:
                continue
            nums = [float(p) for p in parts]
            if len(nums) == 4:
                bbox = BBox(x=nums[0], y=nums[1], w=nums[2], h=nums[3], id=1)
                self.gt_by_frame.setdefault(i, []).append(bbox)
            elif len(nums) >= 5:
                # pode ser frame,x,y,w,h,id ou x,y,w,h,id
                if len(nums) == 5:
                    bbox = BBox(x=nums[0], y=nums[1], w=nums[2], h=nums[3], id=int(nums[4]))
                    self.gt_by_frame.setdefault(i, []).append(bbox)
                else:
                    frame_idx = int(nums[0])
                    bbox = BBox(x=nums[1], y=nums[2], w=nums[3], h=nums[4], id=int(nums[5]))
                    self.gt_by_frame.setdefault(frame_idx, []).append(bbox)

    def _load_labels_dir(self, labels_dir: Path) -> None:
        for idx, frame_path in enumerate(self.frame_paths):
            stem = frame_path.stem
            json_file = labels_dir / f"{stem}.json"
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                bboxes = []
                for ann in data.get("annotations", []):
                    bboxes.append(
                        BBox(
                            x=float(ann["x"]),
                            y=float(ann["y"]),
                            w=float(ann["w"]),
                            h=float(ann["h"]),
                            label=ann.get("label", "object"),
                        )
                    )
                self.gt_by_frame[idx] = bboxes

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)

    def get_frame(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self.frame_paths):
            raise IndexError(f"Frame {idx} fora do intervalo (0..{self.n_frames - 1})")
        bgr = cv2.imread(str(self.frame_paths[idx]))
        if bgr is None:
            raise RuntimeError(f"Falha ao ler frame: {self.frame_paths[idx]}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_gt(self, idx: int) -> List[BBox]:
        return self.gt_by_frame.get(idx, [])
