"""Leitor de arquivos de vídeo com anotações de tracking (CSV / Formato-B)."""

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np

from rede_zero.data.readers.base import BBox, ReaderBase


class VideoReader(ReaderBase):
    """Leitor de vídeo (.mp4/etc.) com anotações de tracking via CSV."""

    def __init__(self, video_path: str | Path, gt_path: Optional[str | Path] = None):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Vídeo não encontrado: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Falha ao abrir vídeo: {self.video_path}")

        self._n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_pos = 0
        self.gt_by_frame: Dict[int, List[BBox]] = {}

        if gt_path is not None:
            self._load_gt(Path(gt_path))

    def _load_gt(self, path: Path) -> None:
        if path.is_dir():
            csv_files = sorted(path.glob("object_id_*.csv"))
            for csv_file in csv_files:
                self._load_csv_file(csv_file)
        elif path.is_file():
            self._load_csv_file(path)

    def _load_csv_file(self, csv_file: Path) -> None:
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip() for h in next(reader, [])]

            col_map = {name: idx for idx, name in enumerate(header)}

            is_tracker_format = "X_min" in col_map and "Y_min" in col_map

            for row in reader:
                if not row or not any(row):
                    continue
                try:
                    if is_tracker_format:
                        f_idx = int(row[col_map["Frame"]])
                        t_id = int(row[col_map["Tracker ID"]])
                        xmin = float(row[col_map["X_min"]])
                        ymin = float(row[col_map["Y_min"]])
                        xmax = float(row[col_map["X_max"]])
                        ymax = float(row[col_map["Y_max"]])
                        if any(math.isnan(v) for v in (xmin, ymin, xmax, ymax)):
                            continue
                        conf = float(row[col_map["Confidence"]]) if "Confidence" in col_map and not math.isnan(float(row[col_map["Confidence"]])) else 1.0
                        label = row[col_map["Label"]] if "Label" in col_map else "object"
                        bbox = BBox(
                            x=xmin,
                            y=ymin,
                            w=max(0.0, xmax - xmin),
                            h=max(0.0, ymax - ymin),
                            id=t_id,
                            label=label,
                            conf=conf,
                        )
                        self.gt_by_frame.setdefault(f_idx, []).append(bbox)
                    else:
                        nums = [float(p) for p in row if p.strip()]
                        if len(nums) == 4:
                            self.gt_by_frame.setdefault(len(self.gt_by_frame), []).append(
                                BBox(nums[0], nums[1], nums[2], nums[3], id=1)
                            )
                        elif len(nums) >= 5:
                            f_idx = int(nums[0])
                            t_id = int(nums[5]) if len(nums) >= 6 else 1
                            self.gt_by_frame.setdefault(f_idx, []).append(
                                BBox(nums[1], nums[2], nums[3], nums[4], id=t_id)
                            )
                except (ValueError, IndexError):
                    continue

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def get_frame(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self._n_frames:
            raise IndexError(f"Frame {idx} fora do intervalo (0..{self._n_frames - 1})")

        if self._current_pos != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._current_pos = idx

        ret, frame = self.cap.read()
        self._current_pos += 1
        if not ret or frame is None:
            raise RuntimeError(f"Falha ao decodificar frame {idx} do vídeo {self.video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_gt(self, idx: int) -> List[BBox]:
        return self.gt_by_frame.get(idx, [])

    def __del__(self) -> None:
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
