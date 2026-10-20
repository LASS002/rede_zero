"""Testes dos leitores de dataset (SCR-01)."""

import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from rede_zero.data.readers import BBox, OTBReader, VideoReader


def test_bbox_properties():
    box = BBox(x=10.0, y=20.0, w=30.0, h=40.0, id=1, label="car", conf=0.9)
    assert box.x1 == 10.0
    assert box.y1 == 20.0
    assert box.x2 == 40.0
    assert box.y2 == 60.0
    assert box.center == (25.0, 40.0)
    assert box.to_tlwh() == (10.0, 20.0, 30.0, 40.0)
    assert box.to_xyxy() == (10.0, 20.0, 40.0, 60.0)


def test_otb_reader_synthetic(tmp_path: Path):
    seq_dir = tmp_path / "seq1"
    images_dir = seq_dir / "images"
    images_dir.mkdir(parents=True)

    img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "frame_000000.jpg"), img1)
    cv2.imwrite(str(images_dir / "frame_000001.jpg"), img2)

    gt_file = seq_dir / "gt.txt"
    gt_file.write_text("10,10,20,20\n12,12,20,20\n", encoding="utf-8")

    reader = OTBReader(seq_dir)
    assert reader.n_frames == 2
    assert len(reader) == 2

    frame0 = reader.get_frame(0)
    assert frame0.shape == (100, 100, 3)
    assert frame0[0, 0, 0] == 50

    gt0 = reader.get_gt(0)
    assert len(gt0) == 1
    assert gt0[0].to_tlwh() == (10.0, 10.0, 20.0, 20.0)

    gt1 = reader.get_gt(1)
    assert len(gt1) == 1
    assert gt1[0].to_tlwh() == (12.0, 12.0, 20.0, 20.0)


def test_otb_reader_json_labels(tmp_path: Path):
    seq_dir = tmp_path / "seq_json"
    images_dir = seq_dir / "images"
    labels_dir = seq_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "frame_000000.jpg"), img)

    json_content = {
        "image": "frame_000000.jpg",
        "width": 100,
        "height": 100,
        "annotations": [{"x": 15, "y": 25, "w": 30, "h": 40, "label": "object"}],
    }
    (labels_dir / "frame_000000.json").write_text(json.dumps(json_content), encoding="utf-8")

    reader = OTBReader(seq_dir)
    assert reader.n_frames == 1
    gt = reader.get_gt(0)
    assert len(gt) == 1
    assert gt[0].x == 15.0
    assert gt[0].y == 25.0
    assert gt[0].w == 30.0
    assert gt[0].h == 40.0


def test_video_reader_synthetic(tmp_path: Path):
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (120, 90))
    for color in (50, 150):
        frame = np.full((90, 120, 3), color, dtype=np.uint8)
        out.write(frame)
    out.release()

    csv_path = tmp_path / "object_id_01.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Tracker ID", "Label", "X_min", "Y_min", "X_max", "Y_max", "Confidence"])
        writer.writerow([0, 1, "object", 10.0, 20.0, 30.0, 45.0, 0.95])
        writer.writerow([1, 1, "object", 12.0, 22.0, 32.0, 47.0, 0.94])

    reader = VideoReader(video_path, csv_path)
    assert reader.n_frames == 2
    f0 = reader.get_frame(0)
    assert f0.shape == (90, 120, 3)

    gt0 = reader.get_gt(0)
    assert len(gt0) == 1
    assert gt0[0].id == 1
    assert gt0[0].x == 10.0
    assert gt0[0].y == 20.0
    assert gt0[0].w == 20.0
    assert gt0[0].h == 25.0
