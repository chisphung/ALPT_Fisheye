#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
ULTRALYTICS_SRC = ROOT / "ultralytics"
if ULTRALYTICS_SRC.exists():
    sys.path.insert(0, str(ULTRALYTICS_SRC))

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n-obb-dcn.yaml")
    # Load standard pretrained backbone weights where architectures match.
    # DCNv2 layers (5 & 7) will be randomly initialized — expected for ablation.
    model.load("yolo11n-obb.pt")

    model.train(
        data="dataset_fisheye/data.yaml",
        epochs=100,
        imgsz=1024,
        device=0,
        batch=32,
        deterministic=False,  # required: deterministic mode can segfault with DCNv2/DeformConv2d
        optimizer="AdamW",
        lr0=0.005,
        lrf=0.05,
        cos_lr=True,
        warmup_epochs=5,
        mosaic=0.5,
        close_mosaic=20,
        patience=30,
        save_period=10,    # save every 10 epochs + best/last (disk safeguard)
        cache=False,       # no image caching to disk
        name="train_dcn_fisheye",
    )


if __name__ == "__main__":
    main()
