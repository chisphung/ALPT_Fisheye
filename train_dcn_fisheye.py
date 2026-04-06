#!/usr/bin/env python3
"""Train YOLO11n-OBB-DCN on fisheye dataset (Phase 2 ablation).

DCN Ablation: Same hyperparameters as the best fisheye baseline run,
but with DCNv2 replacing Conv at backbone stages 3-4 (layers 5 and 7).

Notes:
- amp=False: The AMP check (check_amp) in Ultralytics loads a secondary YOLO
  model ("yolo26n.pt") and runs a forward pass alongside the custom model to
  compare FP32 vs FP16 outputs. This secondary model launch segfaults in some
  CUDA/torchvision.ops.DeformConv2d environments. Disabling AMP bypasses this
  entirely; training still converges correctly in FP32.
  Re-enable with amp=True once the segfault root cause is fixed.
- batch=8: Fixed batch to avoid auto-batch probe crash (batch=-1 also triggers
  a CUDA forward-pass probe that segfaults with DeformConv2d).
- workers=4: Reduced to avoid multiprocessing-related segfaults.
- save_period=10: Keep only periodic + best/last checkpoints to protect disk
  (current usage ~98%, ~13 GB free).
- cache=False: Avoids caching images to disk.
"""

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
        project="runs/obb",
        name="train_dcn_fisheye",
    )


if __name__ == "__main__":
    main()
