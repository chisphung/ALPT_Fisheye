"""Train YOLOv11s-OBB on dataset_normal — 2nd run with tuned hyperparameters.

Changes from Run 1 (all defaults):
    - epochs: 50 -> 100       — model was not converged at epoch 50
    - imgsz: 640 -> 1024      — license plates are small; higher res improves dfl/geometry
    - lr0: default(0.01) -> 0.005   — lower initial LR to reduce oscillation in mAP/P/R
    - lrf: default(0.01) -> 0.05    — slower final LR decay (keep more LR near the end)
    - warmup_epochs: 3 -> 5         — longer warmup to stabilize early training
    - cos_lr: True                  — cosine LR schedule is smoother than linear decay
    - optimizer: default(SGD)->AdamW — better convergence on small/noisy datasets
    - batch: default(16) -> 32      — larger batch → smoother gradients → reduces oscillation
    - mosaic: 0.5                   — reduce mosaic augmentation (may confuse small plates)
    - close_mosaic: 20              — turn mosaic off earlier in training for stability
    - patience: 30                  — more patience for early stopping (15 by default)

Usage:
    python dataset_normal/yolo_training.py
"""

from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    # Resolve data.yaml next to this script so it works from any cwd.
    script_dir = Path(__file__).resolve().parent
    data_yaml = script_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    # YOLOv11 small model variant for oriented bounding box detection.
    model = YOLO("yolo11s-obb.pt")

    # --- 2nd Training: Tuned hyperparameters ---
    model.train(
        data=str(data_yaml),
        epochs=100,             # Run 1 wasn't converged at epoch 50
        imgsz=1024,             # Higher res → better detection for small license plates
        device=0,
        batch=32,               # Larger batch → smoother gradients, less oscillation
        optimizer="AdamW",      # AdamW converges better than SGD for small datasets
        lr0=0.005,              # Lower initial LR to reduce noisy mAP fluctuations
        lrf=0.05,               # Slower LR decay (5% of lr0 at end, vs 1% default)
        cos_lr=True,            # Cosine annealing is smoother than linear decay
        warmup_epochs=5,        # Longer warmup for more stable early training
        mosaic=0.5,             # Reduce mosaic since license plates are small/fine objects
        close_mosaic=20,        # Disable mosaic earlier for fine-tuning stability
        patience=30,            # Wait longer before early stopping
        project="runs/obb",
        name="train2",
    )


if __name__ == "__main__":
    main()
