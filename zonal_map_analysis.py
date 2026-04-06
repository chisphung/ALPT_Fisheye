#!/usr/bin/env python3
"""Phase 1: Zonal mAP analysis for fisheye OBB detection.

This script evaluates a trained OBB model and decomposes performance by radial zone:
- center:    0.00 <= r < 0.33
- middle:    0.33 <= r < 0.66
- periphery: 0.66 <= r <= 1.00

Radial distance r is measured from normalized label-space centroids using
r = distance((cx, cy), (0.5, 0.5)) / sqrt(0.5^2 + 0.5^2), clipped to [0, 1].

Outputs:
- runs/zonal_analysis/zone_metrics.csv
- runs/zonal_analysis/zone_metrics.json
- runs/zonal_analysis/zonal_map_summary.png
- runs/zonal_analysis/detection_vs_radius.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ULTRALYTICS_SRC = ROOT / "ultralytics"
if ULTRALYTICS_SRC.exists():
    sys.path.insert(0, str(ULTRALYTICS_SRC))

from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

ZONE_BINS = [0.0, 0.33, 0.66, 1.01]
ZONE_NAMES = ["center", "middle", "periphery"]
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


@dataclass
class ZoneAccumulator:
    conf: list[float]
    pred_cls: list[int]
    tp: list[np.ndarray]
    target_cls: list[int]
    gt_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zonal mAP analysis for fisheye OBB model")
    parser.add_argument("--model", type=Path, default=Path("best_fisheye.pt"), help="Path to trained model weights")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("dataset_fisheye/images/val"),
        help="Validation images directory",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("dataset_fisheye/labels/val"),
        help="Validation labels directory",
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Inference confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="Inference NMS IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default="0" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/zonal_analysis"),
        help="Output directory for metrics/plots",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap for number of images to process (0 means all)",
    )
    return parser.parse_args()


def resolve_device(requested_device: str) -> str:
    """Resolve a user-requested device string into one Ultralytics can use safely."""
    device = str(requested_device).strip().lower()
    if device == "cpu":
        return "cpu"

    cuda_requested = device in {"cuda", "cuda:0"} or any(ch.isdigit() for ch in device)
    if cuda_requested and not torch.cuda.is_available():
        print(f"CUDA is unavailable in this environment; falling back from device={requested_device!r} to 'cpu'.")
        return "cpu"

    return requested_device


def zone_index_for_radius(radius: float) -> int:
    for i in range(len(ZONE_NAMES)):
        if ZONE_BINS[i] <= radius < ZONE_BINS[i + 1]:
            return i
    return len(ZONE_NAMES) - 1


def load_gt_label_file(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not label_path.exists():
        return np.zeros((0, 4, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    polys = []
    classes = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cls_id = int(float(parts[0]))
            coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 2)
            polys.append(coords)
            classes.append(cls_id)

    if not polys:
        return np.zeros((0, 4, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(polys), np.array(classes, dtype=np.int64)


def radial_distance_from_centroid(poly: np.ndarray) -> float:
    centroid = poly.mean(axis=0)
    dist = float(np.linalg.norm(centroid - np.array([0.5, 0.5], dtype=np.float32)))
    max_dist = math.sqrt(0.5**2 + 0.5**2)
    return float(np.clip(dist / max_dist, 0.0, 1.0))


def greedy_match_tp(
    pred_cls: np.ndarray,
    gt_cls: np.ndarray,
    iou_mat: np.ndarray,
    iou_thresholds: np.ndarray,
) -> np.ndarray:
    """Return TP matrix of shape (num_pred, num_iou_thresholds)."""
    num_pred = pred_cls.shape[0]
    num_iou = iou_thresholds.shape[0]
    tp = np.zeros((num_pred, num_iou), dtype=bool)

    if num_pred == 0 or gt_cls.shape[0] == 0:
        return tp

    class_match = gt_cls[:, None] == pred_cls[None, :]

    for i, thr in enumerate(iou_thresholds):
        candidates = np.argwhere((iou_mat >= thr) & class_match)
        if candidates.size == 0:
            continue

        # Sort all candidate (gt, pred) pairs by IoU descending, then greedily assign one-to-one.
        scores = iou_mat[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(-scores)
        used_gt = set()
        used_pred = set()
        for idx in order:
            g, p = int(candidates[idx, 0]), int(candidates[idx, 1])
            if g in used_gt or p in used_pred:
                continue
            used_gt.add(g)
            used_pred.add(p)
            tp[p, i] = True

    return tp


def collect_predictions(
    model: YOLO,
    image_paths: list[Path],
    labels_dir: Path,
    imgsz: int,
    conf_thres: float,
    iou_thres: float,
    device: str,
) -> tuple[dict[str, ZoneAccumulator], list[float], list[float]]:
    zones = {
        name: ZoneAccumulator(conf=[], pred_cls=[], tp=[], target_cls=[])
        for name in ZONE_NAMES
    }

    scatter_radii: list[float] = []
    scatter_conf: list[float] = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        gt_polys, gt_cls = load_gt_label_file(label_path)

        gt_zone_idx = np.array([zone_index_for_radius(radial_distance_from_centroid(p)) for p in gt_polys], dtype=np.int64)
        gt_radii = np.array([radial_distance_from_centroid(p) for p in gt_polys], dtype=np.float32)

        for z in gt_zone_idx:
            zone = zones[ZONE_NAMES[z]]
            zone.gt_count += 1

        result = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf_thres,
            iou=iou_thres,
            device=device,
            verbose=False,
        )[0]

        if result.obb is None or len(result.obb) == 0:
            continue

        pred_cls = result.obb.cls.detach().cpu().numpy().astype(np.int64)
        pred_conf = result.obb.conf.detach().cpu().numpy().astype(np.float32)
        pred_poly = result.obb.xyxyxyxy.detach().cpu().numpy().astype(np.float32)
        if pred_poly.ndim == 2 and pred_poly.shape[1] == 8:
            pred_poly = pred_poly.reshape(-1, 4, 2)

        # Normalize prediction polygons to label space [0, 1] for fair matching.
        h, w = result.orig_shape
        pred_poly_norm = pred_poly.copy()
        pred_poly_norm[..., 0] /= float(w)
        pred_poly_norm[..., 1] /= float(h)

        pred_radii = np.array([radial_distance_from_centroid(p) for p in pred_poly_norm], dtype=np.float32)
        pred_zone_idx = np.array([zone_index_for_radius(r) for r in pred_radii], dtype=np.int64)

        scatter_radii.extend(pred_radii.tolist())
        scatter_conf.extend(pred_conf.tolist())

        if len(gt_polys):
            gt_xywhr = xyxyxyxy2xywhr(torch.from_numpy(gt_polys.reshape(-1, 8))).float()
            pred_xywhr = xyxyxyxy2xywhr(torch.from_numpy(pred_poly_norm.reshape(-1, 8))).float()
            iou_mat = batch_probiou(gt_xywhr, pred_xywhr).cpu().numpy()
            tp_matrix = greedy_match_tp(pred_cls, gt_cls, iou_mat, IOU_THRESHOLDS)
        else:
            tp_matrix = np.zeros((pred_cls.shape[0], len(IOU_THRESHOLDS)), dtype=bool)

        for zi, zone_name in enumerate(ZONE_NAMES):
            pred_mask = pred_zone_idx == zi
            gt_mask = gt_zone_idx == zi if len(gt_zone_idx) else np.zeros((0,), dtype=bool)

            zone = zones[zone_name]
            if pred_mask.any():
                zone.conf.extend(pred_conf[pred_mask].tolist())
                zone.pred_cls.extend(pred_cls[pred_mask].tolist())
                zone.tp.extend(tp_matrix[pred_mask])

            if gt_mask.any():
                # Class list for AP denominator inside this zone.
                zone.target_cls.extend(gt_cls[gt_mask].tolist())

    return zones, scatter_radii, scatter_conf


def compute_zone_metrics(zones: dict[str, ZoneAccumulator]) -> list[dict]:
    rows = []
    for name in ZONE_NAMES:
        acc = zones[name]

        if acc.tp and acc.target_cls:
            tp = np.array(acc.tp, dtype=bool)
            conf = np.array(acc.conf, dtype=np.float32)
            pred_cls = np.array(acc.pred_cls, dtype=np.int64)
            target_cls = np.array(acc.target_cls, dtype=np.int64)

            tp_out, fp_out, p_out, r_out, _, ap, _, *_ = ap_per_class(
                tp=tp,
                conf=conf,
                pred_cls=pred_cls,
                target_cls=target_cls,
                plot=False,
                names={0: "plate"},
            )

            precision = float(p_out.mean()) if len(p_out) else 0.0
            recall = float(r_out.mean()) if len(r_out) else 0.0
            map50 = float(ap[:, 0].mean()) if ap.size else 0.0
            map5095 = float(ap.mean()) if ap.size else 0.0
            tp50 = int(tp_out.sum()) if tp_out.size else 0
            fp50 = int(fp_out.sum()) if fp_out.size else 0
        else:
            precision = 0.0
            recall = 0.0
            map50 = 0.0
            map5095 = 0.0
            tp50 = 0
            fp50 = 0

        gt_count = acc.gt_count
        fn50 = max(gt_count - tp50, 0)

        rows.append(
            {
                "zone": name,
                "gt_count": gt_count,
                "tp50": tp50,
                "fp50": fp50,
                "fn50": fn50,
                "precision": precision,
                "recall": recall,
                "map50": map50,
                "map50_95": map5095,
            }
        )
    return rows


def save_metrics(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "zone_metrics.csv"
    json_path = output_dir / "zone_metrics.json"

    header = [
        "zone",
        "gt_count",
        "tp50",
        "fp50",
        "fn50",
        "precision",
        "recall",
        "map50",
        "map50_95",
    ]

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(
                ",".join(
                    [
                        str(row["zone"]),
                        str(row["gt_count"]),
                        str(row["tp50"]),
                        str(row["fp50"]),
                        str(row["fn50"]),
                        f"{row['precision']:.6f}",
                        f"{row['recall']:.6f}",
                        f"{row['map50']:.6f}",
                        f"{row['map50_95']:.6f}",
                    ]
                )
                + "\n"
            )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def render_summary_figure(rows: list[dict], output_dir: Path) -> None:
    zones = [r["zone"] for r in rows]
    map50 = [r["map50"] for r in rows]
    map5095 = [r["map50_95"] for r in rows]
    gt_count = [r["gt_count"] for r in rows]

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    x = np.arange(len(zones))
    width = 0.35
    ax1.bar(x - width / 2, map50, width, label="mAP@50")
    ax1.bar(x + width / 2, map5095, width, label="mAP@50-95")
    ax1.set_xticks(x)
    ax1.set_xticklabels(zones)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Score")
    ax1.set_title("Per-zone Detection Quality")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(zones), endpoint=False)
    widths = np.full_like(angles, 2 * np.pi / len(zones))
    radii = np.array(gt_count, dtype=np.float32)
    if radii.max() > 0:
        radii = radii / radii.max()
    bars = ax2.bar(angles, radii, width=widths, bottom=0.0, alpha=0.7)
    for b, z in zip(bars, zones):
        b.set_label(z)
    ax2.set_title("Relative GT Distribution by Zone")
    ax2.set_yticklabels([])

    fig.tight_layout()
    fig.savefig(output_dir / "zonal_map_summary.png", dpi=180)
    plt.close(fig)


def render_confidence_radius_plot(all_radii: list[float], all_conf: list[float], output_dir: Path) -> None:
    if not all_radii or not all_conf:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(all_radii, all_conf, s=8, alpha=0.35)
    ax.set_xlabel("Radial distance (normalized)")
    ax.set_ylabel("Detection confidence")
    ax.set_title("Predicted Confidence vs Radial Distance")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "detection_vs_radius.png", dpi=180)
    plt.close(fig)


def print_table(rows: list[dict]) -> None:
    print("\nZonal metrics summary")
    print("=" * 95)
    print(
        f"{'Zone':<10}{'GT':>8}{'TP@50':>10}{'FP@50':>10}{'FN@50':>10}{'Prec':>10}{'Recall':>10}{'mAP50':>10}{'mAP50-95':>12}"
    )
    print("-" * 95)
    for r in rows:
        print(
            f"{r['zone']:<10}{r['gt_count']:>8}{r['tp50']:>10}{r['fp50']:>10}{r['fn50']:>10}"
            f"{r['precision']:>10.4f}{r['recall']:>10.4f}{r['map50']:>10.4f}{r['map50_95']:>12.4f}"
        )


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)

    image_paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in args.images.glob(ext)
    )
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.images}")

    model = YOLO(str(args.model))
    zones, all_radii, all_conf = collect_predictions(
        model=model,
        image_paths=image_paths,
        labels_dir=args.labels,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device,
    )

    rows = compute_zone_metrics(zones)
    save_metrics(rows, args.output)
    render_summary_figure(rows, args.output)
    render_confidence_radius_plot(all_radii, all_conf, args.output)
    print_table(rows)

    total_gt = sum(r["gt_count"] for r in rows)
    print(f"\nTotal GT objects across zones: {total_gt}")
    print(f"Saved outputs to: {args.output}")


if __name__ == "__main__":
    main()
