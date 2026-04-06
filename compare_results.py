#!/usr/bin/env python3
"""Create comparison table/plot for baseline vs DCNv2 fisheye experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and DCNv2 experiment results")
    parser.add_argument("--output", type=Path, default=Path("runs/obb/comparison"), help="Output directory")
    parser.add_argument(
        "--dcn-map50",
        type=float,
        default=None,
        help="Optional DCNv2 fisheye mAP@50 override once available",
    )
    parser.add_argument(
        "--dcn-map5095",
        type=float,
        default=None,
        help="Optional DCNv2 fisheye mAP@50-95 override once available",
    )
    parser.add_argument("--dcn-precision", type=float, default=None, help="Optional DCNv2 precision override")
    parser.add_argument("--dcn-recall", type=float, default=None, help="Optional DCNv2 recall override")
    return parser.parse_args()


def build_table(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = [
        {
            "Model": "YOLO11n-OBB (baseline)",
            "Dataset": "Normal",
            "mAP@50": 0.937,
            "mAP@50-95": 0.890,
            "Precision": 0.931,
            "Recall": 0.889,
            "Params": "~2.7M",
            "GFLOPs": "~6.9",
        },
        {
            "Model": "YOLO11n-OBB (baseline)",
            "Dataset": "Fisheye",
            "mAP@50": 0.860,
            "mAP@50-95": 0.689,
            "Precision": 0.859,
            "Recall": 0.818,
            "Params": "~2.7M",
            "GFLOPs": "~6.9",
        },
        {
            "Model": "YOLO11n-OBB (baseline)",
            "Dataset": "Mixed",
            "mAP@50": 0.891,
            "mAP@50-95": 0.767,
            "Precision": 0.900,
            "Recall": 0.841,
            "Params": "~2.7M",
            "GFLOPs": "~6.9",
        },
        {
            "Model": "YOLO11n-OBB + DCNv2",
            "Dataset": "Fisheye",
            "mAP@50": args.dcn_map50,
            "mAP@50-95": args.dcn_map5095,
            "Precision": args.dcn_precision,
            "Recall": args.dcn_recall,
            "Params": "~2.9M",
            "GFLOPs": "~7.0",
        },
    ]
    return rows


def plot_map50(rows: list[dict[str, object]], output_dir: Path) -> None:
    plot_rows = [r for r in rows if str(r["Dataset"]) in {"Fisheye", "Normal", "Mixed"}]
    labels = [f"{r['Model']}\n{r['Dataset']}" for r in plot_rows]
    values = [float(r["mAP@50"]) if r["mAP@50"] is not None else 0.0 for r in plot_rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mAP@50")
    ax.set_title("License Plate OBB Results: Baseline vs DCNv2")
    ax.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "map50_comparison.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = build_table(args)
    csv_path = args.output / "comparison_table.csv"
    md_path = args.output / "comparison_table.md"

    headers = ["Model", "Dataset", "mAP@50", "mAP@50-95", "Precision", "Recall", "Params", "GFLOPs"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    plot_map50(rows, args.output)

    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {args.output / 'map50_comparison.png'}")


if __name__ == "__main__":
    main()
