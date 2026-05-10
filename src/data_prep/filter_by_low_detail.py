"""Data prep: low-detail filtering with reconstruction-change scoring.

This script targets aligned datasets in the format:
  <dataset_root>/aligned_112/{train,dev,test}/...

Usage:
    python -m src.data_prep.filter_by_low_detail \
        --input-dataset-dir data/upload_ready/people_gator_full_filtered_drop05_facedet \
        --output-dataset-dir data/upload_ready/people_gator_full_filtered_drop05_facedet_lowdetail \
        --drop-rate 0.03 \
        --splits train dev test
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.data_prep.common_dataset_ops import (
    collect_images,
    copy_dropped_images,
    copy_input_metadata,
    copy_kept_images,
    copy_passthrough_files,
    filter_corresponding_faces,
)


def _reconstruction_change_metrics(img_path: Path, downscale_factor: float) -> dict | None:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    h, w = img.shape
    down_w = max(1, int(round(w * downscale_factor)))
    down_h = max(1, int(round(h * downscale_factor)))

    down = cv2.resize(img, (down_w, down_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

    # Mean absolute pixel difference normalized to [0, 1].
    diff = cv2.absdiff(img, up)
    reconstruction_change = float(diff.mean() / 255.0)
    return {
        "reconstruction_change_pct": reconstruction_change,
    }


def _rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    rank = np.empty_like(order, dtype=np.float32)
    rank[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return rank


def main():
    parser = argparse.ArgumentParser(description="Filter aligned dataset by low-detail score")
    parser.add_argument("--input-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dataset-dir", type=Path, required=True)
    parser.add_argument("--drop-rate", type=float, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--copy-dropped", action="store_true")
    parser.add_argument(
        "--downscale-factor",
        type=float,
        default=0.5,
        help="Downscale factor for reconstruction-change scoring; must be in (0, 1).",
    )
    args = parser.parse_args()

    if args.drop_rate < 0.0 or args.drop_rate >= 1.0:
        raise ValueError("--drop-rate must be in [0, 1)")
    if args.downscale_factor <= 0.0 or args.downscale_factor >= 1.0:
        raise ValueError("--downscale-factor must be in (0, 1)")

    aligned_in = args.input_dataset_dir / "aligned_112"
    if not aligned_in.exists():
        raise FileNotFoundError(f"Missing aligned_112 directory: {aligned_in}")

    if args.output_dataset_dir.exists():
        shutil.rmtree(args.output_dataset_dir)
    args.output_dataset_dir.mkdir(parents=True, exist_ok=True)
    aligned_out = args.output_dataset_dir / "aligned_112"
    aligned_out.mkdir(parents=True, exist_ok=True)

    rows = collect_images(aligned_in, args.splits)
    if not rows:
        raise RuntimeError("No images found for requested splits")

    scored = []
    for row in tqdm(rows, desc="Low-detail scoring"):
        m = _reconstruction_change_metrics(row["abs_path"], args.downscale_factor)
        if m is None:
            continue
        row.update(m)
        scored.append(row)

    change = np.array([r["reconstruction_change_pct"] for r in scored], dtype=np.float32)
    # Lower reconstruction change implies lower detail.
    low_detail = 1.0 - _rank01(change)
    for i, row in enumerate(scored):
        row["low_detail_score"] = float(low_detail[i])

    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        by_split[row["split"]].append(row)

    kept_by_split: dict[str, set[str]] = {s: set() for s in args.splits}
    dropped_by_split: dict[str, set[str]] = {s: set() for s in args.splits}
    split_summary = {}
    for split, split_rows in by_split.items():
        split_rows.sort(key=lambda r: (r["low_detail_score"], r["rel_path"]), reverse=True)
        drop_count = int(len(split_rows) * args.drop_rate)
        dropped = split_rows[:drop_count]
        kept = split_rows[drop_count:]
        kept_by_split[split] = {r["rel_path"] for r in kept}
        dropped_by_split[split] = {r["rel_path"] for r in dropped}
        split_summary[split] = {
            "input": len(split_rows),
            "drop_count": len(dropped),
            "keep_count": len(kept),
            "mean_low_detail_score_dropped": float(np.mean([r["low_detail_score"] for r in dropped])) if dropped else 0.0,
            "mean_low_detail_score_kept": float(np.mean([r["low_detail_score"] for r in kept])) if kept else 0.0,
        }
        split_summary[split]["mean_reconstruction_change_pct_dropped"] = (
            float(np.mean([r["reconstruction_change_pct"] for r in dropped])) if dropped else 0.0
        )
        split_summary[split]["mean_reconstruction_change_pct_kept"] = (
            float(np.mean([r["reconstruction_change_pct"] for r in kept])) if kept else 0.0
        )

    copied, missing = copy_kept_images(aligned_in, aligned_out, kept_by_split)

    if args.copy_dropped:
        copy_dropped_images(aligned_in, args.output_dataset_dir, dropped_by_split)

    filtered_annotations = filter_corresponding_faces(args.input_dataset_dir, args.output_dataset_dir, kept_by_split)
    copy_passthrough_files(args.input_dataset_dir, args.output_dataset_dir)

    metadata_out = args.output_dataset_dir / "_metadata"
    metadata_out.mkdir(parents=True, exist_ok=True)
    copy_input_metadata(args.input_dataset_dir, metadata_out)

    with (metadata_out / "low_detail_scores.jsonl").open("w") as f:
        for row in scored:
            out_row = {
                "split": row["split"],
                "rel_path": row["rel_path"],
                "low_detail_score": row["low_detail_score"],
                "method": "reconstruction",
                "reconstruction_change_pct": row["reconstruction_change_pct"],
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    summary = {
        "input_dataset_dir": str(args.input_dataset_dir),
        "output_dataset_dir": str(args.output_dataset_dir),
        "drop_rate": args.drop_rate,
        "method": "reconstruction",
        "downscale_factor": args.downscale_factor,
        "splits": args.splits,
        "images_copied": copied,
        "missing_from_source": missing,
        "split_summary": split_summary,
        "filtered_annotation_counts": filtered_annotations,
    }
    (metadata_out / "low_detail_filter_summary.json").write_text(json.dumps(summary, indent=2))

    print("Done.")
    for split, s in split_summary.items():
        print(f"  {split}: keep={s['keep_count']}/{s['input']} (dropped={s['drop_count']})")
    print(f"  Output: {args.output_dataset_dir}")


if __name__ == "__main__":
    main()
