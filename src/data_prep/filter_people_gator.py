"""Data prep: quality filtering for people_gator with diversity guardrails.

Usage:
    python -m src.data_prep.filter_people_gator \
        --aligned-dir data/people_gator/aligned_224x224 \
        --output-dir data/people_gator/filter_run_224 \
        --splits train \
        --drop-ratio-per-library 0.15 \
        --min-per-identity 1
"""

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.data_prep.common_dataset_ops import collect_images, write_jsonl


def _percentile_norm(values: np.ndarray, low_q: float = 5.0, high_q: float = 95.0) -> np.ndarray:
    low = float(np.percentile(values, low_q))
    high = float(np.percentile(values, high_q))
    if high <= low:
        return np.ones_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _compute_metrics(path: Path) -> dict | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    clip_low = float(np.mean(gray <= 5))
    clip_high = float(np.mean(gray >= 250))
    return {
        "width": int(w),
        "height": int(h),
        "pixels": int(w * h),
        "file_size_bytes": int(path.stat().st_size),
        "sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "contrast_std": float(np.std(gray)),
        "clip_fraction": clip_low + clip_high,
    }


def _identity_from_rel(rel_path: Path) -> str:
    parent = rel_path.parent
    return str(parent) if str(parent) != "." else "root"


def _select_kept(
    rows: list[dict],
    drop_ratio_per_library: float,
    min_per_identity: int,
    max_per_identity: int | None,
) -> set[str]:
    kept_paths: set[str] = set()
    by_library: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_library[r["library"]].append(r)

    for library, lib_rows in by_library.items():
        lib_rows.sort(key=lambda x: x["quality_score"], reverse=True)
        by_identity: dict[str, list[dict]] = defaultdict(list)
        for row in lib_rows:
            by_identity[row["identity"]].append(row)

        # Keep a minimum number per identity so low-frequency identities are not dropped.
        identity_kept_counts: Counter[str] = Counter()
        lib_kept_count = 0
        for identity, id_rows in by_identity.items():
            id_rows.sort(key=lambda x: x["quality_score"], reverse=True)
            minimum = min(min_per_identity, len(id_rows))
            if max_per_identity is not None:
                minimum = min(minimum, max_per_identity)
            for row in id_rows[:minimum]:
                kept_paths.add(row["rel_path"])
                identity_kept_counts[identity] += 1
                lib_kept_count += 1

        target_keep = max(
            lib_kept_count,
            math.ceil((1.0 - drop_ratio_per_library) * len(lib_rows)),
        )

        for row in lib_rows:
            if lib_kept_count >= target_keep:
                break
            if row["rel_path"] in kept_paths:
                continue
            if max_per_identity is not None and identity_kept_counts[row["identity"]] >= max_per_identity:
                continue
            kept_paths.add(row["rel_path"])
            identity_kept_counts[row["identity"]] += 1
            lib_kept_count += 1

    return kept_paths


def _copy_selected(
    aligned_dir: Path,
    output_dir: Path,
    split: str,
    kept_rows: list[dict],
    rejected_rows: list[dict],
):
    kept_dir = output_dir / "filtered" / split / "kept"
    rejected_dir = output_dir / "filtered" / split / "rejected"
    for row in kept_rows:
        src = aligned_dir / split / row["rel_path"]
        dst = kept_dir / row["rel_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for row in rejected_rows:
        src = aligned_dir / split / row["rel_path"]
        dst = rejected_dir / row["rel_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _process_split(
    aligned_dir: Path,
    split: str,
    drop_ratio_per_library: float,
    min_per_identity: int,
    max_per_identity: int | None,
):
    split_dir = aligned_dir / split
    if not split_dir.exists():
        print(f"[{split}] not found: {split_dir}")
        return [], [], {}

    rows = []
    split_rows = collect_images(aligned_dir, [split])
    for entry in tqdm(split_rows, desc=f"Scoring {split}"):
        rel = Path(entry["rel_path"])
        metrics = _compute_metrics(entry["abs_path"])
        if metrics is None:
            continue
        rel_parts = rel.parts
        rows.append(
            {
                "split": split,
                "rel_path": str(rel),
                "library": rel_parts[0] if rel_parts else "unknown",
                "identity": _identity_from_rel(rel),
                **metrics,
            }
        )

    if not rows:
        return [], [], {}

    sharp = np.array([r["sharpness_laplacian_var"] for r in rows], dtype=np.float32)
    contrast = np.array([r["contrast_std"] for r in rows], dtype=np.float32)
    clip_frac = np.array([r["clip_fraction"] for r in rows], dtype=np.float32)
    sharp_norm = _percentile_norm(sharp)
    contrast_norm = _percentile_norm(contrast)
    exposure_norm = 1.0 - np.clip(clip_frac, 0.0, 1.0)

    for i, row in enumerate(rows):
        score = 0.50 * float(sharp_norm[i]) + 0.30 * float(contrast_norm[i]) + 0.20 * float(exposure_norm[i])
        row["quality_score"] = score

    kept_paths = _select_kept(rows, drop_ratio_per_library, min_per_identity, max_per_identity)
    kept_rows = [r for r in rows if r["rel_path"] in kept_paths]
    rejected_rows = [r for r in rows if r["rel_path"] not in kept_paths]

    report = {
        "split": split,
        "input_images": len(rows),
        "kept_images": len(kept_rows),
        "rejected_images": len(rejected_rows),
        "retention_ratio": float(len(kept_rows) / max(len(rows), 1)),
        "libraries_before": dict(Counter(r["library"] for r in rows)),
        "libraries_after": dict(Counter(r["library"] for r in kept_rows)),
    }
    return kept_rows, rejected_rows, report


def main():
    parser = argparse.ArgumentParser(description="Filter people_gator with quality + diversity rules")
    parser.add_argument("--aligned-dir", type=Path, required=True,
                        help="Path to aligned split directory (e.g. data/people_gator/aligned_224x224)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for manifests and report")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        help="Splits to process (default: train)")
    parser.add_argument("--drop-ratio-per-library", type=float, default=0.15,
                        help="Fraction to reject in each library bucket (default: 0.15)")
    parser.add_argument("--min-per-identity", type=int, default=1,
                        help="Minimum kept samples per identity in each library (default: 1)")
    parser.add_argument("--max-per-identity", type=int, default=0,
                        help="Optional cap of kept samples per identity, 0 = disabled")
    parser.add_argument("--copy-selected", action="store_true",
                        help="Copy kept/rejected files under output_dir/filtered/")
    args = parser.parse_args()

    if args.drop_ratio_per_library < 0.0 or args.drop_ratio_per_library >= 1.0:
        raise ValueError("--drop-ratio-per-library must be in [0, 1)")
    if args.min_per_identity <= 0:
        raise ValueError("--min-per-identity must be >= 1")
    max_per_identity = args.max_per_identity if args.max_per_identity > 0 else None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_reports = []
    all_kept, all_rejected = [], []

    for split in args.splits:
        kept_rows, rejected_rows, report = _process_split(
            args.aligned_dir,
            split,
            args.drop_ratio_per_library,
            args.min_per_identity,
            max_per_identity,
        )
        if report:
            split_reports.append(report)
            all_kept.extend(kept_rows)
            all_rejected.extend(rejected_rows)
            if args.copy_selected:
                _copy_selected(args.aligned_dir, args.output_dir, split, kept_rows, rejected_rows)

    write_jsonl(args.output_dir / "kept_manifest.jsonl", all_kept)
    write_jsonl(args.output_dir / "rejected_manifest.jsonl", all_rejected)

    summary = {
        "aligned_dir": str(args.aligned_dir),
        "splits": args.splits,
        "drop_ratio_per_library": args.drop_ratio_per_library,
        "min_per_identity": args.min_per_identity,
        "max_per_identity": max_per_identity,
        "split_reports": split_reports,
        "total_input": sum(r["input_images"] for r in split_reports),
        "total_kept": len(all_kept),
        "total_rejected": len(all_rejected),
    }
    with (args.output_dir / "filter_report.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. kept={len(all_kept)}, rejected={len(all_rejected)}")
    print(f"  Kept manifest: {args.output_dir / 'kept_manifest.jsonl'}")
    print(f"  Rejected manifest: {args.output_dir / 'rejected_manifest.jsonl'}")
    print(f"  Report: {args.output_dir / 'filter_report.json'}")


if __name__ == "__main__":
    main()
