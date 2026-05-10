"""Data prep: face-confidence filtering with OpenCV Haar detector.

This script is intended for already prepared/aligned datasets that follow:
  <dataset_root>/aligned_112/{train,dev,test}/...
and optionally contain corresponding_faces*.jsonl annotation files.

Usage:
    python -m src.data_prep.filter_by_face_detector \
        --input-dataset-dir data/upload_ready/people_gator_full_filtered_drop05 \
        --output-dataset-dir data/upload_ready/people_gator_full_filtered_drop05_facedet \
        --drop-rate 0.05 \
        --splits train dev test
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from src.data_prep.common_dataset_ops import (
    collect_images,
    copy_dropped_images,
    copy_input_metadata,
    copy_kept_images,
    copy_passthrough_files,
    filter_corresponding_faces,
)


def _detect_confidence(
    img_path: Path,
    detector: cv2.CascadeClassifier,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
) -> tuple[float, int]:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0, 0

    # detectMultiScale3 returns level weights we can use as confidence.
    rects, _, level_weights = detector.detectMultiScale3(
        img,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        outputRejectLevels=True,
    )
    if len(rects) == 0:
        return 0.0, 0
    return float(max(level_weights)), int(len(rects))


def main():
    parser = argparse.ArgumentParser(description="Filter aligned dataset by face detector confidence")
    parser.add_argument("--input-dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dataset-dir", type=Path, required=True)
    parser.add_argument("--drop-rate", type=float, required=True,
                        help="Drop this fraction of lowest-confidence images in each split, e.g. 0.05")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--scale-factor", type=float, default=1.2)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=28)
    parser.add_argument("--copy-dropped", action="store_true",
                        help="Copy dropped images under _dropped/{split}/ for inspection")
    args = parser.parse_args()

    if args.drop_rate < 0.0 or args.drop_rate >= 1.0:
        raise ValueError("--drop-rate must be in [0, 1)")

    aligned_in = args.input_dataset_dir / "aligned_112"
    if not aligned_in.exists():
        raise FileNotFoundError(f"Missing aligned_112 directory: {aligned_in}")

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")

    if args.output_dataset_dir.exists():
        shutil.rmtree(args.output_dataset_dir)
    args.output_dataset_dir.mkdir(parents=True, exist_ok=True)
    aligned_out = args.output_dataset_dir / "aligned_112"
    aligned_out.mkdir(parents=True, exist_ok=True)

    rows = collect_images(aligned_in, args.splits)
    if not rows:
        raise RuntimeError("No images found for requested splits")

    for row in tqdm(rows, desc="Face confidence"):
        score, num_faces = _detect_confidence(
            row["abs_path"],
            detector,
            args.scale_factor,
            args.min_neighbors,
            args.min_size,
        )
        row["face_confidence"] = score
        row["faces_detected"] = num_faces

    by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    kept_by_split: dict[str, set[str]] = {s: set() for s in args.splits}
    dropped_by_split: dict[str, set[str]] = {s: set() for s in args.splits}
    split_summary = {}

    for split, split_rows in by_split.items():
        split_rows.sort(key=lambda r: (r["face_confidence"], r["rel_path"]))
        drop_count = int(len(split_rows) * args.drop_rate)
        dropped = split_rows[:drop_count]
        kept = split_rows[drop_count:]
        kept_by_split[split] = {r["rel_path"] for r in kept}
        dropped_by_split[split] = {r["rel_path"] for r in dropped}

        split_summary[split] = {
            "input": len(split_rows),
            "drop_count": len(dropped),
            "keep_count": len(kept),
            "no_face_input": sum(1 for r in split_rows if r["faces_detected"] == 0),
            "no_face_kept": sum(1 for r in kept if r["faces_detected"] == 0),
            "no_face_dropped": sum(1 for r in dropped if r["faces_detected"] == 0),
        }

    copied, missing = copy_kept_images(aligned_in, aligned_out, kept_by_split)

    if args.copy_dropped:
        copy_dropped_images(aligned_in, args.output_dataset_dir, dropped_by_split)

    filtered_annotations = filter_corresponding_faces(args.input_dataset_dir, args.output_dataset_dir, kept_by_split)
    copy_passthrough_files(args.input_dataset_dir, args.output_dataset_dir)

    metadata_out = args.output_dataset_dir / "_metadata"
    metadata_out.mkdir(parents=True, exist_ok=True)
    copy_input_metadata(args.input_dataset_dir, metadata_out)

    with (metadata_out / "face_detector_scores.jsonl").open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "split": row["split"],
                        "rel_path": row["rel_path"],
                        "face_confidence": row["face_confidence"],
                        "faces_detected": row["faces_detected"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "input_dataset_dir": str(args.input_dataset_dir),
        "output_dataset_dir": str(args.output_dataset_dir),
        "drop_rate": args.drop_rate,
        "splits": args.splits,
        "cascade_path": str(cascade_path),
        "scale_factor": args.scale_factor,
        "min_neighbors": args.min_neighbors,
        "min_size": args.min_size,
        "images_copied": copied,
        "missing_from_source": missing,
        "split_summary": split_summary,
        "filtered_annotation_counts": filtered_annotations,
        "total_faces_detected_distribution": dict(Counter(r["faces_detected"] for r in rows)),
    }
    (metadata_out / "face_detector_filter_summary.json").write_text(json.dumps(summary, indent=2))

    print("Done.")
    for split, s in split_summary.items():
        print(
            f"  {split}: keep={s['keep_count']}/{s['input']} "
            f"(dropped={s['drop_count']}), no-face kept={s['no_face_kept']}, dropped={s['no_face_dropped']}"
        )
    print(f"  Output: {args.output_dataset_dir}")


if __name__ == "__main__":
    main()
