"""Data prep: extract, align, resize, and split people_gator face crops.

Usage:
    python -m src.data_prep.preprocess_people_gator \
        --zip-path /path/to/people_gator__data_export.zip \
        --output-dir data/people_gator
"""

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ARCFACE_KEYPOINTS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
DEFAULT_TARGET_SIZE = (112, 112)
CORRESPONDING_FACES_PREFIX = "people_gator__corresponding_faces__2026-02-11"
ALIGNED_CROPS_SUFFIX = ".peoplegator_aligned_crops"
DATA_PREFIX = "people_gator__data/"


def resize_image_bytes(image_bytes: bytes, target_size: tuple[int, int]) -> np.ndarray:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)


def realign_from_page(
    page_bytes: bytes, keypoints: list[list[float]], target_size: tuple[int, int]
) -> np.ndarray | None:
    """Warp a face from a page scan onto the ArcFace 5-point template."""
    buf = np.frombuffer(page_bytes, dtype=np.uint8)
    page = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if page is None:
        return None
    src = np.array(keypoints, dtype=np.float32)
    if src.shape != (5, 2):
        return None
    dst_template = ARCFACE_KEYPOINTS.copy()
    dst_template[:, 0] *= target_size[0] / DEFAULT_TARGET_SIZE[0]
    dst_template[:, 1] *= target_size[1] / DEFAULT_TARGET_SIZE[1]
    M, _ = cv2.estimateAffinePartial2D(src, dst_template, ransacReprojThreshold=float("inf"))
    return cv2.warpAffine(page, M, target_size, flags=cv2.INTER_CUBIC) if M is not None else None


# --- Zip helpers ---

def _read_faces(zf: zipfile.ZipFile, jsonl_name: str) -> set[str]:
    with zf.open(jsonl_name) as f:
        return {json.loads(line).get("face", "") for line in f} - {""}


def _build_crop_map(zf: zipfile.ZipFile) -> dict[str, str]:
    return {
        name.removeprefix(DATA_PREFIX): name
        for name in zf.namelist()
        if ALIGNED_CROPS_SUFFIX in name and name.lower().endswith((".jpg", ".jpeg", ".png"))
    }


def _assign_splits(crop_map: dict, dev_faces: set, test_faces: set) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    for face in crop_map:
        if face in test_faces:
            splits["test"].append(face)
        elif face in dev_faces:
            splits["dev"].append(face)
        else:
            splits["train"].append(face)
    return splits


def _process_face(zf, face, crop_map, method, page_kpts, page_paths, target_size):
    """Read one face from the archive and return the processed image array."""
    if method == "realign" and face in page_kpts:
        img = realign_from_page(zf.read(page_paths[face]), page_kpts[face], target_size)
        if img is not None:
            return img
    return resize_image_bytes(zf.read(crop_map[face]), target_size)


def _load_page_keypoints(zf: zipfile.ZipFile):
    """Load keypoints from both .people_gator.jsonl and corresponding_faces files.

    Processes .people_gator.jsonl first so that corresponding_faces keypoints
    take precedence for overlapping face keys (they have annotator-verified data).
    """
    kpts: dict[str, list] = {}
    paths: dict[str, str] = {}
    all_names = zf.namelist()
    pg_names = [n for n in all_names if n.endswith(".people_gator.jsonl")]
    cf_names = [n for n in all_names
                if n.startswith(CORRESPONDING_FACES_PREFIX) and n.endswith(".jsonl")]
    for name in pg_names + cf_names:
        is_pg = name.endswith(".people_gator.jsonl")
        with zf.open(name) as f:
            for line in f:
                _parse_keypoint_record(json.loads(line), is_pg, kpts, paths)
    return kpts, paths


def _parse_keypoint_record(rec: dict, is_pg: bool, kpts: dict, paths: dict):
    kp = rec.get("page_keypoints")
    if not kp:
        return
    lib, doc = rec.get("library", ""), rec.get("document", "")
    page_path = f"{DATA_PREFIX}{lib}/{doc}.images/{rec.get('page', '')}"
    if is_pg:
        crop = rec.get("crop_name", "")
        key = f"{lib}/{doc}{ALIGNED_CROPS_SUFFIX}/{crop}" if crop and lib and doc else None
    else:
        key = rec.get("face", "") or None
    if key:
        kpts[key] = kp
        paths[key] = page_path


def _extract_jsonls(zf: zipfile.ZipFile, output_dir: Path, kept_faces: set[str] | None):
    for name in zf.namelist():
        if not (name.startswith(CORRESPONDING_FACES_PREFIX) and name.endswith(".jsonl")):
            continue
        short = name.split("/")[-1].replace(
            "people_gator__corresponding_faces__2026-02-11.", "corresponding_faces_"
        )
        with zf.open(name) as src:
            records = [json.loads(line) for line in src]
        if kept_faces is not None:
            records = [r for r in records if r.get("face", "") in kept_faces]
        with open(output_dir / short, "w") as dst:
            for r in records:
                dst.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {short}: {len(records)} records")


def _target_subdir_name(target_size: tuple[int, int]) -> str:
    if target_size == DEFAULT_TARGET_SIZE:
        return "aligned_112"
    return f"aligned_{target_size[0]}x{target_size[1]}"


def _image_metrics(img: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness_mean": float(np.mean(gray)),
        "contrast_std": float(np.std(gray)),
        "sharpness_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def _write_metadata(records: list[dict], metadata_path: Path, fmt: str):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        fields = [
            "split",
            "face",
            "library",
            "identity",
            "output_path",
            "width",
            "height",
            "file_size_bytes",
            "brightness_mean",
            "contrast_std",
            "sharpness_laplacian_var",
        ]
        with metadata_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        return

    with metadata_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Preprocess people_gator face crops")
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/people_gator"))
    parser.add_argument("--method", choices=["resize", "realign"], default="resize",
                        help="resize = fast crop resize; realign = ArcFace alignment from page scans")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        default=list(DEFAULT_TARGET_SIZE),
                        help="Output face size in pixels (default: 112 112)")
    parser.add_argument("--metadata-path", type=Path, default=None,
                        help="Optional metadata output path (.jsonl or .csv)")
    parser.add_argument("--metadata-format", choices=["jsonl", "csv"], default="jsonl",
                        help="Metadata file format (default: jsonl)")
    args = parser.parse_args()

    if not args.zip_path.exists():
        print(f"Error: {args.zip_path} not found", file=sys.stderr)
        sys.exit(1)

    target_size = (args.target_size[0], args.target_size[1])
    if target_size[0] <= 0 or target_size[1] <= 0:
        print("Error: target width and height must be positive integers", file=sys.stderr)
        sys.exit(1)

    aligned_dir = args.output_dir / _target_subdir_name(target_size)
    if aligned_dir.exists():
        shutil.rmtree(aligned_dir)
    for split in ("train", "dev", "test"):
        (aligned_dir / split).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path, "r") as zf:
        dev_faces = _read_faces(zf, f"{CORRESPONDING_FACES_PREFIX}.dev.jsonl")
        test_faces = _read_faces(zf, f"{CORRESPONDING_FACES_PREFIX}.test.jsonl")
        crop_map = _build_crop_map(zf)
        splits = _assign_splits(crop_map, dev_faces, test_faces)
        print(f"  Crops: {len(crop_map)} total, dev={len(splits['dev'])}, test={len(splits['test'])}")

        page_kpts, page_paths = ({}, {})
        if args.method == "realign":
            page_kpts, page_paths = _load_page_keypoints(zf)
            print(f"  Keypoints for {len(page_kpts)} faces")

        counts = dict.fromkeys(("train", "dev", "test", "skipped"), 0)
        metadata_records: list[dict] = []
        written_faces: set[str] = set()
        for split, faces in splits.items():
            for face in tqdm(faces, desc=split):
                out_path = aligned_dir / split / face
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    img = _process_face(
                        zf, face, crop_map, args.method, page_kpts, page_paths, target_size
                    )
                    if not cv2.imwrite(str(out_path), img):
                        raise ValueError("Failed to write image")
                    counts[split] += 1
                    written_faces.add(face)
                    rel_parts = Path(face).parts
                    identity = str(Path(face).parent)
                    record = {
                        "split": split,
                        "face": face,
                        "library": rel_parts[0] if rel_parts else "",
                        "identity": identity,
                        "output_path": str(out_path.relative_to(args.output_dir)),
                        "width": int(img.shape[1]),
                        "height": int(img.shape[0]),
                        "file_size_bytes": out_path.stat().st_size,
                    }
                    record.update(_image_metrics(img))
                    metadata_records.append(record)
                except Exception as e:
                    tqdm.write(f"  Skipped {face}: {e}")
                    counts["skipped"] += 1

        # Keep annotation files consistent with images that were actually written.
        _extract_jsonls(zf, args.output_dir, written_faces)
        if args.metadata_path is not None:
            _write_metadata(metadata_records, args.metadata_path, args.metadata_format)
            print(f"  Metadata written: {args.metadata_path} ({len(metadata_records)} rows)")

    print(f"\nDone: train={counts['train']}, dev={counts['dev']}, test={counts['test']}", end="")
    print(f", skipped={counts['skipped']}" if counts["skipped"] else "")


if __name__ == "__main__":
    main()
