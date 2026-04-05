"""Extract, resize, and split people_gator aligned face crops.

Usage:
    python -m src.data_prep.preprocess_people_gator \
        --zip-path /path/to/people_gator__data_export.zip \
        --output-dir data/people_gator

    # For a small sample:
    python -m src.data_prep.preprocess_people_gator \
        --zip-path /path/to/people_gator__data_export.zip \
        --output-dir sample_data/people_gator --limit 100
"""

import argparse
import json
import random
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
TARGET_SIZE = (112, 112)
CORRESPONDING_FACES_PREFIX = "people_gator__corresponding_faces__2026-02-11"
ALIGNED_CROPS_SUFFIX = ".peoplegator_aligned_crops"
DATA_PREFIX = "people_gator__data/"


def resize_image_bytes(image_bytes: bytes) -> np.ndarray:
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def realign_from_page(page_bytes: bytes, keypoints: list[list[float]]) -> np.ndarray | None:
    """Warp a face from a page scan onto the ArcFace 5-point template."""
    buf = np.frombuffer(page_bytes, dtype=np.uint8)
    page = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if page is None:
        return None
    src = np.array(keypoints, dtype=np.float32)
    if src.shape != (5, 2):
        return None
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_KEYPOINTS, ransacReprojThreshold=float("inf"))
    return cv2.warpAffine(page, M, TARGET_SIZE, flags=cv2.INTER_CUBIC) if M is not None else None


def sample_diverse(items: list[str], n: int | None) -> list[str]:
    """Sample up to n items, round-robin across the first path component (library)."""
    if n is None or n >= len(items):
        return items
    by_group: dict[str, list[str]] = {}
    for item in items:
        by_group.setdefault(item.split("/")[0], []).append(item)
    for v in by_group.values():
        random.shuffle(v)

    picked: list[str] = []
    groups = sorted(by_group.keys())
    idx = 0
    while len(picked) < n:
        batch = [by_group[g][idx] for g in groups if idx < len(by_group[g]) and len(picked) + 1 <= n]
        if not batch:
            break
        picked.extend(batch[:n - len(picked)])
        idx += 1
    return picked


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


def _process_face(zf, face, crop_map, method, page_kpts, page_paths):
    """Read one face from the archive and return the processed image array."""
    if method == "realign" and face in page_kpts:
        img = realign_from_page(zf.read(page_paths[face]), page_kpts[face])
        if img is not None:
            return img
    return resize_image_bytes(zf.read(crop_map[face]))


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


def main():
    parser = argparse.ArgumentParser(description="Preprocess people_gator face crops")
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/people_gator"))
    parser.add_argument("--method", choices=["resize", "realign"], default="resize",
                        help="resize = fast crop resize; realign = ArcFace alignment from page scans")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max train images (for generating sample subsets)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.zip_path.exists():
        print(f"Error: {args.zip_path} not found", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    aligned_dir = args.output_dir / "aligned_112"
    for split in ("train", "dev", "test"):
        (aligned_dir / split).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path, "r") as zf:
        dev_faces = _read_faces(zf, f"{CORRESPONDING_FACES_PREFIX}.dev.jsonl")
        test_faces = _read_faces(zf, f"{CORRESPONDING_FACES_PREFIX}.test.jsonl")
        crop_map = _build_crop_map(zf)
        splits = _assign_splits(crop_map, dev_faces, test_faces)
        print(f"  Crops: {len(crop_map)} total, dev={len(splits['dev'])}, test={len(splits['test'])}")

        if args.limit:
            splits["train"] = sample_diverse(splits["train"], args.limit)
            splits["dev"] = sample_diverse(splits["dev"], min(args.limit // 3, len(splits["dev"])))
            splits["test"] = sample_diverse(splits["test"], min(args.limit // 3, len(splits["test"])))

        page_kpts, page_paths = ({}, {})
        if args.method == "realign":
            page_kpts, page_paths = _load_page_keypoints(zf)
            print(f"  Keypoints for {len(page_kpts)} faces")

        counts = dict.fromkeys(("train", "dev", "test", "skipped"), 0)
        for split, faces in splits.items():
            for face in tqdm(faces, desc=split):
                out_path = aligned_dir / split / face
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    img = _process_face(zf, face, crop_map, args.method, page_kpts, page_paths)
                    cv2.imwrite(str(out_path), img)
                    counts[split] += 1
                except Exception as e:
                    tqdm.write(f"  Skipped {face}: {e}")
                    counts["skipped"] += 1

        all_kept = {f for faces in splits.values() for f in faces} if args.limit else None
        _extract_jsonls(zf, args.output_dir, all_kept)

    print(f"\nDone: train={counts['train']}, dev={counts['dev']}, test={counts['test']}", end="")
    print(f", skipped={counts['skipped']}" if counts["skipped"] else "")


if __name__ == "__main__":
    main()
