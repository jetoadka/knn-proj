from __future__ import annotations

"""Prepare datasets in the folder-per-identity format for downstream training.

Supports multiple modes:
1. Combine clean + augmented data into a single training directory
2. Create a gallery/probe structure for evaluation
3. Generate class-label mappings (identity → integer)

Usage:
    # Combine WebFace4M + CUT-generated for training (E2)
    python -m src.downstream.prepare_timm_dataset \
        --sources data/webface4m data/augmented_newspaper \
        --output-dir data/downstream/E2_augmented/train \
        --mode train

    # Prepare people_gator test for evaluation
    python -m src.downstream.prepare_timm_dataset \
        --sources data/people_gator/aligned_112/test \
        --output-dir data/downstream/eval_test \
        --mode eval
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(directory: Path) -> list[Path]:
    """Recursively find all image files."""
    return sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_identity(image_path: Path, source_root: Path) -> str:
    """Extract identity name from folder structure.

    Expects: source_root / identity_name / image_file.jpg
    For flat structures (no subfolders), uses filename prefix.
    """
    rel = image_path.relative_to(source_root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    # Flat structure: use filename prefix (before last underscore)
    stem = image_path.stem
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) > 1 else stem


def prepare_train(sources: list[Path], output_dir: Path, symlink: bool = False):
    """Combine multiple source directories into a unified training directory.

    Output structure:
        output_dir/
        ├── 00000_IdentityA/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── 00001_IdentityB/
        │   └── img1.jpg
        ...

    Also writes a class_labels.json mapping identity → integer label.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (identity, image_path) pairs
    all_images: list[tuple[str, Path]] = []
    for source in sources:
        if not source.exists():
            print(f"WARNING: Source {source} does not exist, skipping")
            continue
        images = find_images(source)
        print(f"  {source}: {len(images)} images")
        for img in images:
            identity = get_identity(img, source)
            all_images.append((identity, img))

    if not all_images:
        print("ERROR: No images found in any source directory")
        return

    # Build identity → label mapping
    identities = sorted(set(ident for ident, _ in all_images))
    label_map = {ident: idx for idx, ident in enumerate(identities)}

    print(f"\nTotal: {len(all_images)} images, {len(identities)} identities")

    # Copy/symlink images
    counts = {"copied": 0, "skipped": 0}
    for identity, img_path in tqdm(all_images, desc="Preparing dataset"):
        label = label_map[identity]
        dest_dir = output_dir / f"{label:05d}_{identity}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename: source_dir_name + original_name
        source_prefix = img_path.parent.name
        dest_name = f"{source_prefix}_{img_path.name}"
        dest_path = dest_dir / dest_name

        if dest_path.exists():
            counts["skipped"] += 1
            continue

        try:
            if symlink:
                os.symlink(img_path.resolve(), dest_path)
            else:
                shutil.copy2(img_path, dest_path)
            counts["copied"] += 1
        except Exception as e:
            tqdm.write(f"  Error: {img_path}: {e}")
            counts["skipped"] += 1

    # Save label mapping
    labels_path = output_dir.parent / f"{output_dir.name}_labels.json"
    with open(labels_path, "w") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {counts['copied']} copied, {counts['skipped']} skipped")
    print(f"Label mapping saved to {labels_path}")
    print(f"Number of classes: {len(identities)}")


def prepare_eval(sources: list[Path], output_dir: Path):
    """Prepare evaluation data: flat copy with identity tracking.

    Output structure:
        output_dir/
        ├── images/
        │   ├── 000_identity_img.jpg
        │   ...
        └── pairs.json   (identity → [list of image filenames])
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    pairs: dict[str, list[str]] = {}
    idx = 0

    for source in sources:
        if not source.exists():
            print(f"WARNING: Source {source} does not exist, skipping")
            continue

        images = find_images(source)
        print(f"  {source}: {len(images)} images")

        for img_path in tqdm(images, desc=f"Processing {source.name}"):
            identity = get_identity(img_path, source)
            dest_name = f"{idx:05d}_{identity}_{img_path.name}"
            dest_path = images_dir / dest_name

            shutil.copy2(img_path, dest_path)
            pairs.setdefault(identity, []).append(dest_name)
            idx += 1

    pairs_path = output_dir / "pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"\nDone: {idx} images, {len(pairs)} identities")
    print(f"Pairs file saved to {pairs_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for downstream training/eval")
    parser.add_argument("--sources", nargs="+", type=Path, required=True,
                        help="Source directories of face images")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="train: merge into identity folders; eval: create pairs file")
    parser.add_argument("--symlink", action="store_true",
                        help="Use symlinks instead of copying (saves disk space)")
    args = parser.parse_args()

    if args.mode == "train":
        prepare_train(args.sources, args.output_dir, symlink=args.symlink)
    else:
        prepare_eval(args.sources, args.output_dir)


if __name__ == "__main__":
    main()
