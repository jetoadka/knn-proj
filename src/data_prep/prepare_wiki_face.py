"""Data prep: extract and organize the wiki_face_112_fin dataset.

Usage:
    python -m src.data_prep.prepare_wiki_face \
        --zip-path /path/to/wiki_face_112_fin.zip \
        --output-dir data/wiki_face_112

    # For a small sample:
    python -m src.data_prep.prepare_wiki_face \
        --zip-path /path/to/wiki_face_112_fin.zip \
        --output-dir sample_data/wiki_face_112 --limit 100
"""

import argparse
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

TARGET_SIZE = (112, 112)


def _index_archive(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """Group zip members by identity folder."""
    by_identity: dict[str, list[str]] = {}
    for m in zf.namelist():
        if not m.lower().endswith((".jpg", ".jpeg", ".png")) or m.startswith("__MACOSX"):
            continue
        parts = Path(m).parts
        if len(parts) < 2:
            continue
        identity = parts[1] if parts[0] == "wiki_face_112_fin" else parts[0]
        by_identity.setdefault(identity, []).append(m)
    return by_identity


def _select_identities(by_identity: dict[str, list[str]], limit: int | None) -> list[str]:
    """Return identity list, optionally subsampled to reach limit images."""
    identities = list(by_identity.keys())
    if limit is None or sum(len(v) for v in by_identity.values()) <= limit:
        return identities
    random.shuffle(identities)
    selected, total = [], 0
    for ident in identities:
        selected.append(ident)
        total += len(by_identity[ident])
        if total >= limit:
            break
    return selected


def _extract_images(zf: zipfile.ZipFile, by_identity: dict[str, list[str]],
                    identities: list[str], output_dir: Path, verify: bool):
    """Extract images and return (counts_per_identity, bad_sizes)."""
    counts: Counter[str] = Counter()
    bad_sizes = []
    for ident in tqdm(identities, desc="Extracting"):
        for member in by_identity[ident]:
            dest = output_dir / ident / Path(member).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            counts[ident] += 1
            if verify and Image.open(dest).size != TARGET_SIZE:
                bad_sizes.append((dest, Image.open(dest).size))
    return counts, bad_sizes


def main():
    parser = argparse.ArgumentParser(description="Prepare wiki_face_112_fin dataset")
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/wiki_face_112"))
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images to extract (selects random identities until limit)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-all", action="store_true",
                        help="Check resolution of every extracted image")
    args = parser.parse_args()

    if not args.zip_path.exists():
        print(f"Error: {args.zip_path} not found", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path, "r") as zf:
        by_identity = _index_archive(zf)
        identities = _select_identities(by_identity, args.limit)
        counts, bad_sizes = _extract_images(
            zf, by_identity, identities, args.output_dir, args.verify_all
        )

    total = sum(counts.values())
    vals = sorted(counts.values())
    print(f"Done: {total} images, {len(counts)} identities")
    if vals:
        print(f"  Images per identity: min={vals[0]}, max={vals[-1]}, median={vals[len(vals)//2]}")
    else:
        print("  No images extracted (check archive layout and arguments).")
    if args.verify_all:
        print(f"  {'WARNING: ' + str(len(bad_sizes)) + ' wrong size' if bad_sizes else 'All verified as ' + str(TARGET_SIZE)}")


if __name__ == "__main__":
    main()
