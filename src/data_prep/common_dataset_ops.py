"""Data prep: shared dataset IO and packaging helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_split_from_annotation_name(name: str) -> str | None:
    if ".dev" in name or "_dev" in name:
        return "dev"
    if ".test" in name or "_test" in name:
        return "test"
    return None


def collect_images(aligned_dir: Path, splits: list[str], exts: set[str] | None = None) -> list[dict]:
    exts = exts or IMAGE_EXTS
    rows: list[dict] = []
    for split in splits:
        split_dir = aligned_dir / split
        if not split_dir.exists():
            continue
        for p in split_dir.rglob("*"):
            if p.suffix.lower() not in exts:
                continue
            rows.append(
                {
                    "split": split,
                    "rel_path": str(p.relative_to(split_dir)),
                    "abs_path": p,
                }
            )
    return rows


def copy_kept_images(
    aligned_in: Path,
    aligned_out: Path,
    kept_by_split: dict[str, set[str]],
) -> tuple[int, int]:
    copied = 0
    missing = 0
    for split, rel_paths in kept_by_split.items():
        for rel in rel_paths:
            src = aligned_in / split / rel
            dst = aligned_out / split / rel
            if not src.exists():
                missing += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied, missing


def filter_corresponding_faces(input_root: Path, output_root: Path, kept_by_split: dict[str, set[str]]) -> dict:
    stats = {}
    for jf in sorted(input_root.glob("corresponding_faces*.jsonl")):
        name = jf.name
        split = infer_split_from_annotation_name(name)
        if split is None:
            shutil.copy2(jf, output_root / name)
            continue

        rows = read_jsonl(jf)
        kept_faces = kept_by_split.get(split, set())
        filtered = [r for r in rows if r.get("face", "") in kept_faces]
        with (output_root / name).open("w") as f:
            for row in filtered:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[name] = {"input": len(rows), "kept": len(filtered)}
    return stats


def copy_passthrough_files(
    input_root: Path,
    output_root: Path,
    skip_dirs: set[str] | None = None,
):
    skip_dirs = skip_dirs or {"aligned_112", "_metadata"}
    for p in input_root.iterdir():
        if p.name in skip_dirs:
            continue
        if p.is_file() and not p.name.startswith("corresponding_faces"):
            shutil.copy2(p, output_root / p.name)


def copy_dropped_images(
    aligned_in: Path,
    output_root: Path,
    dropped_by_split: dict[str, set[str]],
) -> tuple[int, int]:
    copied = 0
    missing = 0
    for split, rel_paths in dropped_by_split.items():
        for rel in rel_paths:
            src = aligned_in / split / rel
            dst = output_root / "_dropped" / split / rel
            if not src.exists():
                missing += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    return copied, missing


def copy_input_metadata(input_root: Path, metadata_out: Path):
    in_meta = input_root / "_metadata"
    if not in_meta.exists():
        return
    dst_meta = metadata_out / "input_metadata"
    if dst_meta.exists():
        shutil.rmtree(dst_meta)
    shutil.copytree(in_meta, dst_meta)


def materialize_dataset_from_kept_manifest(
    source_dataset_dir: Path,
    kept_manifest: Path,
    output_dataset_dir: Path,
    aligned_subdir: str = "aligned_112",
) -> dict:
    if output_dataset_dir.exists():
        shutil.rmtree(output_dataset_dir)
    (output_dataset_dir / aligned_subdir).mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(kept_manifest)
    kept_by_split = {"train": set(), "dev": set(), "test": set()}
    for row in rows:
        split = row.get("split")
        rel = row.get("rel_path")
        if split in kept_by_split and rel:
            kept_by_split[split].add(rel)

    copied = 0
    for split, rel_paths in kept_by_split.items():
        for rel in rel_paths:
            src = source_dataset_dir / aligned_subdir / split / rel
            if src.suffix.lower() not in IMAGE_EXTS:
                continue
            if not src.exists():
                continue
            dst = output_dataset_dir / aligned_subdir / split / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    filter_corresponding_faces(source_dataset_dir, output_dataset_dir, kept_by_split)
    copy_passthrough_files(source_dataset_dir, output_dataset_dir, skip_dirs={aligned_subdir, "_metadata"})

    summary = {
        "source_dataset_dir": str(source_dataset_dir),
        "kept_manifest": str(kept_manifest),
        "images_copied": copied,
        "kept_by_split": {k: len(v) for k, v in kept_by_split.items()},
    }
    meta = output_dataset_dir / "_metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "materialize_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
