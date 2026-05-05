"""Print statistics for all prepared datasets.

Scans the data/ directory and reports image counts, resolutions,
identity distributions, and split sizes.

Usage:
    python -m src.data_prep.dataset_stats --data-dir data
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def count_images(directory: Path) -> list[Path]:
    """Recursively find all image files."""
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in directory.rglob("*") if p.suffix.lower() in exts]


def check_resolutions(images: list[Path], sample_size: int = 50) -> dict[tuple[int, int], int]:
    """Sample image resolutions."""
    import random
    sample = random.sample(images, min(sample_size, len(images)))
    res_counts: Counter[tuple[int, int]] = Counter()
    for p in sample:
        try:
            img = Image.open(p)
            res_counts[img.size] += 1
        except Exception:
            res_counts[("error",)] += 1
    return dict(res_counts)


def stats_people_gator(data_dir: Path):
    """Report people_gator dataset statistics."""
    pg_dir = data_dir / "people_gator" / "aligned_112"
    if not pg_dir.exists():
        print("  [not found]")
        return

    for split in ("train", "dev", "test"):
        split_dir = pg_dir / split
        if split_dir.exists():
            images = count_images(split_dir)
            print(f"  {split}: {len(images)} images")
            if images:
                resolutions = check_resolutions(images)
                print(f"    Resolutions (sampled): {resolutions}")

                libraries = Counter()
                for p in images:
                    rel = p.relative_to(split_dir)
                    if len(rel.parts) >= 2:
                        libraries[rel.parts[0]] += 1
                print(f"    Libraries: {dict(libraries)}")
        else:
            print(f"  {split}: [not found]")

    jsonls = list((data_dir / "people_gator").glob("corresponding_faces_*.jsonl"))
    if jsonls:
        print(f"  Annotation files: {[j.name for j in sorted(jsonls)]}")


def stats_wiki_face(data_dir: Path):
    """Report wiki_face_112 dataset statistics."""
    wiki_dir = data_dir / "wiki_face_112"
    if not wiki_dir.exists():
        print("  [not found]")
        return

    images = count_images(wiki_dir)
    identities = set()
    imgs_per_id: Counter[str] = Counter()
    for p in images:
        rel = p.relative_to(wiki_dir)
        if len(rel.parts) >= 2:
            identity = rel.parts[0]
            identities.add(identity)
            imgs_per_id[identity] += 1

    print(f"  Total images: {len(images)}")
    print(f"  Identities: {len(identities)}")
    if imgs_per_id:
        counts = sorted(imgs_per_id.values())
        print(f"  Images per identity: min={counts[0]}, max={counts[-1]}, "
              f"median={counts[len(counts)//2]}")
    if images:
        resolutions = check_resolutions(images)
        print(f"  Resolutions (sampled): {resolutions}")


def stats_webface4m(data_dir: Path):
    """Report WebFace4M dataset statistics."""
    wf_dir = data_dir / "webface4m"
    if not wf_dir.exists():
        print("  [not found]")
        return

    shards = sorted(wf_dir.glob("webface4m-*.tar.gz"))
    loose_images = count_images(wf_dir)
    if shards:
        print(f"  Shards: {len(shards)}")
        total_bytes = sum(s.stat().st_size for s in shards)
        print(f"  Total size: {total_bytes / 1e9:.2f} GB")
        print(f"  Estimated images: ~{len(shards) * 53_000:,}")
        print(f"  Shard range: {shards[0].name} .. {shards[-1].name}")
    if loose_images:
        print(f"  Sample images: {len(loose_images)}")
        resolutions = check_resolutions(loose_images)
        print(f"    Resolutions (sampled): {resolutions}")
    if not shards and not loose_images:
        print("  [empty]")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def stats_filtering(filter_report: Path | None, kept_manifest: Path | None, rejected_manifest: Path | None):
    """Report filtering results and distribution drift."""
    if not filter_report and not kept_manifest and not rejected_manifest:
        return

    print("\n[filtering] Quality and diversity filtering")
    report_data = None
    if filter_report:
        if filter_report.exists():
            report_data = json.loads(filter_report.read_text())
            print(f"  Report file: {filter_report}")
            print(f"  Total input: {report_data.get('total_input', 'n/a')}")
            print(f"  Kept: {report_data.get('total_kept', 'n/a')}")
            print(f"  Rejected: {report_data.get('total_rejected', 'n/a')}")
            for split_report in report_data.get("split_reports", []):
                split = split_report.get("split", "unknown")
                kept = split_report.get("kept_images", 0)
                inp = max(split_report.get("input_images", 0), 1)
                print(f"  {split}: kept {kept}/{inp} ({100.0 * kept / inp:.1f}%)")
                before = split_report.get("libraries_before", {})
                after = split_report.get("libraries_after", {})
                if before and after:
                    drifts = []
                    for lib in sorted(before):
                        b = before[lib] / inp
                        a = after.get(lib, 0) / max(kept, 1)
                        drifts.append((lib, a - b))
                    drifts.sort(key=lambda x: abs(x[1]), reverse=True)
                    top = ", ".join(f"{lib}:{delta:+.3f}" for lib, delta in drifts[:5])
                    print(f"    Library share drift (top): {top}")
        else:
            print(f"  Report file not found: {filter_report}")

    kept_rows = _read_jsonl(kept_manifest) if kept_manifest and kept_manifest.exists() else []
    rejected_rows = _read_jsonl(rejected_manifest) if rejected_manifest and rejected_manifest.exists() else []
    if kept_manifest and not kept_rows and not kept_manifest.exists():
        print(f"  Kept manifest not found: {kept_manifest}")
    if rejected_manifest and not rejected_rows and not rejected_manifest.exists():
        print(f"  Rejected manifest not found: {rejected_manifest}")

    if kept_rows:
        scores = sorted(
            r.get("quality_score")
            for r in kept_rows
            if isinstance(r.get("quality_score"), (int, float))
        )
        if scores:
            n = len(scores)
            q25 = scores[int(0.25 * (n - 1))]
            q50 = scores[int(0.50 * (n - 1))]
            q75 = scores[int(0.75 * (n - 1))]
            print(f"  Kept quality score quantiles: q25={q25:.4f}, q50={q50:.4f}, q75={q75:.4f}")
        print(f"  Kept manifest rows: {len(kept_rows)}")
    if rejected_rows:
        print(f"  Rejected manifest rows: {len(rejected_rows)}")


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Root data directory (default: data)")
    parser.add_argument("--filter-report", type=Path, default=None,
                        help="Optional path to filter_report.json")
    parser.add_argument("--kept-manifest", type=Path, default=None,
                        help="Optional path to kept_manifest.jsonl")
    parser.add_argument("--rejected-manifest", type=Path, default=None,
                        help="Optional path to rejected_manifest.jsonl")
    args = parser.parse_args()

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print("\n[people_gator] Target domain - historical newspaper faces")
    stats_people_gator(args.data_dir)

    print("\n[wiki_face_112] Supplementary - Wikipedia historical faces")
    stats_wiki_face(args.data_dir)

    print("\n[webface4m] Source domain - clean faces (WebDataset)")
    stats_webface4m(args.data_dir)
    stats_filtering(args.filter_report, args.kept_manifest, args.rejected_manifest)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
