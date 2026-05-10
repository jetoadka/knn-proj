"""Data prep: download WebFace4M shards from HuggingFace.

Downloads selected shards from the gaunernst/webface4m-wds-gz dataset.
Each shard is a .tar.gz file containing ~53k face images (112x112, aligned)
in WebDataset format ({key}.jpg + {key}.cls pairs).

Usage:
    python -m src.data_prep.download_webface4m \
        --output-dir data/webface4m \
        --num-shards 2
"""

import argparse
import sys
from pathlib import Path


REPO_ID = "gaunernst/webface4m-wds-gz"
TOTAL_SHARDS = 121


def main():
    parser = argparse.ArgumentParser(description="Download WebFace4M shards")
    parser.add_argument("--output-dir", type=Path, default=Path("data/webface4m"),
                        help="Output directory (default: data/webface4m)")
    parser.add_argument("--num-shards", type=int, default=2,
                        help="Number of shards to download (default: 2, each ~80MB with ~53k images)")
    parser.add_argument("--start-shard", type=int, default=0,
                        help="First shard index (default: 0)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify downloaded shards by reading a sample image")
    args = parser.parse_args()

    if args.num_shards <= 0:
        print("Error: --num-shards must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.start_shard < 0 or args.start_shard >= TOTAL_SHARDS:
        print(f"Error: --start-shard must be in [0, {TOTAL_SHARDS - 1}]", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub",
              file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    end_shard = min(args.start_shard + args.num_shards, TOTAL_SHARDS)
    shard_names = [f"webface4m-{i:04d}.tar.gz" for i in range(args.start_shard, end_shard)]
    if not shard_names:
        print("Error: computed shard list is empty; check --start-shard and --num-shards", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {len(shard_names)} shard(s) to {args.output_dir}")
    print(f"  Shards: {shard_names[0]} ... {shard_names[-1]}")

    for shard_name in shard_names:
        print(f"\n  Downloading {shard_name} ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=shard_name,
            repo_type="dataset",
            local_dir=str(args.output_dir),
        )
        print(f"    Saved to {local_path}")

    if args.verify:
        print("\nVerifying first shard ...")
        verify_shard(args.output_dir / shard_names[0])

    print("\nDone!")
    print(f"  Downloaded {len(shard_names)} shards to {args.output_dir}")
    print(f"  Estimated images: ~{len(shard_names) * 53_000:,}")
    print(f"\n  For timm-face training, use: --ds_path 'wds://{args.output_dir}/*.tar.gz'")


def verify_shard(shard_path: Path):
    """Read a sample from the shard to verify it's a valid WebDataset."""
    import io
    import tarfile

    from PIL import Image

    with tarfile.open(shard_path, "r:gz") as tar:
        count = 0
        sample_img = None
        sample_cls = None
        for member in tar:
            if member.name.endswith(".jpg") and sample_img is None:
                f = tar.extractfile(member)
                if f:
                    sample_img = Image.open(io.BytesIO(f.read()))
            elif member.name.endswith(".cls") and sample_cls is None:
                f = tar.extractfile(member)
                if f:
                    sample_cls = f.read().decode().strip()
            count += 1
            if count > 100:
                break

        if sample_img:
            print(f"    Sample image size: {sample_img.size}, mode: {sample_img.mode}")
        if sample_cls:
            print(f"    Sample class label: {sample_cls}")
        print(f"    Members scanned: {count}")


if __name__ == "__main__":
    main()
