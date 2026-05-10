"""Extract embeddings from a trained model for PeopleGator retrieval evaluation.

This is the bridge between our face recognition model and the teacher's
PeopleGator evaluation scripts. It:
  1. Reads face paths from the test JSONL annotations
  2. Loads each aligned 112×112 face crop from people_gator__data
  3. Extracts embeddings using the trained model
  4. Saves image_paths.txt, image_embeddings.npy, dataset_config.json, engine_config.json

The output is ready to be consumed by:
  - peoplegator_namedfaces.retrieval.run (retrieval)
  - peoplegator_namedfaces.retrieval.evaluate (metrics)

Usage:
    python -m src.evaluation.extract_pg_embeddings \\
        --checkpoint checkpoints/E2-combined200/best_model.pth \\
        --data-dir ../people_gator__data_export/people_gator__data \\
        --annotations ../people_gator__data_export/people_gator__corresponding_faces__2026-02-11.test.jsonl \\
        --output-dir eval_output/E2-combined200 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ============================================================================
# Device
# ============================================================================

def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Dataset — loads images by face path from JSONL
# ============================================================================

class PeopleGatorDataset(Dataset):
    """Load people_gator face crops using paths from the annotations JSONL.

    The JSONL file contains records with a "face" field that gives the
    relative path to the aligned face crop inside the data directory.
    For example:
        {"face": "kvkli/bf9e81dc-.../009dbaa7-...__face_0.jpg", ...}

    The actual image is at: data_dir/kvkli/bf9e81dc-.../009dbaa7-...__face_0.jpg
    """

    def __init__(self, data_dir: Path, annotations_jsonl: Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Read all unique face paths from the annotations
        face_paths: set[str] = set()
        with open(annotations_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                face = record.get("face")
                if face:
                    face_paths.add(face)

        # Verify which images actually exist on disk
        self.face_paths: list[str] = []
        missing = 0
        for fp in sorted(face_paths):
            full_path = self.data_dir / fp
            if full_path.exists():
                self.face_paths.append(fp)
            else:
                missing += 1

        print(f"  PeopleGator dataset: {len(self.face_paths)} images found, "
              f"{missing} missing from disk")

    def __len__(self):
        return len(self.face_paths)

    def __getitem__(self, idx):
        face_path = self.face_paths[idx]
        full_path = self.data_dir / face_path
        img = Image.open(full_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, face_path


# ============================================================================
# Embedding extraction
# ============================================================================

@torch.no_grad()
def extract_embeddings(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Extract normalized embeddings from a model.

    Returns:
        (embeddings, face_paths) — embeddings shape (N, D), face_paths is list of strings
    """
    model.eval()
    all_embeddings = []
    all_paths = []

    for images, paths in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device)
        emb = model.get_embedding(images)
        # L2 normalize
        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu().numpy())
        all_paths.extend(paths)

    return np.concatenate(all_embeddings), all_paths


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract PeopleGator embeddings for teacher's retrieval evaluation"
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Path to people_gator__data directory containing face crops")
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Path to corresponding_faces test JSONL")
    parser.add_argument("--backbone", type=str, default="convnext_atto")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for embeddings and config files")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    n_classes = config.get("n_classes", 100)

    backbone_kwargs = {}
    if "vit" in args.backbone:
        backbone_kwargs["img_size"] = 112

    # Import the model class
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from downstream.train_downstream import FaceRecModel

    model = FaceRecModel(
        backbone_name=args.backbone,
        n_classes=n_classes,
        embedding_dim=args.embedding_dim,
        backbone_kwargs=backbone_kwargs,
    ).to(device)

    # Load weights (handle missing head weights for different n_classes)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
    model.load_state_dict(state_dict, strict=False)
    print(f"Model loaded from {args.checkpoint}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = PeopleGatorDataset(
        data_dir=args.data_dir,
        annotations_jsonl=args.annotations,
        transform=transform,
    )

    if len(dataset) == 0:
        print("ERROR: No images found. Check --data-dir and --annotations paths.")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------------
    # Extract embeddings
    # ------------------------------------------------------------------
    embeddings, face_paths = extract_embeddings(model, dataloader, device)
    print(f"Extracted {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. image_paths.txt
    paths_file = output_dir / "image_paths.txt"
    with open(paths_file, "w") as f:
        for p in face_paths:
            f.write(p + "\n")
    print(f"Saved {len(face_paths)} image paths to {paths_file}")

    # 2. image_embeddings.npy
    emb_file = output_dir / "image_embeddings.npy"
    np.save(emb_file, embeddings)
    print(f"Saved embeddings {embeddings.shape} to {emb_file}")

    # 3. dataset_config.json (for teacher's run.py)
    dataset_config = {
        "image_paths": str(paths_file.resolve()),
        "image_embeddings": str(emb_file.resolve()),
    }
    config_file = output_dir / "dataset_config.json"
    with open(config_file, "w") as f:
        json.dump(dataset_config, f, indent=2)
    print(f"Saved dataset config to {config_file}")

    # 4. engine_config.json (for teacher's run.py)
    engine_config = {"engine": "image_embedding"}
    engine_file = output_dir / "engine_config.json"
    with open(engine_file, "w") as f:
        json.dump(engine_config, f, indent=2)
    print(f"Saved engine config to {engine_file}")

    print(f"\n✓ All outputs saved to {output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Run retrieval:")
    print(f"     python -m peoplegator_namedfaces.retrieval.run \\")
    print(f"         --dataset {config_file} \\")
    print(f"         --queries <query_file.jsonl> \\")
    print(f"         --engine {engine_file} \\")
    print(f"         --output {output_dir}/results.pkl")
    print(f"  2. Run evaluation:")
    print(f"     python -m peoplegator_namedfaces.retrieval.evaluate \\")
    print(f"         --predictions {output_dir}/results.pkl \\")
    print(f"         --ground-truth <ground_truth.jsonl> \\")
    print(f"         --dataset {config_file} \\")
    print(f"         --top-k 1 5 10 50 \\")
    print(f"         --output-file {output_dir}/retrieval_metrics.csv")


if __name__ == "__main__":
    main()
