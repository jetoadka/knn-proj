from __future__ import annotations

"""Evaluation metrics for face recognition.

Provides functions for:
- Rank-1 identification accuracy (gallery vs probe)
- FAR / FRR computation across thresholds
- TAR @ FAR = 1e-4 (standard IJB metric)
- Pairwise verification accuracy (10-fold cross-validation)

Usage:
    python -m src.evaluation.metrics \
        --model-checkpoint checkpoints/E2_augmented/best_model.pth \
        --eval-dir data/downstream/eval_test \
        --backbone convnext_atto \
        --device mps
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


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
# Metrics
# ============================================================================

def cosine_similarity_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of embeddings.

    Args:
        embeddings_a: (N, D) normalized embeddings
        embeddings_b: (M, D) normalized embeddings

    Returns:
        (N, M) similarity matrix
    """
    # Ensure normalized
    norm_a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
    norm_b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
    return norm_a @ norm_b.T


def rank1_accuracy(
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    probe_embeddings: np.ndarray,
    probe_labels: np.ndarray,
) -> float:
    """Compute Rank-1 identification accuracy.

    For each probe, find the most similar gallery image.
    If the top-1 match has the same identity, it's correct.
    """
    sim_matrix = cosine_similarity_matrix(probe_embeddings, gallery_embeddings)
    top1_indices = np.argmax(sim_matrix, axis=1)
    top1_labels = gallery_labels[top1_indices]
    correct = np.sum(top1_labels == probe_labels)
    return correct / len(probe_labels)


def rank_k_accuracy(
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    probe_embeddings: np.ndarray,
    probe_labels: np.ndarray,
    k: int = 5,
) -> float:
    """Compute Rank-K identification accuracy."""
    sim_matrix = cosine_similarity_matrix(probe_embeddings, gallery_embeddings)
    topk_indices = np.argsort(-sim_matrix, axis=1)[:, :k]
    topk_labels = gallery_labels[topk_indices]
    correct = np.any(topk_labels == probe_labels[:, None], axis=1).sum()
    return correct / len(probe_labels)


def compute_far_frr(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    n_thresholds: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute FAR and FRR at various thresholds.

    Args:
        genuine_scores: cosine similarities for genuine pairs (same identity)
        impostor_scores: cosine similarities for impostor pairs (different identity)

    Returns:
        (thresholds, FAR, FRR) arrays
    """
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.linspace(all_scores.min(), all_scores.max(), n_thresholds)

    far = np.array([np.mean(impostor_scores >= t) for t in thresholds])
    frr = np.array([np.mean(genuine_scores < t) for t in thresholds])

    return thresholds, far, frr


def tar_at_far(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    target_far: float = 1e-4,
) -> float:
    """Compute TAR (True Accept Rate) at a specific FAR threshold.

    Standard biometric benchmark metric (used by IJB-B/C).
    """
    thresholds, far_arr, frr_arr = compute_far_frr(genuine_scores, impostor_scores)

    # Find threshold closest to target FAR
    idx = np.argmin(np.abs(far_arr - target_far))
    tar = 1.0 - frr_arr[idx]
    actual_far = far_arr[idx]

    return tar, actual_far, thresholds[idx]


def kfold_verification_accuracy(
    labels: np.ndarray,
    scores: np.ndarray,
    n_folds: int = 10,
) -> float:
    """10-fold cross-validation verification accuracy.

    Given pair labels (1=same, 0=different) and similarity scores,
    find optimal threshold on train fold, evaluate on test fold.
    """
    kfold = KFold(n_folds, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in kfold.split(labels):
        # Find optimal threshold on train fold
        train_labels = labels[train_idx]
        train_scores = scores[train_idx]

        fpr, tpr, thresholds = roc_curve(train_labels, train_scores)
        # Maximize accuracy = TPR * P(positive) + (1-FPR) * P(negative)
        p_pos = train_labels.mean()
        accuracy_at_thresholds = tpr * p_pos + (1 - fpr) * (1 - p_pos)
        optimal_th = thresholds[np.argmax(accuracy_at_thresholds)]

        # Evaluate on test fold
        test_preds = scores[test_idx] >= optimal_th
        test_acc = np.mean(test_preds == labels[test_idx])
        accs.append(test_acc)

    return np.mean(accs)


def generate_pairs_from_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_pairs: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate genuine and impostor pairs for verification evaluation.

    Returns:
        (pair_labels, pair_scores) where pair_labels is 1 for genuine, 0 for impostor
    """
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    label_to_indices = {l: np.where(labels == l)[0] for l in unique_labels}

    pair_labels = []
    pair_scores = []

    # Generate genuine pairs
    genuine_count = 0
    for label in unique_labels:
        indices = label_to_indices[label]
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sim = float(embeddings[indices[i]] @ embeddings[indices[j]])
                    pair_labels.append(1)
                    pair_scores.append(sim)
                    genuine_count += 1
                    if genuine_count >= n_pairs // 2:
                        break
                if genuine_count >= n_pairs // 2:
                    break
        if genuine_count >= n_pairs // 2:
            break

    # Generate impostor pairs
    impostor_count = 0
    while impostor_count < n_pairs // 2:
        l1, l2 = rng.choice(unique_labels, 2, replace=False)
        i1 = rng.choice(label_to_indices[l1])
        i2 = rng.choice(label_to_indices[l2])
        sim = float(embeddings[i1] @ embeddings[i2])
        pair_labels.append(0)
        pair_scores.append(sim)
        impostor_count += 1

    return np.array(pair_labels), np.array(pair_scores)


# ============================================================================
# Embedding extraction
# ============================================================================

class ImageFolderFlat(Dataset):
    """Load images from a flat or nested directory for embedding extraction.

    When *annotations_jsonl* is provided the labels are derived from the
    ``person_name`` field in the JSONL file (one JSON object per line).
    Only images that appear in the annotations are included – this avoids
    treating unannotated crops as identities.

    Without annotations the class falls back to inferring the identity from
    the directory structure (parent-folder name or filename prefix), which
    is correct for datasets like ``wiki_face_112`` but **incorrect** for
    ``people_gator`` where folder names represent libraries, not persons.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, root: Path, transform=None,
                 annotations_jsonl: Path | str | None = None):
        self.root = Path(root)
        self.transform = transform

        all_images = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        # --- Build identity labels ---
        self.images: list[Path] = []
        self.labels: list[int] = []
        label_map: dict[str, int] = {}

        if annotations_jsonl is not None:
            # Use JSONL annotations for ground-truth person identities
            from evaluation.identity_mapping import load_identity_map_for_dir
            abs_to_person = load_identity_map_for_dir(self.root, annotations_jsonl)
            if not abs_to_person:
                print(f"  WARNING: No annotations matched images in {self.root}")

            for p in all_images:
                person = abs_to_person.get(p.resolve())
                if person is None:
                    continue  # skip unannotated images
                if person not in label_map:
                    label_map[person] = len(label_map)
                self.images.append(p)
                self.labels.append(label_map[person])

            n_skipped = len(all_images) - len(self.images)
            if n_skipped:
                print(f"  Skipped {n_skipped} unannotated images")
            print(f"  Loaded {len(self.images)} annotated images, "
                  f"{len(label_map)} person identities")
        else:
            # Fallback: infer identity from folder structure
            self.images = all_images
            for p in self.images:
                identity = p.parent.name if p.parent != self.root else p.stem.rsplit("_", 1)[0]
                if identity not in label_map:
                    label_map[identity] = len(label_map)
                self.labels.append(label_map[identity])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


@torch.no_grad()
def extract_embeddings(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized embeddings from a model."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
        images = images.to(device)
        emb = model.get_embedding(images)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(np.array(labels))

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


# ============================================================================
# Full evaluation pipeline
# ============================================================================

def run_full_evaluation(
    embeddings: np.ndarray,
    labels: np.ndarray,
    experiment_name: str = "unknown",
) -> dict:
    """Run all metrics on a set of embeddings.

    Returns a dict of all computed metrics.
    """
    results = {"experiment": experiment_name}

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norms

    # --- Rank-based identification ---
    # Split into gallery (first image per identity) and probe (rest)
    unique_labels = np.unique(labels)
    gallery_idx = []
    probe_idx = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        gallery_idx.append(indices[0])
        probe_idx.extend(indices[1:])

    if probe_idx:
        gallery_idx = np.array(gallery_idx)
        probe_idx = np.array(probe_idx)

        r1 = rank1_accuracy(
            embeddings[gallery_idx], labels[gallery_idx],
            embeddings[probe_idx], labels[probe_idx],
        )
        r5 = rank_k_accuracy(
            embeddings[gallery_idx], labels[gallery_idx],
            embeddings[probe_idx], labels[probe_idx], k=5,
        )
        results["rank1_accuracy"] = r1
        results["rank5_accuracy"] = r5
        print(f"  Rank-1 accuracy: {r1:.4f}")
        print(f"  Rank-5 accuracy: {r5:.4f}")
    else:
        print("  WARNING: Not enough images per identity for rank-based evaluation")

    # --- Verification (pairwise) ---
    pair_labels, pair_scores = generate_pairs_from_embeddings(embeddings, labels)

    if len(pair_labels) > 0:
        genuine_scores = pair_scores[pair_labels == 1]
        impostor_scores = pair_scores[pair_labels == 0]

        if len(genuine_scores) > 0 and len(impostor_scores) > 0:
            tar, actual_far, threshold = tar_at_far(genuine_scores, impostor_scores, 1e-4)
            results["tar_at_far_1e4"] = tar
            results["threshold_at_far_1e4"] = threshold
            print(f"  TAR@FAR=1e-4: {tar:.4f} (threshold={threshold:.4f}, actual_far={actual_far:.6f})")

            kfold_acc = kfold_verification_accuracy(pair_labels, pair_scores)
            results["kfold_verification_accuracy"] = kfold_acc
            print(f"  10-fold verification accuracy: {kfold_acc:.4f}")
        else:
            print("  WARNING: Not enough pairs for verification metrics")

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate face recognition model")
    parser.add_argument("--model-checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="Evaluation data directory (folder-per-identity)")
    parser.add_argument("--backbone", type=str, default="convnext_atto")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="eval")
    parser.add_argument("--annotations-jsonl", type=Path, default=None,
                        help="JSONL file with person identity annotations")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results as JSON")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(str(args.model_checkpoint), map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    n_classes = config.get("n_classes", 100)  # Will be overridden by checkpoint
    backbone_kwargs = {}
    if "vit" in args.backbone:
        backbone_kwargs["img_size"] = 112

    # We need to import the model class
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

    # Data
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    eval_ds = ImageFolderFlat(
        args.eval_dir, transform=transform,
        annotations_jsonl=args.annotations_jsonl,
    )
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Eval dataset: {len(eval_ds)} images")

    # Extract embeddings
    embeddings, labels = extract_embeddings(model, eval_loader, device)
    print(f"Embeddings: {embeddings.shape}")

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {args.experiment_name}")
    print(f"{'='*60}")
    results = run_full_evaluation(embeddings, labels, args.experiment_name)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types to Python types for JSON
        json_results = {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in results.items()}
        with open(args.output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Log to W&B
    if wandb and wandb.run:
        for key, val in results.items():
            if isinstance(val, (int, float)):
                wandb.summary[f"test/{key}"] = val


if __name__ == "__main__":
    main()
