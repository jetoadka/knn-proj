from __future__ import annotations

"""Standalone evaluation driver for downstream face recognition experiments.

Combines the trained model with the full evaluation pipeline from
``src.evaluation.metrics`` and logs results to W&B.

Supports:
- Single checkpoint evaluation
- Batch evaluation of multiple checkpoints
- Gallery/probe split from folder-per-identity datasets
- JSON + W&B output

Usage:
    # Evaluate a single checkpoint on the held-out test set
    python -m src.downstream.evaluate \
        --checkpoint checkpoints/E2-augmented-newspaper/best_model.pth \
        --test-dir sample_data/people_gator/aligned_112/test \
        --backbone convnext_atto \
        --experiment-name E2-augmented

    # Batch evaluate all experiments
    python -m src.downstream.evaluate \
        --checkpoint \
            checkpoints/E1-baseline-clean/best_model.pth \
            checkpoints/E2-augmented-newspaper/best_model.pth \
            checkpoints/E3-augmented-mixed/best_model.pth \
        --test-dir sample_data/people_gator/aligned_112/test \
        --backbone convnext_atto \
        --experiment-name E1-baseline E2-augmented E3-mixed \
        --output results/all_results.json

    # Quick test (MPS, offline W&B)
    python -m src.downstream.evaluate \
        --checkpoint checkpoints/smoke-test/epoch_2.pth \
        --test-dir sample_data/people_gator/aligned_112/test \
        --backbone convnext_atto \
        --experiment-name smoke-test \
        --device mps --wandb-mode disabled
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    import wandb
except ImportError:
    wandb = None

# ---------------------------------------------------------------------------
# Imports from sibling modules
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from downstream.train_downstream import FaceRecModel, get_device  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    ImageFolderFlat,
    extract_embeddings,
    run_full_evaluation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def load_model_from_checkpoint(
    checkpoint_path: Path,
    backbone: str,
    embedding_dim: int,
    device: torch.device,
) -> FaceRecModel:
    """Load a trained FaceRecModel from a checkpoint file.

    Handles cases where the checkpoint was saved with a different number
    of classes (head weights are loaded with ``strict=False``).
    """
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    n_classes = int(config.get("n_classes", 100))
    loss_type = config.get("loss", "cosface")
    saved_backbone = config.get("backbone", backbone)

    backbone_kwargs = {}
    if "vit" in saved_backbone:
        backbone_kwargs["img_size"] = 112

    model = FaceRecModel(
        backbone_name=saved_backbone,
        n_classes=n_classes,
        loss_type=loss_type,
        embedding_dim=embedding_dim,
        backbone_kwargs=backbone_kwargs,
    )

    state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
    
    # Filter state_dict to avoid shape mismatch errors (e.g. head.weight sizes varying by training dataset)
    model_state = model.state_dict()
    valid_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state and v.shape == model_state[k].shape:
            valid_state_dict[k] = v
        elif k in model_state:
            print(f"    Skipped {k} (shape mismatch: {v.shape} vs {model_state[k].shape})")
            
    model.load_state_dict(valid_state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"  Loaded checkpoint: {checkpoint_path.name}")
    print(f"    backbone={saved_backbone}, loss={loss_type}, "
          f"n_classes={n_classes}, epoch={ckpt.get('epoch', '?')}")

    return model


def evaluate_single(
    checkpoint_path: Path,
    test_dir: Path,
    backbone: str,
    embedding_dim: int,
    device: torch.device,
    experiment_name: str,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict:
    """Run the full evaluation pipeline for a single checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}")

    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path, backbone, embedding_dim, device,
    )

    # Load test data
    test_ds = ImageFolderFlat(test_dir, transform=EVAL_TRANSFORM)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"  Test dataset: {len(test_ds)} images")

    # Extract embeddings
    embeddings, labels = extract_embeddings(model, test_loader, device)
    print(f"  Embeddings: {embeddings.shape}")

    # Run full evaluation
    results = run_full_evaluation(embeddings, labels, experiment_name)
    results["checkpoint"] = str(checkpoint_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate face recognition checkpoints on a test set"
    )
    parser.add_argument(
        "--checkpoint", nargs="+", type=Path, required=True,
        help="One or more checkpoint .pth files to evaluate",
    )
    parser.add_argument(
        "--test-dir", type=Path, required=True,
        help="Test dataset directory (folder-per-identity or flat)",
    )
    parser.add_argument(
        "--experiment-name", nargs="+", type=str, default=None,
        help="Names for each experiment (must match --checkpoint count)",
    )
    parser.add_argument("--backbone", type=str, default="convnext_atto")
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Save all results as a JSON file")
    parser.add_argument("--wandb-mode", type=str, default="disabled",
                        choices=["online", "offline", "disabled"],
                        help="W&B mode for evaluation logging")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # Validate arguments
    if not args.test_dir.exists():
        print(f"ERROR: Test directory {args.test_dir} does not exist")
        sys.exit(1)

    checkpoints = args.checkpoint
    names = args.experiment_name or [p.parent.name for p in checkpoints]
    if len(names) != len(checkpoints):
        print(f"ERROR: Got {len(checkpoints)} checkpoints but {len(names)} names")
        sys.exit(1)

    # W&B init (one run for all evaluations)
    if wandb and args.wandb_mode != "disabled":
        import os
        os.environ["WANDB_MODE"] = args.wandb_mode
        run_name = "eval-" + "-vs-".join(names)
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "knn-proj"),
            project=os.environ.get("WANDB_PROJECT", "downstream-face-rec"),
            name=run_name,
            config={
                "mode": "evaluation",
                "test_dir": str(args.test_dir),
                "experiments": names,
                "backbone": args.backbone,
            },
        )

    # Run evaluations
    all_results = []
    for ckpt_path, name in zip(checkpoints, names):
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint {ckpt_path} not found, skipping")
            continue

        results = evaluate_single(
            ckpt_path, args.test_dir, args.backbone,
            args.embedding_dim, device, name,
            args.batch_size, args.num_workers,
        )
        all_results.append(results)

        # Log to W&B
        if wandb and wandb.run:
            for key, val in results.items():
                if isinstance(val, (int, float, np.floating)):
                    wandb.summary[f"{name}/{key}"] = float(val)

    # Print summary table
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"{'Experiment':<25} {'Rank-1':>8} {'Rank-5':>8} {'TAR@1e-4':>10} {'10-fold':>8}")
        print("-" * 65)
        for r in all_results:
            name = r.get("experiment", "?")
            r1 = f"{r['rank1_accuracy']:>8.4f}" if r.get("rank1_accuracy") is not None else f"{'—':>8}"
            r5 = f"{r['rank5_accuracy']:>8.4f}" if r.get("rank5_accuracy") is not None else f"{'—':>8}"
            tar = f"{r['tar_at_far_1e4']:>10.4f}" if r.get("tar_at_far_1e4") is not None else f"{'—':>10}"
            kf = f"{r['kfold_verification_accuracy']:>8.4f}" if r.get("kfold_verification_accuracy") is not None else f"{'—':>8}"
            print(f"{name:<25} {r1} {r5} {tar} {kf}")

        print("-" * 65)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for r in all_results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, np.floating):
                    sr[k] = float(v)
                elif isinstance(v, (int, float, str, bool, type(None))):
                    sr[k] = v
                else:
                    sr[k] = str(v)
            serializable.append(sr)
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if wandb and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
