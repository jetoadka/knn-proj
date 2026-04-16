from __future__ import annotations

"""Fine-tune a face recognition model on prepared datasets with W&B tracking.

Supports training on folder-per-identity datasets using timm backbones
with ArcFace/CosFace/AdaFace loss heads. Automatically detects CUDA/MPS/CPU.

Usage:
    # Train baseline (E1) on clean data
    python -m src.downstream.train_downstream --config configs/baseline_clean.yaml

    # Train with CLI overrides
    python -m src.downstream.train_downstream \
        --config configs/augmented_newspaper.yaml \
        --batch-size 64 --device mps

    # Quick smoke test on sample_data (Mac MPS)
    python -m src.downstream.train_downstream \
        --train-dir sample_data/webface4m \
        --val-dir sample_data/wiki_face_112 \
        --backbone convnext_atto \
        --epochs 2 --batch-size 8 --device mps \
        --run-name smoke-test --wandb-mode offline
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError:
    print("ERROR: timm not installed. Run: pip install timm>=1.0.0")
    sys.exit(1)

try:
    import wandb
except ImportError:
    wandb = None
    print("WARNING: wandb not installed. Logging disabled. Run: pip install wandb")

try:
    import yaml
except ImportError:
    yaml = None


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
# Dataset
# ============================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class FolderFaceDataset(Dataset):
    """Face dataset supporting both nested and flat directory structures.

    Nested (folder-per-identity):
        root/
        ├── identity_0/
        │   ├── img1.jpg
        │   └── img2.jpg
        ...

    Flat (WebFace4M style: IDENTITY_SEQNUM.jpg):
        root/
        ├── 008633_018.jpg   (identity = 008633)
        ├── 008633_019.jpg
        ├── 013254_017.jpg   (identity = 013254)
        ...

    Auto-detects which layout is present.
    """

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform

        # Detect layout: check if there are subdirectories with images
        identity_dirs = sorted(
            d for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        self.samples: list[tuple[Path, int]] = []

        if identity_dirs:
            # --- Nested layout (folder-per-identity) ---
            self.class_to_idx = {d.name: idx for idx, d in enumerate(identity_dirs)}
            for d in identity_dirs:
                label = self.class_to_idx[d.name]
                for img in sorted(d.rglob("*")):
                    if img.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((img, label))
        else:
            # --- Flat layout (IDENTITY_SEQ.jpg) ---
            flat_images = sorted(
                f for f in self.root.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
            )
            identity_set: dict[str, int] = {}
            for img in flat_images:
                # Identity is everything before the last underscore
                parts = img.stem.rsplit("_", 1)
                identity = parts[0] if len(parts) > 1 else img.stem
                if identity not in identity_set:
                    identity_set[identity] = len(identity_set)
                self.samples.append((img, identity_set[identity]))
            self.class_to_idx = identity_set

        self.n_classes = len(self.class_to_idx)

        if not self.samples:
            print(f"WARNING: No images found in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
# Model: Backbone + Classification Head
# ============================================================================

class ArcFaceHead(nn.Module):
    """ArcFace / CosFace / AdaFace classification head."""

    def __init__(self, embedding_dim: int, n_classes: int, loss_type: str = "cosface",
                 s: float = 64.0, m: float = 0.4):
        super().__init__()
        self.loss_type = loss_type
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(n_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None):
        # Normalize
        normed_emb = F.normalize(embeddings, dim=1)
        normed_w = F.normalize(self.weight, dim=1)

        # Cosine similarity
        cosine = F.linear(normed_emb, normed_w)

        if labels is None:
            return cosine

        # Apply margin
        if self.loss_type == "cosface":
            one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
            logits = cosine - one_hot * self.m
        elif self.loss_type == "arcface":
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
            logits = torch.cos(theta + one_hot * self.m)
        else:
            logits = cosine

        return logits * self.s


class FaceRecModel(nn.Module):
    """Face recognition model: timm backbone + ArcFace/CosFace head."""

    def __init__(self, backbone_name: str, n_classes: int, loss_type: str = "cosface",
                 embedding_dim: int = 512, backbone_kwargs: dict | None = None):
        super().__init__()

        bk = backbone_kwargs or {}
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,  # Remove classifier, get features
            **bk,
        )

        # Get actual feature dim from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 112, 112)
            feat_dim = self.backbone(dummy).shape[1]

        # Project to embedding space
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        self.head = ArcFaceHead(embedding_dim, n_classes, loss_type)
        self.embedding_dim = embedding_dim

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized embedding (for inference/eval)."""
        features = self.backbone(x)
        emb = self.projection(features)
        return F.normalize(emb, dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        features = self.backbone(x)
        emb = self.projection(features)
        logits = self.head(emb, labels)
        return logits, emb


# ============================================================================
# Training
# ============================================================================

class CosineSchedule:
    """Cosine annealing with linear warmup."""

    def __init__(self, base_lr: float, total_steps: int, warmup_frac: float = 0.05):
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_frac)

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.base_lr * step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train_one_epoch(
    model: FaceRecModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineSchedule,
    device: torch.device,
    epoch: int,
    global_step: int,
    log_interval: int = 50,
) -> tuple[float, int]:
    """Train for one epoch, return (avg_loss, updated_global_step)."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Update learning rate
        lr = scheduler.get_lr(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward
        logits, embeddings = model(images, labels)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1
        global_step += 1

        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

        if wandb and wandb.run and global_step % log_interval == 0:
            wandb.log({
                "train/loss": loss_val,
                "train/lr": lr,
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/epoch": epoch,
            }, step=global_step)

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: FaceRecModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model using embedding-based rank-1 accuracy.

    Instead of computing classification accuracy against training class
    indices (which is meaningless when the validation set has different
    identities), this extracts normalized embeddings and computes rank-1
    identification accuracy using a gallery/probe split.
    """
    model.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Evaluating", dynamic_ncols=True):
        images = images.to(device)
        emb = model.get_embedding(images)
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Normalize embeddings
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8
    all_embeddings = all_embeddings / norms

    # Split into gallery (first image per identity) and probe (rest)
    unique_labels = np.unique(all_labels)
    gallery_idx = []
    probe_idx = []
    for label in unique_labels:
        indices = np.where(all_labels == label)[0]
        gallery_idx.append(indices[0])
        if len(indices) > 1:
            probe_idx.extend(indices[1:].tolist())

    if not probe_idx:
        # Not enough images per identity for rank-based evaluation;
        # fall back to a simple nearest-neighbour leave-one-out accuracy
        sim = all_embeddings @ all_embeddings.T
        np.fill_diagonal(sim, -1.0)  # exclude self-match
        top1 = np.argmax(sim, axis=1)
        correct = np.sum(all_labels[top1] == all_labels)
        accuracy = correct / len(all_labels)
    else:
        gallery_idx = np.array(gallery_idx)
        probe_idx = np.array(probe_idx)
        sim = all_embeddings[probe_idx] @ all_embeddings[gallery_idx].T
        top1 = np.argmax(sim, axis=1)
        top1_labels = all_labels[gallery_idx[top1]]
        correct = np.sum(top1_labels == all_labels[probe_idx])
        accuracy = correct / len(probe_idx)

    return {
        "accuracy": accuracy,
        "embeddings": torch.from_numpy(all_embeddings),
        "labels": torch.from_numpy(all_labels),
    }


# ============================================================================
# Main
# ============================================================================

def load_config(config_path: str | None) -> dict:
    """Load YAML config file if provided."""
    if config_path is None or yaml is None:
        return {}
    path = Path(config_path)
    if not path.exists():
        print(f"WARNING: Config {path} not found, using CLI args only")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Train face recognition model")

    # Data
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument("--train-dir", type=Path, help="Training data directory")
    parser.add_argument("--val-dir", type=Path, help="Validation data directory")

    # Model
    parser.add_argument("--backbone", type=str, default="convnext_atto",
                        help="timm backbone name")
    parser.add_argument("--loss", type=str, default="cosface",
                        choices=["cosface", "arcface", "adaface"],
                        help="Loss function for face recognition")
    parser.add_argument("--embedding-dim", type=int, default=512)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--num-workers", type=int, default=4)

    # Infrastructure
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="debug")
    parser.add_argument("--wandb-mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--eval-interval", type=int, default=5,
                        help="Evaluate every N epochs")

    args = parser.parse_args()

    # Merge config file with CLI args (CLI takes precedence)
    config = load_config(args.config)
    for key, val in config.items():
        cli_key = key.replace("-", "_")
        if hasattr(args, cli_key) and getattr(args, cli_key) == parser.get_default(cli_key):
            setattr(args, cli_key, val)

    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}")
    print(f"Loss: {args.loss}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    transform_train = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dir = Path(args.train_dir) if args.train_dir else None
    val_dir = Path(args.val_dir) if args.val_dir else None

    if train_dir is None:
        print("ERROR: --train-dir is required")
        sys.exit(1)

    train_ds = FolderFaceDataset(train_dir, transform=transform_train)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if val_dir and val_dir.exists():
        val_ds = FolderFaceDataset(val_dir, transform=transform_val)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    n_classes = train_ds.n_classes
    print(f"Train: {len(train_ds)} images, {n_classes} classes")
    if val_loader:
        print(f"Val:   {len(val_ds)} images, {val_ds.n_classes} classes")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    # Note: we use pretrained defaults — don't override patch_size etc.
    # timm-face uses custom patch_size=2 only when training from scratch.
    backbone_kwargs = {}
    if "vit" in args.backbone:
        backbone_kwargs["img_size"] = 112

    model = FaceRecModel(
        backbone_name=args.backbone,
        n_classes=n_classes,
        loss_type=args.loss,
        embedding_dim=args.embedding_dim,
        backbone_kwargs=backbone_kwargs,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = CosineSchedule(args.lr, total_steps)

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if wandb and args.wandb_mode != "disabled":
        os.environ["WANDB_MODE"] = args.wandb_mode
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "knn-proj"),
            project=os.environ.get("WANDB_PROJECT", "downstream-face-rec"),
            name=args.run_name,
            config={
                "backbone": args.backbone,
                "loss": args.loss,
                "embedding_dim": args.embedding_dim,
                "n_classes": n_classes,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "total_steps": total_steps,
                "train_dir": str(train_dir),
                "val_dir": str(val_dir) if val_dir else None,
                "device": str(device),
                "total_params": total_params,
            },
        )
        wandb.watch(model, log="gradients", log_freq=200)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    save_dir = args.save_dir / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_val_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs, {total_steps} steps")
    print(f"Checkpoints: {save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, epoch, global_step,
        )

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{args.epochs} — loss: {avg_loss:.4f}, time: {epoch_time:.1f}s")

        # Evaluate
        if val_loader and epoch % args.eval_interval == 0:
            val_results = evaluate(model, val_loader, device)
            val_acc = val_results["accuracy"]
            print(f"  Val accuracy: {val_acc:.4f}")

            if wandb and wandb.run:
                wandb.log({
                    "val/accuracy": val_acc,
                    "val/epoch": epoch,
                }, step=global_step)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_path = save_dir / "best_model.pth"
                config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
                config_dict["n_classes"] = n_classes
                torch.save({
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "config": config_dict,
                }, ckpt_path)
                print(f"  New best model saved: {ckpt_path} (acc={val_acc:.4f})")

                if wandb and wandb.run:
                    artifact = wandb.Artifact(
                        f"face-rec-{args.run_name}", type="model",
                        metadata={"epoch": epoch, "val_acc": val_acc},
                    )
                    artifact.add_file(str(ckpt_path))
                    wandb.log_artifact(artifact)

        # Save periodic checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = save_dir / f"epoch_{epoch}.pth"
            config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            config_dict["n_classes"] = n_classes
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config_dict,
            }, ckpt_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Checkpoints in:   {save_dir}")
    print(f"{'='*60}")

    if wandb and wandb.run:
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.finish()


if __name__ == "__main__":
    main()
