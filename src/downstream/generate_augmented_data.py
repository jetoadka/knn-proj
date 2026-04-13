from __future__ import annotations

"""Generate newspaper-style augmented face images using trained CUT model.

Runs the trained CUT generator on clean face images (WebFace4M) to produce
synthetic newspaper-style faces while preserving identity folder structure.

Usage:
    python -m src.downstream.generate_augmented_data \
        --input-dir data/webface4m_flat \
        --output-dir data/augmented_newspaper \
        --checkpoint-dir src/style_transfer/cut_model/checkpoints/exp_combined \
        --batch-size 16

    # Quick test on sample data:
    python -m src.downstream.generate_augmented_data \
        --input-dir sample_data/webface4m \
        --output-dir data/augmented_sample \
        --checkpoint-dir src/style_transfer/cut_model/checkpoints/exp_combined \
        --batch-size 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# CUT generator loader
# ---------------------------------------------------------------------------

def load_cut_generator(checkpoint_dir: Path, device: str):
    """Load the trained CUT generator network.

    The CUT model saves generator weights as ``latest_net_G.pth``
    inside its checkpoint directory.
    """
    # We need to add the CUT model path so we can import its network defs
    cut_model_path = Path(__file__).resolve().parent.parent / "style_transfer" / "cut_model"
    if str(cut_model_path) not in sys.path:
        sys.path.insert(0, str(cut_model_path))

    from models.networks import define_G  # type: ignore

    # CUT default generator args (ResNet 9-block for 112x112)
    netG = define_G(
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="resnet_9blocks",
        norm="instance",
        use_dropout=False,
        init_type="xavier",
        init_gain=0.02,
        no_antialias=False,
        no_antialias_up=False,
        gpu_ids=[],
    )

    ckpt_path = checkpoint_dir / "latest_net_G.pth"
    if not ckpt_path.exists():
        # Try numbered checkpoints
        ckpt_candidates = sorted(checkpoint_dir.glob("*_net_G.pth"))
        if ckpt_candidates:
            ckpt_path = ckpt_candidates[-1]
        else:
            print(f"ERROR: No generator checkpoint found in {checkpoint_dir}")
            sys.exit(1)

    print(f"Loading generator from {ckpt_path}")
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    
    # Handle DataParallel/DDP wrappers in checkpoints
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
            
    netG.load_state_dict(clean_state_dict)
    netG.to(device)
    netG.eval()
    return netG


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

TRANSFORM_INPUT = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a [-1, 1] tensor back to a PIL Image."""
    arr = tensor.detach().cpu().numpy()
    arr = (arr * 0.5 + 0.5) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def collect_images(input_dir: Path) -> list[Path]:
    """Recursively find all image files."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in exts)


def generate(
    generator,
    input_dir: Path,
    output_dir: Path,
    device: str,
    batch_size: int = 16,
    quality: int = 95,
):
    """Run CUT generator on all images, preserving folder structure."""
    images = collect_images(input_dir)
    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images)} images to transform")
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_paths: list[Path] = []
    batch_tensors: list[torch.Tensor] = []

    for img_path in tqdm(images, desc="Generating newspaper-style images"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM_INPUT(img)
            batch_paths.append(img_path)
            batch_tensors.append(tensor)
        except Exception as e:
            tqdm.write(f"  Skipped {img_path}: {e}")
            continue

        if len(batch_tensors) >= batch_size:
            _process_batch(generator, batch_tensors, batch_paths,
                           input_dir, output_dir, device, quality)
            batch_paths = []
            batch_tensors = []

    # Process remaining
    if batch_tensors:
        _process_batch(generator, batch_tensors, batch_paths,
                       input_dir, output_dir, device, quality)

    print(f"\nDone! Generated images saved to {output_dir}")


def _process_batch(
    generator,
    batch_tensors: list[torch.Tensor],
    batch_paths: list[Path],
    input_dir: Path,
    output_dir: Path,
    device: str,
    quality: int,
):
    """Transform a batch of images through the generator."""
    batch = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        fake_batch = generator(batch)

    for i, img_path in enumerate(batch_paths):
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fake_img = tensor_to_image(fake_batch[i])
        fake_img.save(str(out_path), "JPEG", quality=quality)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate newspaper-style augmented faces using trained CUT"
    )
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Directory of clean face images (preserves subfolder structure)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for generated newspaper-style images")
    parser.add_argument("--checkpoint-dir", type=Path, required=True,
                        help="CUT model checkpoint directory (contains *_net_G.pth)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for generation (reduce for MPS/CPU)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG quality for output images (default: 95)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (auto-detected if not set)")
    args = parser.parse_args()

    device = args.device or get_device()
    print(f"Using device: {device}")

    if not args.input_dir.exists():
        print(f"ERROR: Input directory {args.input_dir} not found")
        sys.exit(1)

    generator = load_cut_generator(args.checkpoint_dir, device)
    generate(generator, args.input_dir, args.output_dir, device,
             args.batch_size, args.quality)


if __name__ == "__main__":
    main()
