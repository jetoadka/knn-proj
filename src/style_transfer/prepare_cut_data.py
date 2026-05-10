import os
import shutil
import argparse
from pathlib import Path


def copy_images_flat(source_dir, dest_dir):
    source_path = Path(source_dir).resolve()
    dest_path = Path(dest_dir).resolve()

    if not source_path.exists():
        print(f"ERROR: Source directory {source_path} does not exist.")
        return

    dest_path.mkdir(parents=True, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_files = []
    for ext in extensions:
        image_files.extend(source_path.rglob(ext))

    count = 0
    for img_path in image_files:
        parent_name = img_path.parent.name
        unique_name = f"{parent_name}_{img_path.name}"
        dest_file = dest_path / unique_name

        try:
            shutil.copy2(img_path, dest_file)
            count += 1
        except Exception as e:
            print(f"ERROR: copying {img_path} to {dest_file}: {e}")

    print(f"Copied {count} images from {source_dir} to {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CUT dataset by copying images to a flat structure.")
    parser.add_argument("--clean", nargs='+', required=True, help="Paths to the clean images directory (trainA)")
    parser.add_argument("--noisy", nargs='+', required=True, help="Paths to the noisy images directory (trainB)")
    parser.add_argument("--output", default="cut_dataset", help="Output directory for images")
    args = parser.parse_args()
    
    print("Preparing CUT dataset...")
    for clean_dir in args.clean:
        copy_images_flat(clean_dir, os.path.join(args.output, "trainA"))
    for noisy_dir in args.noisy:
        copy_images_flat(noisy_dir, os.path.join(args.output, "trainB"))
    print("Dataset preparation completed.")