import argparse
import os
from pathlib import Path
import shutil


def copy_images_flat(source_dir, dest_dir):
    """Recursively finds all images in source_dir and copies them to dest_dir with unique filenames."""
    source_path = Path(source_dir).resolve()
    dest_path = Path(dest_dir).resolve()

    # Verify that the source directory exists
    if not source_path.exists():
        print(f"ERROR: Source directory {source_path} does not exist.")
        return

    # Create the destination directory if it doesn't exist yet
    dest_path.mkdir(parents=True, exist_ok=True)

    print(
        f"  Searching for images in {source_path}... (this might take a while for large datasets)"
    )

    # Include both lowercase and uppercase extensions for Linux compatibility (case-sensitive OS)
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    image_files = []
    for ext in extensions:
        # rglob performs a recursive search through all subdirectories
        image_files.extend(source_path.rglob(ext))

    total_files = len(image_files)
    print(f"Found {total_files} images. Starting copy to {dest_dir}...")

    count = 0
    for img_path in image_files:
        # Prefix the filename with its parent folder name to prevent overwriting files with identical names
        parent_name = img_path.parent.name
        unique_name = f"{parent_name}_{img_path.name}"
        dest_file = dest_path / unique_name

        try:
            shutil.copy2(img_path, dest_file)
            count += 1
            # Print progress every 1000 images to monitor long-running operations
            if count % 1000 == 0:
                print(f"    Copied: {count}/{total_files}...")
        except Exception as e:
            print(f"ERROR: Failed to copy {img_path} to {dest_file}: {e}")

    print(f"Done! Copied {count} images from {source_dir} to {dest_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CUT dataset by copying images from multiple directories"
            " into a flat structure."
        )
    )
    # Arguments are optional (default=None) so they can be run independently
    parser.add_argument(
        "--clean",
        nargs="+",
        default=None,
        help="Paths to the clean images directory (trainA)",
    )
    parser.add_argument(
        "--noisy",
        nargs="+",
        default=None,
        help="Paths to the noisy images directory (trainB)",
    )
    parser.add_argument(
        "--output",
        default="cut_dataset",
        help="Output root directory for the prepared dataset",
    )
    args = parser.parse_args()

    # Ensure at least one input argument (--clean or --noisy) is provided
    if not args.clean and not args.noisy:
        parser.error(
            "You must specify at least one argument: --clean or --noisy."
        )

    print("Preparing CUT dataset...")

    # Process clean images (trainA) if specified
    if args.clean:
        for clean_dir in args.clean:
            copy_images_flat(clean_dir, os.path.join(args.output, "trainA"))

    # Process noisy images (trainB) if specified
    if args.noisy:
        for noisy_dir in args.noisy:
            copy_images_flat(noisy_dir, os.path.join(args.output, "trainB"))

    print("Dataset preparation completed.")