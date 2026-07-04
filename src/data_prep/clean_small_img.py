from pathlib import Path
from PIL import Image


def remove_small_images(folder_path, min_width=100, min_height=100):
    """Recursively scans a directory and removes images smaller than the specified dimensions."""
    folder = Path(folder_path).resolve()

    # Verify that the target directory exists
    if not folder.exists():
        print(f"ERROR: Directory {folder} does not exist!")
        return

    print(f"    Inspecting images in {folder}...")

    # Include both lowercase and uppercase extensions for cross-platform compatibility
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    image_files = []
    for ext in extensions:
        # rglob recursively searches through all subdirectories
        image_files.extend(folder.rglob(ext))

    removed_count = 0
    for img_path in image_files:
        try:
            # Image.open() only reads the file header (metadata) to get dimensions,
            # which is significantly faster than loading the entire pixel data into memory
            with Image.open(img_path) as img:
                width, height = img.size

            # Remove the file if either dimension is below the threshold
            if width < min_width or height < min_height:
                print(
                    f"Deleting low-res image ({width}x{height}):"
                    f" {img_path.name}"
                )
                img_path.unlink()  # Permanently deletes the file from disk
                removed_count += 1
        except Exception as e:
            print(f"ERROR: Reading {img_path.name}: {e}")

    print(f"Done! Removed {removed_count} low-quality images.")


if __name__ == "__main__":
    remove_small_images(
        "../style_transfer/data/trainB", min_width=100, min_height=100
    )