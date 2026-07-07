import os
import shutil
import argparse
from pathlib import Path
from PIL import Image

def process_and_filter_dataset(input_dir, target_size, min_size):
    """
    Filters out low-resolution images, resizes remaining images to target_size
    using high-quality Lanczos interpolation, and removes empty directories.
    """
    input_path = Path(input_dir).resolve()
    if not input_path.exists():
        print(f"ERROR: Input directory {input_path} does not exist.")
        return

    valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_files = []
    
    print(f"Scanning directory: {input_path}...")
    for ext in valid_extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))

    total_found = len(image_files)
    print(f"Found {total_found} images. Starting processing (Target: {target_size}x{target_size}, Min: {min_size}px)...")

    deleted_count = 0
    processed_count = 0
    error_count = 0

    # Process images (Delete low-res, resize others)
    for idx, img_path in enumerate(image_files, 1):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                
                # Check if image is below the minimum required resolution
                if width < min_size or height < min_size:
                    img_path.unlink()  # Delete the low-res image file
                    deleted_count += 1
                    continue
                
                # If resolution is valid, check if resizing is needed
                if (width, height) != (target_size, target_size):
                    # Convert to RGB to handle grayscale or RGBA safely
                    rgb_img = img.convert("RGB")
                    # Lanczos resampling provides the best biometrical quality for down/upsampling
                    resized_img = rgb_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
                    resized_img.save(img_path, quality=95)
                
                processed_count += 1

        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
            error_count += 1

        # Print progress every 5000 images
        if idx % 5000 == 0 or idx == total_found:
            print(f"Progress: [{idx}/{total_found}] | Processed: {processed_count} | Deleted: {deleted_count} | Errors: {error_count}")

    # Clean up empty directories from bottom to top (bottom-up traversal)
    print("\nCleaning up empty directories...")
    removed_dirs = 0
    for root, dirs, files in os.walk(input_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                # If the directory is completely empty, delete it
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)
                    removed_dirs += 1
            except Exception as e:
                print(f"Could not remove directory {dir_path}: {e}")

    print("\n--- DATASET PROCESSING SUMMARY ---")
    print(f"Total images scanned:  {total_found}")
    print(f"Successfully resized:  {processed_count}")
    print(f"Deleted (below {min_size}px): {deleted_count}")
    print(f"Removed empty folders: {removed_dirs}")
    print(f"Corrupted/Errors:      {error_count}")
    print("----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter low-res images, resize to target biometric resolution, and clean empty folders.")
    parser.add_argument("--input_dir", required=True, help="Path to the aligned_filtered dataset directory.")
    parser.add_argument("--target_size", type=int, choices=[112, 256], required=True, help="Target resolution: 112 or 256.")
    parser.add_argument("--min_size", type=int, default=100, help="Minimum width/height threshold to keep an image (default: 100).")
    
    args = parser.parse_args()
    
    # Auto-adjust default min_size if user chose 256 and didn't override min_size
    if args.target_size == 256 and args.min_size == 100:
        print("Notice: Target is 256px, automatically adjusting minimum size threshold to 200px.")
        args.min_size = 200

    process_and_filter_dataset(args.input_dir, args.target_size, args.min_size)