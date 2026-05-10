import tarfile
import os
import argparse

def extract_images(shard_path, output_dir, target_count):
    """
    Extracts a specific number of .jpg images from a WebDataset tar.gz shard.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    extracted_count = 0

    print(f"Opening shard {shard_path}...")
    
    # Open the compressed tar archive
    with tarfile.open(shard_path, "r:gz") as tar:
        for member in tar:
            # We only care about image files, skip .cls metadata files
            if member.name.endswith(".jpg"):
                # Extract the image file to the target directory
                tar.extract(member, path=output_dir)
                extracted_count += 1
                
                # Print progress every 1000 images
                if extracted_count % 1000 == 0:
                    print(f"Extracted {extracted_count} / {target_count} images...")
                    
                # Stop extraction once we reach the desired count
                if extracted_count >= target_count:
                    break

    print(f"Done! {extracted_count} clean faces are ready in '{output_dir}'.")

if __name__ == "__main__":
    # Setup command-line arguments
    parser = argparse.ArgumentParser(description="Extract a specific number of images from a tar.gz shard.")
    
    parser.add_argument("--shard-path", type=str, default="data/webface4m/webface4m-0000.tar.gz",
                        help="Path to the downloaded tar.gz shard")
    parser.add_argument("--output-dir", type=str, default="data/webface4m/clean_15k",
                        help="Directory where extracted images will be saved")
    parser.add_argument("--count", type=int, default=15000,
                        help="Number of images to extract (default: 15000)")
    
    args = parser.parse_args()
    
    # Run the extraction
    extract_images(args.shard_path, args.output_dir, args.count)