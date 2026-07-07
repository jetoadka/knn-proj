import os
import json
import random
import shutil
import argparse
from pathlib import Path

def split_dataset(source_dir, output_dir, train_ratio=0.80, val_ratio=0.10, seed=42):
    """
    Splits dataset into train/val/test sets by identity (subject-disjoint split)
    to ensure the same person never appears in both train and test sets.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    if not source_path.exists():
        print(f"ERROR: Source directory {source_path} does not exist.")
        return

    # Group images by their immediate parent directory (Identity/Subject)
    # This ensures subject-disjoint split
    identity_folders = {}
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    print(f"Scanning directory structure: {source_path}...")
    for file_path in source_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            # Get relative folder path as identity ID (e.g., 'cuni_fsv/id_123')
            rel_folder = str(file_path.parent.relative_to(source_path))
            
            if rel_folder not in identity_folders:
                identity_folders[rel_folder] = []
            identity_folders[rel_folder].append(file_path)
            
    all_identities = list(identity_folders.keys())
    total_identities = len(all_identities)
    
    if total_identities == 0:
        print("ERROR: No valid identity folders with images found!")
        return

    print(f"Found {total_identities} unique identity folders.")
    
    # Shuffle identities randomly
    random.shuffle(all_identities)
    
    # Calculate split indices
    train_end = int(total_identities * train_ratio)
    val_end = train_end + int(total_identities * val_ratio)
    
    splits = {
        "train": all_identities[:train_end],
        "val": all_identities[train_end:val_end],
        "test": all_identities[val_end:]
    }
    
    test_ratio = round(1.0 - train_ratio - val_ratio, 2)
    
    # Copy files and build metadata
    metadata = {
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "total_identities": total_identities,
        "seed": seed,
        "stats": {},
        "identities_map": {}
    }
    
    print("Copying files to train/val/test directories...")
    for split_name, folders in splits.items():
        split_img_count = 0
        metadata["identities_map"][split_name] = folders
        
        for folder in folders:
            images = identity_folders[folder]
            split_img_count += len(images)
            
            for img_path in images:
                # Maintain relative structure inside train/val/test
                rel_path = img_path.relative_to(source_path)
                dest_path = output_path / split_name / rel_path
                
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_path)
                
        # Record stats for metadata
        metadata["stats"][split_name] = {
            "folders_count": len(folders),
            "images_count": split_img_count
        }
        print(f"[{split_name.upper()}] Folders (Identities): {len(folders)} | Images: {split_img_count}")

    # Save metadata
    metadata_file = output_path / "dataset_split_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSuccess! Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test sets by identity/subject.")
    
    # Required arguments
    parser.add_argument("--input_dir", required=True, help="Path to the source directory (e.g., aligned_faces_filtered).")
    parser.add_argument("--output_dir", required=True, help="Path where train/val/test folders will be created.")
    
    # Optional arguments with defaults
    parser.add_argument("--train_ratio", type=float, default=0.80, help="Proportion of identities for training (default: 0.80).")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Proportion of identities for validation (default: 0.10).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    
    args = parser.parse_args()
    
    # Validation check for ratios
    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("The sum of --train_ratio and --val_ratio must be strictly less than 1.0 to leave room for the test set!")
        
    split_dataset(
        source_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )