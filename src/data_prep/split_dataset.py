import os
import json
import random
import shutil
from pathlib import Path

def split_dataset(source_dir, output_dir, train_ratio=0.80, val_ratio=0.10, seed=42):
    # Set random seed for reproducibility
    random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Group images by their immediate parent directory (Identity/Subject)
    # This ensures subject-disjoint split
    identity_folders = {}
    
    print("Scanning directory structure and grouping by identity...")
    for file_path in source_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg']:
            # Get relative folder path as identity ID (e.g., 'cuni_fsv/id_123')
            rel_folder = str(file_path.parent.relative_to(source_path))
            
            if rel_folder not in identity_folders:
                identity_folders[rel_folder] = []
            identity_folders[rel_folder].append(file_path)
            
    all_identities = list(identity_folders.keys())
    total_identities = len(all_identities)
    
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
    
    # Copy files and build metadata
    metadata = {
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": round(1.0 - train_ratio - val_ratio, 2)},
        "total_identities": total_identities,
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
        print(f"[{split_name.upper()}] Folders: {len(folders)} | Images: {split_img_count}")

    # Save metadata
    metadata_file = output_path / "dataset_split_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nSuccess! Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    SOURCE_DIR = r"../../../aligned_faces_filtered"
    OUTPUT_DIR = r"../../../dataset_splitted"
    
    # 80% train, 10% val, 10% test (automatically calculated as remainder)
    split_dataset(SOURCE_DIR, OUTPUT_DIR, train_ratio=0.80, val_ratio=0.10)