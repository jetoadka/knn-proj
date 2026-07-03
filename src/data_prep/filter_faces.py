import shutil
from pathlib import Path

def filter_faces(source_dir, dest_dir, min_confidence=0.80):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Iterate over all files recursively
    for file_path in source_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg']:
            
            name_part = file_path.stem
            
            if 'confidence_' in name_part:
                try:
                    # Extract confidence value from the filename
                    conf_str = name_part.split('confidence_')[-1]
                    confidence = float(conf_str)
                except ValueError:
                    continue
                
                # Filter based on threshold
                if confidence >= min_confidence:
                    relative_path = file_path.relative_to(source_path)
                    new_dest = dest_path / relative_path
                    
                    # Create subdirectories if they don't exist
                    new_dest.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(file_path, new_dest)
                    print(f"Copied: {relative_path} (conf: {confidence})")

if __name__ == "__main__":
    SOURCE = r"../../../aligned_faces" 
    DESTINATION = r"../../../aligned_faces_filtered"
    
    print("Starting filtering process...")
    filter_faces(SOURCE, DESTINATION)
    print("Filtering complete!")