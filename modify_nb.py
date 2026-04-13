import json

with open('notebooks/downstream_colab.ipynb', 'r') as f:
    nb = json.load(f)

# Modify cell 11 (Directories)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if "CKPT_DIR = Path(DRIVE_BASE) /" in src:
            new_src = """import os
from pathlib import Path

# Set up directories
# We use local Colab storage for fast I/O during extraction and training
DATA_DIR = Path("/content/knn-proj/data")
CKPT_DIR = Path("/content/knn-proj/checkpoints")

if USE_GOOGLE_DRIVE:
    print(f" Using Google Drive for final backup: {DRIVE_BASE}")

DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Symlink so scripts find data at standard paths
for link_path, target in [(Path("data"), DATA_DIR), (Path("checkpoints"), CKPT_DIR)]:
    if not link_path.exists():
        os.symlink(str(target), str(link_path))

print(f" Data: {DATA_DIR}")
print(f" Checkpoints: {CKPT_DIR}")
"""     
            # convert to list of strings
            lines = new_src.splitlines(True)
            cell['source'] = lines

        if "drive_results = Path(DRIVE_BASE) /" in src:
            new_src = """if USE_GOOGLE_DRIVE:
    import shutil
    drive_results = Path(DRIVE_BASE) / "results"
    drive_results.mkdir(parents=True, exist_ok=True)
    for f in Path("results").glob("*"):
        shutil.copy2(f, drive_results / f.name)
        print(f" {f.name}")
    print(f"\\nResults saved to {drive_results}")

    # Export (save) downstream models to Google Drive
    drive_checkpoints = Path(DRIVE_BASE) / "checkpoints"
    drive_checkpoints.mkdir(parents=True, exist_ok=True)
    print("\\nBacking up checkpoints to Google Drive...")
    !cp -r {CKPT_DIR}/* {drive_checkpoints}/
    print(f" Models exported to {drive_checkpoints}")
else:
    print(" Download results before disconnecting!")
    from google.colab import files
    if Path("results/final_comparison.json").exists():
        files.download("results/final_comparison.json")
"""
            lines = new_src.splitlines(True)
            cell['source'] = lines

with open('notebooks/downstream_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
    # add trailing newline
    f.write("\n")

