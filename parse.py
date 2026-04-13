import json

with open('notebooks/downstream_colab.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if "results saved to" in src.lower() or "shutil" in src.lower() or "drive_results" in src.lower():
            print(f"--- Cell {i} ---")
            print(src)

