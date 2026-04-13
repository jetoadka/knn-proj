import json

with open('notebooks/downstream_colab.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        if "CKPT_DIR =" in src:
            print(f"--- Cell {i} ---")
            print(src)

