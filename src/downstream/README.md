# Downstream: Face Recognition Fine-tuning & Evaluation

Fine-tune a **ConvNeXt-Atto + CosFace** face recognition model on WebFace4M
clean images and/or CUT-augmented newspaper-style counterparts, then evaluate
on held-out newspaper faces from the PeopleGator dataset.

## Pipeline Overview

```
WebFace4M (HuggingFace)
    │
extract_webface_subset.py        ← download balanced subset by identity
    │
dataset_clean_dense_2500/testA   (2,500 ids × ~29 img — SOURCE domain)
    │
    ├── [optional] CUT fake_B/   (same 2,500 ids, newspaper-styled)
    │
prepare_timm_dataset.py          ← merge clean + CUT into folder-per-identity
    │
data/downstream/<experiment>/train/
    │
train_downstream.py              ← ConvNeXt-Atto + CosFace, 30 epochs
    │                               validated on wiki_face_112 (early stopping)
checkpoints/<experiment>/best_model.pth
    │
    ├── evaluate.py              ← own metrics (Rank-1/5, TAR@FAR)
    │       results/<experiment>.json
    │
    └── extract_pg_embeddings.py ← extract PeopleGator embeddings
            eval_output/<experiment>/
                image_embeddings.npy
                image_paths.txt
                ground_truth.jsonl
                    │
                teacher's run.py + evaluate.py
                    │
                eval_output/<experiment>/retrieval_metrics.csv
                (Precision@k, Recall@k, MAP, MRR, NDCG)
```

## Quick Start

### Run a single experiment end-to-end:
```bash
cd knn-proj

# Baseline (clean data only):
EXPERIMENTS=E0-dense-baseline bash src/evaluation/scripts/run_all_experiments.sh

# Specific CUT variant:
EXPERIMENTS=E4a-cut-only-combined bash src/evaluation/scripts/run_all_experiments.sh

# Evaluate an already-trained model without re-training:
EXPERIMENT=E0-dense-baseline bash src/evaluation/scripts/run_eval_only.sh
```

### Run all experiments sequentially:
```bash
bash src/evaluation/scripts/run_all_experiments.sh
```

### Prepare the E0-doubled volume-control dataset (one-time, ~2.5 GB download):
```bash
python3 -m src.downstream.extract_webface_subset \
    --output-dir ../dataset_clean_dense_2500_doubled \
    --n-identities 5000 \
    --exclude-ids-file data/dense2500_excluded_ids.txt \
    --max-images-per-id 29 \
    --n-shards 31
```

## Experiment Series

All experiments use the same 2,500-identity `dataset_clean_dense_2500` as the
clean source. CUT fake\_B directories are auto-discovered from
`../style_transfer/cut_model/results/`.

| Experiment | Clean data | CUT data | Total imgs | Purpose |
|---|---|---|---|---|
| **E0-dense-baseline** | dense-2500 (2,500 id) | — | ~72k | Baseline: no augmentation |
| **E0-doubled** | dense-2500-doubled (5,000 id) | — | ~144k | Volume/identity-diversity control |
| **E1-combined** | dense-2500 | combined (4 styles) | ~144k | All CUT styles + clean |
| **E1-people** | dense-2500 | CUT people | ~144k | People Gator style + clean |
| **E1-pg-colored** | dense-2500 | CUT pg-colored | ~144k | Colored PG style + clean |
| **E1-pg-grayscale** | dense-2500 | CUT pg-grayscale | ~144k | Grayscale PG style + clean |
| **E1-wiki** | dense-2500 | CUT wiki | ~144k | WikiFace style + clean |
| **E4a-cut-only-combined** | — | combined (4 styles) | ~72k | All CUT styles, NO clean |
| **E4b-cut-only-people** | — | CUT people | ~72k | People Gator style only |
| **E4c-cut-only-pgcolored** | — | CUT pg-colored | ~72k | Colored PG style only |
| **E4d-cut-only-pggrayscale** | — | CUT pg-grayscale | ~72k | Grayscale PG style only |
| **E4e-cut-only-wiki** | — | CUT wiki | ~72k | WikiFace style only |

### Key findings (from completed runs)

- All augmented/extended strategies achieve **P@1 = 0.74–0.79** vs baseline 0.72
- Best P@1 = **0.788** shared by: E0-doubled, E4a-cut-only-combined, E1-pg-colored
- **Style diversity > single domain**: E4a (4 CUT styles) beats all single-style variants
- **Color fidelity matters**: pg-colored consistently outperforms pg-grayscale
- **Identity diversity ≈ CUT augmentation**: E0-doubled (5k clean ids) matches E4a

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Backbone | `convnext_atto` | patch\_size=4 for fine-tuning |
| Loss | CosFace | Subtractive margin |
| Embedding dim | 512 | |
| LR | 5e-5 | Cosine annealing schedule |
| LR schedule | Cosine annealing | Decay to 0 over 30 epochs |
| Weight decay | 0.1 | |
| Batch size | 64 | |
| Precision | bfloat16 | CUDA only |
| Epochs | 30 | |
| Eval interval | 5 epochs | |
| Early stopping | patience=5 | Based on wiki\_face\_112 val accuracy |
| Validation set | wiki\_face\_112 | 1,538 ids, 3,223 images — separate from PG test |

## Files

| File | Purpose |
|---|---|
| `train_downstream.py` | Training loop with W&B logging, CosFace, early stopping |
| `prepare_timm_dataset.py` | Merge clean + CUT data into folder-per-identity layout |
| `extract_webface_subset.py` | Download balanced WebFace4M subset by identity ID |
| `evaluate.py` | Standalone evaluation: Rank-1/5, TAR@FAR, 10-fold verification |
| `create_webface_val.py` | Build a held-out WebFace4M validation split |

## Environment Variables

All scripts respect these overrides:

| Variable | Default | Description |
|---|---|---|
| `DENSE_2500` | `../dataset_clean_dense_2500/testA` | Primary clean data source |
| `DENSE_E0` | `../dataset_clean_dense_2500_doubled` | E0-doubled clean data |
| `CUT_OUTPUT_DIR` | `../style_transfer/cut_model/results` | Root for CUT fake\_B dirs |
| `WIKI_FACE` | `../wiki_face_112_fin` | Validation dataset |
| `EPOCHS` | `30` | Training epochs |
| `LR` | `5e-5` | Learning rate |
| `EXPERIMENTS` | *(all)* | Comma-separated filter, e.g. `E0-dense-baseline,E4a-cut-only-combined` |
