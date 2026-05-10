# Evaluation: Metrics & Experiment Orchestration

Tools for evaluating face recognition models and orchestrating the full
experiment pipeline on the PeopleGator newspaper-face test set.

## Evaluation Methodology

Each trained model is assessed in two independent ways:

### 1. Own metrics (`metrics.py`)
Computed directly from model embeddings on the **PeopleGator test set**
(1,399 images, 119 identities):

| Metric | Description |
|---|---|
| **Rank-1 / Rank-5** | Closed-set identification accuracy at top-1 / top-5 |
| **TAR@FAR=1e-4** | True Accept Rate at False Accept Rate = 0.01% |
| **10-Fold Verification** | Cross-validated same/different pair accuracy (LFW protocol) |

Results saved to `results/<experiment>.json`.

### 2. Teacher retrieval evaluation
Embeddings are extracted from all PeopleGator images and indexed as a
retrieval database. The teacher's `run.py` + `evaluate.py` scripts query
245 probe images and compute:

| Metric | Description |
|---|---|
| **Precision@k** | Fraction of top-k results with correct identity |
| **Recall@k** | Fraction of all relevant results found in top-k |
| **MAP@k** | Mean Average Precision |
| **MRR@k** | Mean Reciprocal Rank |
| **NDCG@k** | Normalized Discounted Cumulative Gain |

Evaluated at k ∈ {1, 5, 10, 50}. Results saved to
`eval_output/<experiment>/retrieval_metrics.csv`.

> **Note:** The PeopleGator test set (119 identities) is strictly held out.
> Validation during training uses `wiki_face_112` only to prevent data leakage.

## Scripts

### `run_all_experiments.sh` — Full orchestration
Trains and evaluates all experiments sequentially. Supports filtering.

```bash
# Run all experiments
bash src/evaluation/scripts/run_all_experiments.sh

# Run specific experiments only
EXPERIMENTS=E0-dense-baseline,E4a-cut-only-combined \
bash src/evaluation/scripts/run_all_experiments.sh

# Override dataset paths
DENSE_2500=/path/to/clean WIKI_FACE=/path/to/wiki \
bash src/evaluation/scripts/run_all_experiments.sh
```

**Experiments included** (12 total):

| ID | Clean data | CUT data | Total | Purpose |
|---|---|---|---|---|
| `E0-dense-baseline` | dense-2500 (2,500 id) | — | 72k | Baseline |
| `E0-doubled` | dense-2500-doubled (5,000 id) | — | 144k | Volume/identity control |
| `E1-combined` | dense-2500 | combined (4 styles) | 144k | Clean + all CUT |
| `E1-people` | dense-2500 | CUT people | 144k | Clean + PG people |
| `E1-pg-colored` | dense-2500 | CUT pg-colored | 144k | Clean + colored PG |
| `E1-pg-grayscale` | dense-2500 | CUT pg-grayscale | 144k | Clean + grayscale PG |
| `E1-wiki` | dense-2500 | CUT wiki | 144k | Clean + WikiFace |
| `E4a-cut-only-combined` | — | combined (4 styles) | 72k | All CUT, no clean |
| `E4b-cut-only-people` | — | CUT people | 72k | PG people only |
| `E4c-cut-only-pgcolored` | — | CUT pg-colored | 72k | Colored PG only |
| `E4d-cut-only-pggrayscale` | — | CUT pg-grayscale | 72k | Grayscale PG only |
| `E4e-cut-only-wiki` | — | CUT wiki | 72k | WikiFace only |

CUT `fake_B` directories are **auto-discovered** from
`../style_transfer/cut_model/results/` — no manual path configuration needed.

---

### `run_eval_only.sh` — Evaluate without re-training
Runs embedding extraction + teacher evaluation on an already-trained model.

```bash
# Evaluate a single experiment
EXPERIMENT=E0-dense-baseline bash src/evaluation/scripts/run_eval_only.sh

# Evaluate all checkpoints found in checkpoints/
bash src/evaluation/scripts/run_eval_only.sh
```

---

### `run_pipeline.sh` — Single experiment pipeline
Low-level script for one training + evaluation run. Called internally by
`run_all_experiments.sh`. Can be used directly for custom configurations.

```bash
bash src/evaluation/scripts/run_pipeline.sh \
    --experiment-name my-exp \
    --clean-dir ../dataset_clean_dense_2500/testA \
    --cut-dir ../style_transfer/cut_model/results/exp_pg_colored_200/fake_B \
    --epochs 30
```

---

## Standalone Usage

### Evaluate a single trained model
```bash
python -m src.evaluation.metrics \
    --model-checkpoint checkpoints/E0-dense-baseline/best_model.pth \
    --eval-dir ../people_gator_test_faces \
    --backbone convnext_atto \
    --experiment-name E0-dense-baseline \
    --output results/E0-dense-baseline.json
```

### Compare all completed experiments
```bash
# Print comparison table to stdout
python -m src.evaluation.compare_experiments

# Save as Markdown
python -m src.evaluation.compare_experiments \
    --markdown results/comparison.md \
    --output results/comparison.json
```

### Extract PeopleGator embeddings only
```bash
python -m src.evaluation.extract_pg_embeddings \
    --checkpoint checkpoints/E0-dense-baseline/best_model.pth \
    --pg-data ../people_gator__data_export \
    --output-dir eval_output/E0-dense-baseline
```

## Key Results Summary

| Experiment | P@1 | NDCG@10 | Rank-1 | TAR@1e-4 |
|---|---|---|---|---|
| E0-dense-baseline | 0.718 | 0.591 | 0.234 | 0.199 |
| **E0-doubled** | **0.788** | **0.650** | **0.337** | 0.137 |
| E1-combined | 0.771 | 0.638 | 0.335 | **0.223** |
| E1-pg-colored | 0.784 | 0.661 | 0.302 | 0.169 |
| **E4a-cut-only-combined** | **0.788** | **0.650** | 0.327 | 0.170 |
| E4e-cut-only-wiki | 0.698 | 0.568 | 0.241 | 0.150 |

Best P@1 = **0.788** (+7.0 pp vs baseline), achieved by E0-doubled and
E4a-cut-only-combined via different mechanisms (identity diversity vs. CUT style diversity).

## Files

| File | Purpose |
|---|---|
| `metrics.py` | Rank-1/5, TAR@FAR, 10-fold verification computation |
| `extract_pg_embeddings.py` | Extract and save PeopleGator face embeddings |
| `compare_experiments.py` | Load results from `results/` and print comparison table |
| `generate_ground_truth.py` | Build ground-truth JSONL from PeopleGator annotations |
| `identity_mapping.py` | Map PeopleGator identity labels for retrieval evaluation |
| `scripts/run_all_experiments.sh` | **Main entry point** — train + evaluate all experiments |
| `scripts/run_eval_only.sh` | Evaluate existing checkpoints without re-training |
| `scripts/run_pipeline.sh` | Single-experiment training + evaluation pipeline |
