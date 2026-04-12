# Evaluation: Metrics & Comparison

This module provides evaluation tools for comparing face recognition experiments.

## Metrics Computed

| Metric | Description | Standard Use |
|--------|-------------|-------------|
| **Rank-1 Accuracy** | % of probes where the top-1 most similar gallery image has the correct identity | Face identification |
| **Rank-5 Accuracy** | % of probes where the correct identity is in top-5 matches | Face identification |
| **TAR@FAR=1e-4** | True Accept Rate at False Accept Rate = 0.01% | IJB-B/C benchmark |
| **FAR/FRR** | False Accept Rate / False Reject Rate across thresholds | Biometric system tuning |
| **10-Fold Verification** | Cross-validated accuracy for same/different pair classification | Standard in LFW |

## Usage

### Evaluate a single model

```bash
python -m src.evaluation.metrics \
    --model-checkpoint checkpoints/E2-augmented-newspaper/best_model.pth \
    --eval-dir data/people_gator/aligned_112/test \
    --backbone convnext_atto \
    --experiment-name E2-augmented \
    --output results/E2_results.json
```

### Compare all experiments

```bash
# Fetch from W&B and print comparison table
python -m src.evaluation.compare_experiments

# Save as Markdown for the report
python -m src.evaluation.compare_experiments \
    --markdown results/comparison.md \
    --output results/comparison.json
```

## Files

| File | Purpose |
|------|---------|
| `metrics.py` | Rank-1/K, FAR/FRR, TAR@FAR, 10-fold verification |
| `compare_experiments.py` | Pull W&B results, generate comparison tables |
