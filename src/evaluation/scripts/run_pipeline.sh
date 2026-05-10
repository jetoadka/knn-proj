#!/usr/bin/env bash
# ============================================================================
# run_pipeline.sh — Full face recognition experiment pipeline
#
# Runs the complete pipeline in sequence:
#   1. Prepare training dataset (merge clean + augmented sources)
#   2. Train face recognition model (with W&B logging)
#   3. Evaluate on held-out test set (own metrics: Rank-1, TAR@FAR)
#   4. Extract embeddings for teacher's PeopleGator retrieval evaluation
#   5. Run teacher's retrieval metrics (MAP, NDCG, Precision@k)
#
# Usage:
#   # Run baseline (no augmentation, sample data, smoke test)
#   SMOKE_TEST=1 bash src/evaluation/scripts/run_pipeline.sh
#
#   # Run with a CUT generator on GPU
#   DEVICE=cuda CUT_DATA=../CUT_datast/exp_wiki_100/test_100/images/fake_B \
#   bash src/evaluation/scripts/run_pipeline.sh
#
# IMPORTANT: Run from the repository root (knn-proj/)
# ============================================================================
set -euo pipefail

# ── Resolve project root (this script lives in src/evaluation/scripts/) ──────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          KNN Face Recognition — Experiment Pipeline         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Project root: ${PROJECT_ROOT}"

# ============================================================================
# 0. Configuration — edit these or override via environment variables
# ============================================================================

# ── Experiment name ──────────────────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-E0-baseline}"

# ── Smoke test mode ─────────────────────────────────────────────────────────
# Set SMOKE_TEST=1 to run a quick 5-epoch test with reduced data
SMOKE_TEST="${SMOKE_TEST:-0}"

# ── Data paths ───────────────────────────────────────────────────────────────
# Clean face source — WebFace4M test-15k subset (15,000 images).
# These are the SOURCE images that CUT was applied to.
# The fake_B directories contain the same 15k images transformed into
# newspaper style, preserving identity (same filenames → same identities).
CLEAN_DATA="${CLEAN_DATA:-../CUT_datast/test-15k}"

# CUT-augmented data — SET THIS to your CUT generator output directory
# Examples:
#   CUT_DATA=../CUT_datast/exp_wiki_100/test_100/images/fake_B
#   CUT_DATA=../CUT_datast/exp_combined_200/test_200/images/fake_B
#   CUT_DATA=""  ← baseline (no augmentation)
CUT_DATA="${CUT_DATA:-}"

# Validation data (used during training for early stopping)
# This MUST be people_gator dev split, not a tiny sample!
VAL_DATA="${VAL_DATA:-../people_gator__data_export/people_gator__data}"

# Test data + annotations (used for final evaluation)
TEST_DATA="${TEST_DATA:-../people_gator__data_export/people_gator__data}"
TEST_ANNOTATIONS="${TEST_ANNOTATIONS:-../people_gator__data_export/people_gator__corresponding_faces__2026-02-11.test.jsonl}"

# Teacher's evaluation data
QUERY_FILE="${QUERY_FILE:-../image_queries.union.tst.jsonl}"
GT_FILE="${GT_FILE:-${TEST_ANNOTATIONS}}"
PG_EVAL_DIR="${PG_EVAL_DIR:-../PeopleGatorNamedFaces-merge_implementations}"

# ── Model settings ───────────────────────────────────────────────────────────
BACKBONE="${BACKBONE:-convnext_atto}"
LOSS="${LOSS:-cosface}"
EMBEDDING_DIM="${EMBEDDING_DIM:-512}"

# ── Training hyperparameters (aligned with timm-face reference) ──────────────
if [[ "${SMOKE_TEST}" == "1" ]]; then
    EPOCHS="${EPOCHS:-5}"
else
    EPOCHS="${EPOCHS:-100}"
fi
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

# ── Infrastructure ───────────────────────────────────────────────────────────
DEVICE="${DEVICE:-cpu}"
WANDB_MODE_SETTING="${WANDB_MODE_SETTING:-online}"

# ── Output directories ──────────────────────────────────────────────────────
PREPARED_DIR="${PREPARED_DIR:-data/downstream/${EXPERIMENT}/train}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-results}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-eval_output/${EXPERIMENT}}"

# ============================================================================
# 0.1 Load W&B API key from .env
# ============================================================================
ENV_FILE="${PROJECT_ROOT}/src/evaluation/.env"

if [[ -f "${ENV_FILE}" ]]; then
    echo "Loading W&B API key from ${ENV_FILE}"
    # Read W_B_API from .env — handles both W_B_API="..." and W_B_API = "..."
    W_B_API="$(grep -E '^W_B_API' "${ENV_FILE}" | sed 's/.*=\s*"\(.*\)"/\1/' | tr -d '[:space:]')"
    if [[ -n "${W_B_API}" ]]; then
        export WANDB_API_KEY="${W_B_API}"
        echo "  ✓ WANDB_API_KEY set"
    else
        echo "  ⚠ W_B_API not found in .env, W&B may prompt for login"
    fi
else
    echo "⚠ No .env found at ${ENV_FILE}"
    echo "  W&B will use existing login or prompt for API key"
fi

# ============================================================================
# 0.2 Print experiment configuration
# ============================================================================
echo ""
echo "┌──────────────────────────────────────────────────────────────┐"
echo "│ Experiment: ${EXPERIMENT}"
if [[ "${SMOKE_TEST}" == "1" ]]; then
echo "│ *** SMOKE TEST MODE (${EPOCHS} epochs) ***"
fi
echo "│"
echo "│ Data:"
echo "│   Clean source:  ${CLEAN_DATA}"
if [[ -n "${CUT_DATA}" ]]; then
echo "│   CUT augmented: ${CUT_DATA}"
else
echo "│   CUT augmented: (none — baseline)"
fi
echo "│   Validation:    ${VAL_DATA}"
echo "│   Test:          ${TEST_DATA}"
echo "│"
echo "│ Model:  ${BACKBONE} + ${LOSS} (emb=${EMBEDDING_DIM})"
echo "│ Train:  ${EPOCHS} epochs, bs=${BATCH_SIZE}, lr=${LR}, wd=${WEIGHT_DECAY}"
echo "│ Device: ${DEVICE} | AMP: ${AMP_DTYPE} | W&B: ${WANDB_MODE_SETTING}"
echo "└──────────────────────────────────────────────────────────────┘"
echo ""

# ============================================================================
# 1. PREPARE TRAINING DATASET
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1/5: Preparing training dataset"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Build the list of data sources
SOURCES=("${CLEAN_DATA}")
if [[ -n "${CUT_DATA}" ]]; then
    if [[ -d "${CUT_DATA}" ]]; then
        SOURCES+=("${CUT_DATA}")
        echo "  Merging: ${CLEAN_DATA} + ${CUT_DATA}"
    else
        echo "  ERROR: CUT_DATA directory not found: ${CUT_DATA}"
        exit 1
    fi
else
    echo "  Baseline mode: using only clean data"
fi

python3 -m src.downstream.prepare_timm_dataset \
    --sources "${SOURCES[@]}" \
    --output-dir "${PREPARED_DIR}" \
    --mode train

echo ""
echo "  ✓ Dataset ready: ${PREPARED_DIR}"
echo ""

# ============================================================================
# 2. TRAIN
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2/5: Training face recognition model"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -m src.downstream.train_downstream \
    --train-dir "${PREPARED_DIR}" \
    --val-dir "${VAL_DATA}" \
    --backbone "${BACKBONE}" \
    --loss "${LOSS}" \
    --embedding-dim "${EMBEDDING_DIM}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --num-workers "${NUM_WORKERS}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --amp-dtype "${AMP_DTYPE}" \
    --device "${DEVICE}" \
    --run-name "${EXPERIMENT}" \
    --save-dir "${CHECKPOINT_DIR}" \
    --wandb-mode "${WANDB_MODE_SETTING}"

echo ""
echo "  ✓ Training complete"
echo ""

# ============================================================================
# 3. EVALUATE (own metrics)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3/5: Evaluating on test set (own metrics)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

BEST_CHECKPOINT="${CHECKPOINT_DIR}/${EXPERIMENT}/best_model.pth"
RESULTS_FILE="${RESULTS_DIR}/${EXPERIMENT}.json"

if [[ ! -f "${BEST_CHECKPOINT}" ]]; then
    echo "  ERROR: Checkpoint not found: ${BEST_CHECKPOINT}"
    echo "  Training may have failed or no val improvement was recorded."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

python3 -m src.evaluation.metrics \
    --model-checkpoint "${BEST_CHECKPOINT}" \
    --eval-dir "${TEST_DATA}" \
    --backbone "${BACKBONE}" \
    --embedding-dim "${EMBEDDING_DIM}" \
    --experiment-name "${EXPERIMENT}" \
    --annotations-jsonl "${TEST_ANNOTATIONS}" \
    --output "${RESULTS_FILE}" \
    --device "${DEVICE}"

echo ""
echo "  ✓ Own evaluation complete"
echo ""

# ============================================================================
# 4. EXTRACT EMBEDDINGS FOR TEACHER'S EVALUATION
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4/5: Extracting PeopleGator embeddings"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p "${EVAL_OUTPUT_DIR}"

python3 -m src.evaluation.extract_pg_embeddings \
    --checkpoint "${BEST_CHECKPOINT}" \
    --data-dir "${TEST_DATA}" \
    --annotations "${TEST_ANNOTATIONS}" \
    --backbone "${BACKBONE}" \
    --embedding-dim "${EMBEDDING_DIM}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --device "${DEVICE}"

echo ""
echo "  ✓ Embeddings extracted to ${EVAL_OUTPUT_DIR}"
echo ""

# ============================================================================
# 5. TEACHER'S RETRIEVAL EVALUATION
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 5/5: Running teacher's retrieval evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [[ -f "${QUERY_FILE}" ]] && [[ -d "${PG_EVAL_DIR}" ]]; then
    # Run retrieval
    PYTHONPATH="${PG_EVAL_DIR}:${PYTHONPATH:-}" \
    python3 -m peoplegator_namedfaces.retrieval.run \
        --dataset "${EVAL_OUTPUT_DIR}/dataset_config.json" \
        --queries "${QUERY_FILE}" \
        --engine "${EVAL_OUTPUT_DIR}/engine_config.json" \
        --output "${EVAL_OUTPUT_DIR}/results.pkl"

    # Run evaluation
    PYTHONPATH="${PG_EVAL_DIR}:${PYTHONPATH:-}" \
    python3 -m peoplegator_namedfaces.retrieval.evaluate \
        --predictions "${EVAL_OUTPUT_DIR}/results.pkl" \
        --ground-truth "${GT_FILE}" \
        --dataset "${EVAL_OUTPUT_DIR}/dataset_config.json" \
        --top-k 1 5 10 50 \
        --output-file "${EVAL_OUTPUT_DIR}/retrieval_metrics.csv"

    echo ""
    echo "  ✓ Retrieval evaluation complete"
    echo "  Results: ${EVAL_OUTPUT_DIR}/retrieval_metrics.csv"
else
    echo "  ⚠ Skipping teacher's evaluation:"
    [[ ! -f "${QUERY_FILE}" ]] && echo "    - Query file not found: ${QUERY_FILE}"
    [[ ! -d "${PG_EVAL_DIR}" ]] && echo "    - PeopleGator eval dir not found: ${PG_EVAL_DIR}"
fi

echo ""

# ============================================================================
# 6. SUMMARY
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    EXPERIMENT COMPLETE                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Experiment:  ${EXPERIMENT}"
echo "  Checkpoint:  ${BEST_CHECKPOINT}"
echo "  Own metrics: ${RESULTS_FILE}"
echo "  Retrieval:   ${EVAL_OUTPUT_DIR}/retrieval_metrics.csv"
echo ""

# Print results if jq is available, otherwise cat the JSON
if command -v jq &>/dev/null; then
    jq '.[0] | {
        experiment,
        rank1_accuracy,
        rank5_accuracy,
        "tar_at_far_1e-4": .tar_at_far_1e4,
        kfold_verification_accuracy
    }' "${RESULTS_FILE}" 2>/dev/null || cat "${RESULTS_FILE}"
else
    cat "${RESULTS_FILE}"
fi

echo ""
echo "Done. To compare multiple experiments:"
echo "  python3 -m src.evaluation.compare_experiments"
