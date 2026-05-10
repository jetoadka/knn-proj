#!/usr/bin/env bash
# ============================================================================
# run_all_experiments.sh — Run baseline + all CUT generator experiments
#
# This script runs run_pipeline.sh once for baseline (clean only) and
# once for each CUT-augmented dataset found in the augmented data directory.
#
# CUT datasets stored as .zip files are automatically extracted before use.
#
# Usage:
#   # Run all experiments (smoke test)
#   SMOKE_TEST=1 bash src/evaluation/scripts/run_all_experiments.sh
#
#   # Run all experiments with real data on GPU
#   DEVICE=cuda \
#   CLEAN_DATA=../CUT_datast/test-15k \
#   CUT_ZIP_DIR=../CUT_datast \
#   EPOCHS=100 \
#   bash src/evaluation/scripts/run_all_experiments.sh
#
# IMPORTANT: Run from the repository root (knn-proj/)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="${SCRIPT_DIR}/run_pipeline.sh"

# ── Configuration ────────────────────────────────────────────────────────────
# Directory containing CUT experiment .zip files and extracted folders
CUT_ZIP_DIR="${CUT_ZIP_DIR:-../CUT_datast}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        KNN Face Recognition — Multi-Experiment Runner       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# 0. EXTRACT CUT DATASETS (if needed)
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 0: Extracting CUT experiment datasets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Define experiment names and their zip files
declare -A CUT_EXPERIMENTS=(
    ["E1-wiki100"]="exp_wiki_100"
    ["E2-combined200"]="exp_combined_200"
    ["E3-people200"]="exp_people_200"
    ["E4-pg-colored200"]="exp_pg_colored_200"
    ["E5-pg-grayscale200"]="exp_pg_grayscale_200"
)

for EXP_NAME in "${!CUT_EXPERIMENTS[@]}"; do
    FOLDER_NAME="${CUT_EXPERIMENTS[$EXP_NAME]}"
    ZIP_FILE="${CUT_ZIP_DIR}/${FOLDER_NAME}.zip"
    EXTRACT_DIR="${CUT_ZIP_DIR}/${FOLDER_NAME}"

    if [[ -d "${EXTRACT_DIR}" ]]; then
        echo "  ✓ ${FOLDER_NAME} already extracted"
    elif [[ -f "${ZIP_FILE}" ]]; then
        echo "  ⏳ Extracting ${ZIP_FILE}..."
        unzip -q "${ZIP_FILE}" -d "${EXTRACT_DIR}"
        echo "  ✓ ${FOLDER_NAME} extracted"
    else
        echo "  ⚠ ${ZIP_FILE} not found — skipping ${EXP_NAME}"
    fi
done
echo ""

# ============================================================================
# 1. Locate fake_B directories for each CUT experiment
# ============================================================================
find_fake_b() {
    local base_dir="$1"
    local result
    result="$(find "${base_dir}" -type d -name "fake_B" 2>/dev/null | head -1)"
    if [[ -n "${result}" ]]; then
        echo "${result}"
    else
        echo ""
    fi
}

# ============================================================================
# 2. BASELINE (no augmentation)
# ============================================================================
echo "▶ Running experiment: E0-baseline (clean data only)..."
echo ""

EXPERIMENT="E0-baseline" CUT_DATA="" bash "${PIPELINE}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# 3. CUT GENERATOR EXPERIMENTS
# ============================================================================
for EXP_NAME in $(echo "${!CUT_EXPERIMENTS[@]}" | tr ' ' '\n' | sort); do
    FOLDER_NAME="${CUT_EXPERIMENTS[$EXP_NAME]}"
    EXTRACT_DIR="${CUT_ZIP_DIR}/${FOLDER_NAME}"

    if [[ ! -d "${EXTRACT_DIR}" ]]; then
        echo "⚠ Skipping ${EXP_NAME}: ${EXTRACT_DIR} not found"
        continue
    fi

    FAKE_B_DIR="$(find_fake_b "${EXTRACT_DIR}")"

    if [[ -z "${FAKE_B_DIR}" ]]; then
        echo "⚠ Skipping ${EXP_NAME}: no fake_B directory found in ${EXTRACT_DIR}"
        continue
    fi

    echo "▶ Running experiment: ${EXP_NAME} (${FAKE_B_DIR})..."
    echo ""

    EXPERIMENT="${EXP_NAME}" CUT_DATA="${FAKE_B_DIR}" bash "${PIPELINE}"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
done

# ============================================================================
# 4. COMPARE ALL RESULTS
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  COMPARISON OF ALL EXPERIMENTS              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

RESULTS_DIR="${RESULTS_DIR:-results}"

if [[ -d "${RESULTS_DIR}" ]] && ls "${RESULTS_DIR}"/*.json &>/dev/null; then
    echo "Results files:"
    for f in "${RESULTS_DIR}"/*.json; do
        echo "  - $(basename "${f}")"
    done
    echo ""

    # Merge all JSON results into a single comparison
    if command -v python3 &>/dev/null; then
        python3 -c "
import json, glob, sys

files = sorted(glob.glob('${RESULTS_DIR}/*.json'))
all_results = []
for f in files:
    with open(f) as fh:
        data = json.load(fh)
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)

if not all_results:
    print('  No results found.')
    sys.exit(0)

# Print comparison table
header = f\"{'Experiment':<25} {'Rank-1':>8} {'Rank-5':>8} {'TAR@1e-4':>10} {'10-fold':>8}\"
print(header)
print('-' * len(header))
for r in all_results:
    name = r.get('experiment', '?')[:24]
    r1 = f\"{r['rank1_accuracy']:>8.4f}\" if r.get('rank1_accuracy') is not None else f\"{'—':>8}\"
    r5 = f\"{r['rank5_accuracy']:>8.4f}\" if r.get('rank5_accuracy') is not None else f\"{'—':>8}\"
    tar = f\"{r['tar_at_far_1e4']:>10.4f}\" if r.get('tar_at_far_1e4') is not None else f\"{'—':>10}\"
    kf = f\"{r['kfold_verification_accuracy']:>8.4f}\" if r.get('kfold_verification_accuracy') is not None else f\"{'—':>8}\"
    print(f'{name:<25} {r1} {r5} {tar} {kf}')
print()

# Save combined results
combined_path = '${RESULTS_DIR}/all_experiments.json'
with open(combined_path, 'w') as fh:
    json.dump(all_results, fh, indent=2)
print(f'Combined results saved to {combined_path}')
"
    fi
else
    echo "  No results found in ${RESULTS_DIR}/"
fi

# ============================================================================
# 5. COMPARE RETRIEVAL RESULTS
# ============================================================================
EVAL_DIR="eval_output"
if [[ -d "${EVAL_DIR}" ]]; then
    echo ""
    echo "Retrieval evaluation results:"
    for csv_file in "${EVAL_DIR}"/*/retrieval_metrics.csv; do
        if [[ -f "${csv_file}" ]]; then
            exp_name="$(basename "$(dirname "${csv_file}")")"
            echo ""
            echo "  ${exp_name}:"
            cat "${csv_file}" | sed 's/^/    /'
        fi
    done
fi
