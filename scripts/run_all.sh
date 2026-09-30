#!/usr/bin/env bash
# Reproduce every analysis, figure and the supplementary document from scratch.
# Requires the real cohort in data/private/ (not distributed). Synthesizer settings: src/common.py (ORIGINAL_PARAMS).
# Outputs: results/tables, results/figures, results/supplement; synthetic data in results/synthetic_runs (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LOG=results/logs
mkdir -p "$LOG"

rm -rf results/synthetic_runs  # regenerate every synthetic dataset

# Analyses (each reuses the synthetic datasets cached by the first step that needs them).
for step in missingness utility_analysis fidelity_analysis privacy_analysis holdout_analysis distribution_tests \
            relationship_metrics reidentification_analysis identifier_comparison attack_by_anonymization; do
    echo "=== $step $(date +%H:%M) ==="
    python "src/$step.py" > "$LOG/$step.log" 2>&1
done

# Figures that combine several analyses, then the supplementary document.
for step in umap_figure descriptive_figures identifier_heatmap privacy_figure build_supplement; do
    echo "=== $step $(date +%H:%M) ==="
    python "src/$step.py" > "$LOG/$step.log" 2>&1
done
echo "=== done $(date +%H:%M) ==="
