#!/usr/bin/env bash
# Reproduce every revision analysis from scratch (synthesizer settings: common.ORIGINAL_PARAMS).
# Real data (clean_data.csv) and all outputs stay local; revision/outputs/ is gitignored.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
LOG=revision/outputs/logs
mkdir -p "$LOG"

rm -rf revision/outputs/synth  # regenerate every synthetic dataset with the original settings

for step in missingness utility_analysis fidelity_analysis privacy_analysis holdout_analysis \
            distribution_tests relationship_metrics reidentification_analysis identifier_comparison; do
    echo "=== $step $(date +%H:%M) ==="
    python "revision/$step.py" > "$LOG/$step.log" 2>&1
done
echo "=== done $(date +%H:%M) ==="
