#!/bin/bash

# Exit immediately if any command fails
set -e

# Resolve the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Output directory for comparison results
OUTPUT_DIR="$SCRIPT_DIR/outputs/comparisons"

echo "========================================================================"
echo " Starting Models Comparative Evaluation"
echo "========================================================================"

# Run the comparison script
# Loads .npz predictions from:
#   1. outputs/baseline/npz_files/
#   2. outputs/stats_only/npz_files/
#   3. outputs/enhanced/npz_files/
# Computes average L2 curves and saves them to outputs/comparisons/
python "$SCRIPT_DIR/inference/compare_models.py" \
    --model_dirs \
        "$SCRIPT_DIR/outputs/baseline" \
        "$SCRIPT_DIR/outputs/stats_only" \
        "$SCRIPT_DIR/outputs/enhanced" \
    --labels \
        "Baseline (XYZ Only)" \
        "Stats-Only Encoder" \
        "Enhanced (Stats + Part ID)" \
    --output_dir "$OUTPUT_DIR"

echo "========================================================================"
echo " Comparison finished successfully!"
echo "   Plot & Text summary saved in: $OUTPUT_DIR"
echo "========================================================================"
