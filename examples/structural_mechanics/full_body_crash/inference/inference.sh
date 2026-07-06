#!/bin/bash

# Exit immediately if any command fails
set -e

# Validate arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <raw_test_data_dir> <encoder_type> [wandb_run_id]"
    echo "  encoder_type choices: baseline, stats_only, enhanced, dec"
    echo "Example: $0 /mnt/1t/mit-project/Dataset/full_body enhanced"
    exit 1
fi

RAW_DATA_ROOT="$1"
ENCODER="$2"
WANDB_RUN_ID="$3"

# Shift parameters to capture any remaining custom arguments (e.g. --train_sims, --val_sim)
if [ -n "$WANDB_RUN_ID" ]; then
    shift 3
else
    shift 2
fi

# Validate encoder choices
if [ "$ENCODER" != "baseline" ] && [ "$ENCODER" != "stats_only" ] && [ "$ENCODER" != "enhanced" ] && [ "$ENCODER" != "dec" ]; then
    echo "Error: encoder_type must be one of: baseline, stats_only, enhanced, dec"
    exit 1
fi

# Resolve absolute path of the raw data directory
ABS_RAW_DATA_ROOT=$(cd "$RAW_DATA_ROOT" && pwd)

# Get the directory of this script to call python scripts correctly
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Resolve outputs and results directories
ENCODER_ROOT="$SCRIPT_DIR/../outputs/$ENCODER"
NPZ_DIR="$ENCODER_ROOT/npz_files"
RESULTS_DIR="$ENCODER_ROOT/results"

mkdir -p "$NPZ_DIR"
mkdir -p "$RESULTS_DIR"

echo "========================================================================"
echo " Pipeline Initialization"
echo "   Raw Test Data Dir : $ABS_RAW_DATA_ROOT"
echo "   Encoder Type      : $ENCODER"
echo "   Encoder Output Dir: $ENCODER_ROOT"
if [ -n "$WANDB_RUN_ID" ]; then
    echo "   Resuming WandB Run: $WANDB_RUN_ID"
fi
echo "========================================================================"

# Step 1: Run rollouts and save .npz files
echo ""
echo "[Step 1/3] Running inference rollouts and generating NPZ files..."
python "$SCRIPT_DIR/run_inference.py" \
    --data_root "$ABS_RAW_DATA_ROOT" \
    --encoder "$ENCODER" \
    --output_dir "$SCRIPT_DIR/../outputs" \
    "$@"

# Step 2: Compute quantitative metrics
echo ""
echo "[Step 2/3] Computing metrics..."
python "$SCRIPT_DIR/compute_metrics.py" \
    --data_dir "$NPZ_DIR" \
    --output_dir "$RESULTS_DIR" \
    ${WANDB_RUN_ID:+--wandb_run_id "$WANDB_RUN_ID"}

# Step 3: Generate animated crash trajectory GIFs and upload to WandB
echo ""
echo "[Step 3/3] Generating trajectory visualizations..."
python "$SCRIPT_DIR/visualize_predictions.py" \
    --data_dir "$NPZ_DIR" \
    --output_dir "$RESULTS_DIR" \
    ${WANDB_RUN_ID:+--wandb_run_id "$WANDB_RUN_ID"}

echo ""
echo "========================================================================"
echo " Inference pipeline completed successfully!"
echo "   NPZ Outputs : $NPZ_DIR"
echo "   Results     : $RESULTS_DIR"
echo "========================================================================"
