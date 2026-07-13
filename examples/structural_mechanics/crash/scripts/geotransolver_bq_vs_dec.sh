#!/bin/bash
# BQ-vs-DEC ablation on the bumper beam: three GeoTransolver arms trained
# sequentially with identical data, splits, seed, epochs, and optimizer.
#
#   arm 1  bq_baseline   ball queries only            (bumper_geotransolver_oneshot)
#   arm 2  dec_nobq      DEC only, ball queries OFF   (bumper_geotransolver_oneshot_dec_nobq)
#   arm 3  dec_bq        ball queries + DEC           (bumper_geotransolver_oneshot_dec_both)
#
# Run a single arm with:  ./geotransolver_bq_vs_dec.sh dec_nobq

set -e

# Resolve the absolute path to the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Navigate to the crash workspace directory (one level up from scripts)
cd "$SCRIPT_DIR/.."
PROJ_DIR="$(pwd)"

# ==============================================================================
# CONFIGURATION BLOCK
# Edit these parameters to match your target experiment.
# ==============================================================================

# 1. Base Paths (no need to change this)
DATA_DIR="/mnt/1t/mit-project/Dataset"
META_DIR="${DATA_DIR}/metadata"

# 2. Config Paths
SIM_DIR="${DATA_DIR}/simulations"
GLOBAL_FEATURES="${PROJ_DIR}/global_features.json"
MASTER_CSV="${META_DIR}/bumper_beam_master_with_split.csv"

# 3. Shared settings — identical across all arms for comparability
NUM_TRAIN_SAMPLES=20
NUM_VAL_SAMPLES=3
EPOCHS=1000
SEED=42
WANDB_PROJECT="dec-bumper-ablation"

# 4. Arms: <exp_name>:<hydra config>
ARMS=(
    "bq_baseline:bumper_geotransolver_oneshot.yaml"
    "dec_nobq:bumper_geotransolver_oneshot_dec_nobq.yaml"
    "dec_bq:bumper_geotransolver_oneshot_dec_both.yaml"
)

# ==============================================================================
# RUN TIME EXECUTION
# ==============================================================================

# Generate global features if not already present
if [ ! -f "$GLOBAL_FEATURES" ]; then
    echo "Global features file not found at $GLOBAL_FEATURES. Generating it from $MASTER_CSV..."
    python make_global_features.py --master-csv "$MASTER_CSV" --out "$GLOBAL_FEATURES"
fi

run_arm() {
    local exp_name="$1" config_name="$2"
    echo ""
    echo "=============================================================="
    echo " ARM: ${exp_name}  (${config_name})"
    echo "=============================================================="
    HYDRA_FULL_ERROR=1 python train.py --config-name="$config_name" \
        training.raw_data_dir="$SIM_DIR" \
        training.raw_data_dir_validation="$SIM_DIR" \
        training.global_features_filepath="$GLOBAL_FEATURES" \
        inference.raw_data_dir_test="$SIM_DIR" \
        reader.master_csv="$MASTER_CSV" \
        training.num_training_samples="$NUM_TRAIN_SAMPLES" \
        training.num_validation_samples="$NUM_VAL_SAMPLES" \
        training.epochs="$EPOCHS" \
        training.seed="$SEED" \
        training.wandb_project="$WANDB_PROJECT" \
        hydra.run.dir="./outputs/${exp_name}/"
}

for arm in "${ARMS[@]}"; do
    exp_name="${arm%%:*}"
    config_name="${arm#*:}"
    # optional CLI filter: run only the named arm
    if [ -n "$1" ] && [ "$1" != "$exp_name" ]; then
        continue
    fi
    run_arm "$exp_name" "$config_name"
done
