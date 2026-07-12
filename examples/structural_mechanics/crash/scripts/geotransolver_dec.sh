#!/bin/bash

# Resolve the absolute path to the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Navigate to the crash workspace directory (one level up from scripts)
cd "$SCRIPT_DIR/.."
PROJ_DIR="$(pwd)"

# ==============================================================================
# CONFIGURATION BLOCK
# Edit these parameters to match your target experiment.
# ==============================================================================

# 1. Hydra Experiment Configuration Name
#    DEC ablation arms (all bumper, identical data/splits — pick one):
#      bumper_geotransolver_oneshot_dec_local.yaml  DEC -> per-node fx stream only
#      bumper_geotransolver_oneshot_dec_ctx.yaml    DEC -> geometry context only
#      bumper_geotransolver_oneshot_dec_both.yaml   DEC -> both streams
#      bumper_geotransolver_oneshot.yaml            baseline (no DEC)
CONFIG_NAME="bumper_geotransolver_oneshot_dec_local.yaml"
EXP_NAME="geotransolver_dec"

# 2. Base Paths (no need to change this)
DATA_DIR="/mnt/1t/mit-project/Dataset"
META_DIR="${DATA_DIR}/metadata"
OUT_DIR="./outputs/${EXP_NAME}/"

# 3. Config Paths
SIM_DIR="${DATA_DIR}/simulations"
GLOBAL_FEATURES="${PROJ_DIR}/global_features.json"
MASTER_CSV="${META_DIR}/bumper_beam_master_with_split.csv"

# 4. Dataset Size Overrides (Matches the size of your downloaded data)
NUM_TRAIN_SAMPLES=20
NUM_VAL_SAMPLES=3

# 5. Training Epoch Count Override
EPOCHS=1000

# 6. WandB project (kept separate from the shared MIT-InitialRuns / full-car projects)
WANDB_PROJECT="dec-bumper-ablation"

# ==============================================================================
# RUN TIME EXECUTION
# ==============================================================================

# Generate global features if not already present
if [ ! -f "$GLOBAL_FEATURES" ]; then
    echo "Global features file not found at $GLOBAL_FEATURES. Generating it from $MASTER_CSV..."
    python make_global_features.py --master-csv "$MASTER_CSV" --out "$GLOBAL_FEATURES"
fi

# Launch training with configuration overrides
HYDRA_FULL_ERROR=1 python train.py --config-name="$CONFIG_NAME" \
    training.raw_data_dir="$SIM_DIR" \
    training.raw_data_dir_validation="$SIM_DIR" \
    training.global_features_filepath="$GLOBAL_FEATURES" \
    inference.raw_data_dir_test="$SIM_DIR" \
    reader.master_csv="$MASTER_CSV" \
    training.num_training_samples="$NUM_TRAIN_SAMPLES" \
    training.num_validation_samples="$NUM_VAL_SAMPLES" \
    training.wandb_project="$WANDB_PROJECT" \
    hydra.run.dir="$OUT_DIR" \
    training.epochs="$EPOCHS"
