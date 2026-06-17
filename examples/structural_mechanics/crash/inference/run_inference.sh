#!/usr/bin/env bash
set -euo pipefail

INFERENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRASH_DIR="$(cd "$INFERENCE_DIR/.." && pwd)"
DATA=/mnt/1t/mit-project/Dataset
SIMS=$DATA/simulations
GLOBAL_FEATURES_JSON=$CRASH_DIR/global_features.json
OUTPUTS_DIR=$CRASH_DIR/outputs

# --- defaults ---
RUN=""
CONFIG_NAME=""
N_SAMPLES=0   # 0 = all

# --- parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)         RUN="$2";         shift 2 ;;
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --n_samples)   N_SAMPLES="$2";   shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -z "$RUN" ]]         && { echo "Error: --run is required"         >&2; exit 1; }
[[ -z "$CONFIG_NAME" ]] && { echo "Error: --config-name is required" >&2; exit 1; }

OUTPUT_DIR=$OUTPUTS_DIR/$RUN/results

# --- collect sim dirs, capped to N_SAMPLES if set ---
TMPDIR_SIMS=$(mktemp -d)
trap 'rm -rf "$TMPDIR_SIMS"' EXIT

SIM_NAMES=()
count=0
while IFS= read -r -d '' sim_path; do
  [[ $N_SAMPLES -gt 0 && $count -ge $N_SAMPLES ]] && break
  sim_name=$(basename "$sim_path")
  ln -s "$sim_path" "$TMPDIR_SIMS/$sim_name"
  SIM_NAMES+=("$sim_name")
  (( count++ )) || true
done < <(find "$SIMS" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

echo "Running inference on ${#SIM_NAMES[@]} sample(s): ${SIM_NAMES[*]}"

# --- inference ---
PYTHONPATH="$CRASH_DIR:${PYTHONPATH:-}" python "$INFERENCE_DIR/inference.py" \
  --config-path="$CRASH_DIR/conf" \
  --config-name="$CONFIG_NAME" \
  inference.raw_data_dir_test="$TMPDIR_SIMS" \
  reader.master_csv="$DATA/metadata/bumper_beam_master_with_split.csv" \
  training.global_features_filepath="$GLOBAL_FEATURES_JSON" \
  training.ckpt_path="$OUTPUTS_DIR/$RUN/checkpoints" \
  inference.output_dir_pred="$OUTPUT_DIR" \
  datapipe.stats_dir="$OUTPUTS_DIR/$RUN/stats"

# --- per-sim: visualize + evaluate ---
echo "Visualizing and evaluating ${#SIM_NAMES[@]} sample(s)..."
for sim_name in "${SIM_NAMES[@]}"; do
  echo "  -> $sim_name"
  SIM_OUT="$OUTPUT_DIR/$sim_name"
  mkdir -p "$SIM_OUT"

  PYTHONPATH="$CRASH_DIR:${PYTHONPATH:-}" python "$INFERENCE_DIR/visualize_crash.py" "$sim_name" \
    --data-dir "$SIMS" \
    --master-csv "$DATA/metadata/bumper_beam_master_with_split.csv" \
    --out "$SIM_OUT/${sim_name}_crash.gif"

  NPZ="$OUTPUT_DIR/eval_arrays/${sim_name}.npz"
  if [[ -f "$NPZ" ]]; then
    PYTHONPATH="$CRASH_DIR:${PYTHONPATH:-}" python "$INFERENCE_DIR/evaluate_crash.py" "$NPZ" \
      --out-prefix "$SIM_OUT/$sim_name"
  else
    echo "    Warning: no eval npz at $NPZ, skipping evaluate_crash.py"
  fi
done

# --- aggregate L2 error across all sims ---
NPZ_DIR="$OUTPUT_DIR/eval_arrays"
if [[ -d "$NPZ_DIR" ]]; then
  echo "Computing aggregate L2 error..."
  PYTHONPATH="$CRASH_DIR:${PYTHONPATH:-}" python "$INFERENCE_DIR/compute_l2_error_npz.py" \
    --npz_dir "$NPZ_DIR" \
    --out "$OUTPUT_DIR/l2_error.png" \
    --csv "$OUTPUT_DIR/l2_error.csv"
else
  echo "Warning: no eval_arrays dir at $NPZ_DIR, skipping compute_l2_error_npz.py"
fi

echo "Done. Results written to $OUTPUT_DIR/"
