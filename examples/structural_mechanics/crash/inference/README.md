# Inference

`run_inference.sh` runs a three-phase pipeline: model inference, per-simulation visualization and evaluation, and aggregate L2 error reporting.

## Usage

```bash
./run_inference.sh --run <model> --config-name <config> [--n_samples <N>]
```

| Flag | Required | Description |
|------|----------|-------------|
| `--run` | yes | Name of the trained run (must match the folder under `outputs/`) |
| `--config-name` | yes | Hydra config name (file under `conf/`, without `.yaml`) |
| `--n_samples` | no | Number of simulations to process (default: all) |

## Example

```bash
./run_inference.sh --run transolver --config-name bumper_transolver_oneshot --n_samples 5
```

## What the script does

### 1. Collect simulations

Scans `/mnt/1t/mit-project/Dataset/simulations/` for subdirectories (sorted), takes the first `N` (or all if `--n_samples` is omitted), and symlinks them into a temporary directory. The temp directory is cleaned up automatically on exit.

### 2. Run inference (`inference.py`)

Runs `inference.py` via Hydra with the given config. Key overrides passed on the command line:

- `inference.raw_data_dir_test` — the temp dir of symlinked simulations
- `reader.master_csv` — `Dataset/metadata/bumper_beam_master_with_split.csv`
- `training.global_features_filepath` — `crash/global_features.json`
- `training.ckpt_path` — `outputs/<run>/checkpoints/`
- `inference.output_dir_pred` — `outputs/<run>/results/`
- `datapipe.stats_dir` — `outputs/<run>/stats/`

Inference writes prediction files and saves evaluation arrays to `outputs/<run>/results/eval_arrays/<sim_name>.npz`.

### 3. Per-simulation: visualize + evaluate

For each simulation:

- **`visualize_crash.py`** — reads the ground-truth simulation from `Dataset/simulations/<sim_name>/` and the master CSV, then writes an animated GIF to `outputs/<run>/results/<sim_name>/<sim_name>_crash.gif`.
- **`evaluate_crash.py`** — if `eval_arrays/<sim_name>.npz` exists, reads the prediction/ground-truth arrays and writes per-simulation metric plots/files with prefix `outputs/<run>/results/<sim_name>/<sim_name>`.

### 4. Aggregate L2 error (`compute_l2_error_npz.py`)

After all simulations are processed, reads every `.npz` under `outputs/<run>/results/eval_arrays/` and produces:

- `outputs/<run>/results/l2_error.png` — L2 error plot across all simulations
- `outputs/<run>/results/l2_error.csv` — L2 error table

## Output structure

```
outputs/<run>/
  checkpoints/          # model checkpoint (input, must exist)
  stats/                # normalization stats (input, must exist)
  results/
    eval_arrays/
      <sim_name>.npz    # prediction + ground-truth arrays from inference
    <sim_name>/
      <sim_name>_crash.gif     # animated visualization
      <sim_name>_*.*           # per-sim metric files from evaluate_crash.py
    l2_error.png        # aggregate L2 error plot
    l2_error.csv        # aggregate L2 error table
```
