# Running inference on the remote machine

`inference.py` runs on **every** simulation in the folder you point it at
(`inference.raw_data_dir_test`). So point it at the folder that contains the
sims you want to evaluate.

> Note: for the bumper (VTKHDF) dataset the split column in `master_csv` is
> **not** used to filter sims at inference time — the folder contents decide
> what gets evaluated.

## Steps

```bash
cd physicsnemo/examples/structural_mechanics/crash


This writes per-sim results to `outputs/<model_name>/results/eval_arrays/<sim>.npz`.
Evaluate one with:

```bash
# adjust these paths for the remote machine
DATA=/mnt/1t/mit-project/Dataset
SIMS=$DATA/simulations
GLOBAL_FEATURES_JSON=/mnt/1t/mit-project/physicsnemo/examples/structural_mechanics/crash/global_features.json
DIR=/mnt/1t/mit-project/physicsnemo/examples/structural_mechanics/crash/outputs 
RUN=transolver     # adjust this according to the name of your model geotransolver, figconvnet etc.
OUTPUT_DIR=results

python inference.py --config-name=bumper_transolver_oneshot \
  inference.raw_data_dir_test="$SIMS" \
  reader.master_csv="$DATA/metadata/bumper_beam_master_with_split.csv" \
  training.global_features_filepath="$GLOBAL_FEATURES_JSON" \
  training.ckpt_path="$DIR/$RUN/checkpoints" \
  inference.output_dir_pred="$DIR/$RUN/$OUTPUT_DIR" \
  datapipe.stats_dir="$DIR/$RUN/stats"
```

This writes per-sim results to `outputs/predicted_vtps/eval_arrays/<sim>.npz`.
Evaluate one with:

```bash
python evaluate_crash.py "$DIR/$RUN/$OUTPUT_DIR/eval_arrays/sim_00000.npz" \
  --stride 5 --out-prefix "$DIR/$RUN/$OUTPUT_DIR/sim_00000"
```

## Notes

- **Folder layout:** each sim lives in its own subdirectory as
  `<sim>/<sim>.vtkhdf`. The loader discovers them automatically.
- **Checkpoint:** the trained model must exist at `training.ckpt_path`
  (default `outputs/checkpoints`). Copy it to the remote or set the path
  explicitly as above.
- **Evaluate everything vs. a subset:** inference processes the whole folder.
  To evaluate only some sims, point `raw_data_dir_test` at a folder that
  contains just those sims.
