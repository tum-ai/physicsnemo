# Training Automation & Customization

This directory contains shell script templates to automate training models on the server. It includes the customizable template script **`sample.sh`**, which is configured for the `/mnt/1t/mit-project/` environment.

---

## How to Create your Execution Script

To set up a new training run:

1. **Copy the template**:
   Create a copy of `sample.sh` for the specific run:
   ```bash
   cp scripts/sample.sh scripts/train_run.sh
   chmod +x scripts/train_run.sh
   ```

2. **Configure the script parameters**:
   Open the copy in an editor and modify the variables in the **`CONFIGURATION BLOCK`**:

   ```bash
   # 1. Hydra Experiment Configuration Name (from conf/ directory, excluding the .yaml extension)
   CONFIG_NAME="bumper_vtkhdf_geotransolver_oneshot"
   
   # 2. Name the experiment (this determines the subfolder name under ./outputs/)
   EXP_NAME="my_geotransolver_run_1"
   
   # 3. Adjust dataset sizes to match the downloaded simulations (these work for the first simulation)
   NUM_TRAIN_SAMPLES=20
   NUM_VAL_SAMPLES=3
   ```

3. **Verify paths**:
   By default, the script is configured to use the following paths:
   * **Simulations**: `/mnt/1t/mit-project/Dataset/simulations`
   * **Master CSV**: `/mnt/1t/mit-project/Dataset/metadata/bumper_beam_master_with_split.csv`
   * **Project Directory**: `/mnt/1t/mit-project/physicsnemo/examples/structural_mechanics/crash`

4. **Execute the script**:
   From the `examples/structural_mechanics/crash/` directory, run:
   ```bash
   ./scripts/train_run.sh
   ```

---

## Configuration Variables Explained

| Variable | Description |
| :--- | :--- |
| **`CONFIG_NAME`** | The Hydra configuration file to use from the `conf/` directory (e.g. `bumper_vtkhdf_geotransolver_oneshot`). |
| **`EXP_NAME`** | The label for this specific run. This isolates output checkpoints and logs from other runs. |
| **`DATA_DIR`** | The root path of the dataset directory on your server. |
| **`PROJ_DIR`** | The absolute path to this project workspace. |
| **`META_DIR`** | The directory containing the master split CSV files. |
| **`OUT_DIR`** | The directory where Hydra and model weights/logs will be written. Maps dynamically to `./outputs/${EXP_NAME}/`. |
| **`NUM_TRAIN_SAMPLES`** | Overrides `training.num_training_samples` to match your downloaded dataset size. |
| **`NUM_VAL_SAMPLES`** | Overrides `training.num_validation_samples` to match your validation split size. |

---

## Running Multi-GPU / Distributed Training

To run the script on multiple GPUs using PyTorch's `torchrun`, change the execution command at the bottom of your script from `python train.py ...` to:

```bash
HYDRA_FULL_ERROR=1 torchrun --nproc_per_node=<NUMBER_OF_GPUS> train.py --config-name="$CONFIG_NAME" \
    ...
```

---

## Output Isolation (No Overwrites)

The script automatically overrides the Hydra run directory by setting:
```bash
hydra.run.dir="$OUT_DIR"
```
Because `$OUT_DIR` is set to `./outputs/${EXP_NAME}/`, every training run with a unique `EXP_NAME` is fully isolated in its own directory. 
* Checkpoints, TensorBoard logs, and Hydra configuration outputs will be cleanly saved under `outputs/<EXP_NAME>/`.
* This prevents concurrent or subsequent runs from overwriting each other's weights or overlaying TensorBoard graphs.
