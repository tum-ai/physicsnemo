<!-- markdownlint-disable -->
# Configuration Layout

## Start here: experiment configs

**Each YAML file in `conf/` is a self-contained experiment config.** Run training or inference by selecting one:

```bash
python train.py --config-name=bumper_geotransolver_oneshot
python train.py --config-name=crash_geotransolver_oneshot
python inference.py --config-name=crash_geotransolver_oneshot
```

To add a new experiment, copy an existing file in `conf/` and edit data paths, model, and features.

---

## Configuration Conventions

To maintain a clean, standardized, and modular codebase, please adhere to the following guidelines:

* **Naming Convention**: Name experiment configuration files according to the template: `<dataset>_<model>_<rollout_type>.yaml` (e.g., `bumper_geotransolver_oneshot.yaml`).
* **Format Independence**: Avoid creating new configurations purely for different storage formats (e.g., VTP vs. Zarr). Instead, dynamically switch the reader using overrides (e.g., `reader=vtkhdf`).
* **Environment Portability**: Do not hardcode environment-specific directory paths directly inside the configuration files. Pass these paths dynamically using your execution shell scripts to keep the repository versatile and portable across different infrastructures.

## Component configs (advanced)

The subfolders (`model/`, `datapipe/`, `reader/`, `training/`, `inference/`) contain configs referenced by experiments. You rarely need to edit them unless customizing models, readers, or training defaults.

| Path           | Purpose                                      |
|----------------|----------------------------------------------|
| `model/`       | Model architectures (selected via experiment) |
| `datapipe/`    | Dataset and feature configs                  |
| `reader/`      | Data format readers (VTP, Zarr)               |
| `training/`    | Training hyperparameters                      |
| `inference/`   | Inference options                            |
