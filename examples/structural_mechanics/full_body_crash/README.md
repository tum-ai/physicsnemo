# Modular Geometric Encoding & Benchmarking for Full-Body Crash

This subfolder contains a modular training, evaluation, and comparative benchmarking pipeline for injecting geometric context into **GeoTransolver** models applied to full-body vehicle crash simulations.

---

## 1. General Usage

### Data Directory Setup
The pipeline expects the input simulation dataset to be organized into three distinct subdirectories under the dataset root folder:

```
/path/to/dataset/root/
├── train/
│   ├── neon_full_frontal_ml_0002/
│   ├── neon_full_frontal_ml_0003/
│   └── ...
├── val/
│   ├── neon_full_frontal_ml_0010/
│   └── ...
└── test/
    ├── neon_full_frontal_ml_0001/
    ├── neon_full_frontal_ml_0004/
    └── ...
```

---

### Part Filtering & Slicing (No Subsampling)
Instead of randomly subsampling the entire vehicle structure, the data loader filters the mesh to extract the exact components you are interested in (e.g., bumper beams or the main chassis frame) based on a CSV file.

#### CSV Schema
The CSV file should contain target numerical `part_id`s in the first column. A header row is optional (the code automatically skips any row that begins with `part_id`).

Example CSV file (`chassis_parts.csv`):
```csv
part_id,part_name
102,main_chassis_frame
105,subframe_rear
112,front_member
```

When you specify this CSV path via `--filter_parts_csv`, the data loader slices all simulation coordinate, target, and part ID arrays to include **only** the nodes belonging to these components. All matching nodes are processed and fed directly to the model without any further subsampling. If `--filter_parts_csv` is omitted (`None`), the model loads the entire car mesh.

---

### Running the End-to-End Pipeline
You can trigger training followed immediately by testing, metric calculations, and animated trajectory uploads in the **same WandB experiment** using the wrapper script [train.sh](./train.sh):

```bash
./train.sh --encoder enhanced \
           --epochs 200 \
           --data_root /path/to/dataset/root
```

This runs training, creates model weights under `outputs/enhanced/weights/`, and automatically triggers the evaluation rollouts and visualizations on the test set simulations.

---

### Command Line Arguments

#### Training (`train.py`)
Launch training on a single model configuration with [train.py](./train.py):
- `--encoder`: Choice of geometric encoder: `baseline` (none), `stats_only`, or `enhanced` (default: `enhanced`).
- `--data_root`: Path to the raw simulation data.
- `--output_dir`: Directory to save outputs (default: `outputs`).
- `--epochs`: Number of training epochs (default: `200`).
- `--lr`: Learning rate (default: `1e-3`).
- `--n_steps`: Save checkpoint weights every $N$ epochs (default: `25`).
- `--log_every`: Validate and log metrics to WandB every $N$ epochs (default: `10`).
- `--filter_parts_csv`: Path to CSV containing part IDs to filter/work with. If None, uses the full car.
- `--n_hidden`, `--n_layers`, `--n_head`: Backbone transformer sizes.

#### Inference & Evaluation (`inference.sh`)
Evaluate the model and visualize predictions using [inference.sh](./inference/inference.sh):
```bash
./inference/inference.sh /path/to/dataset/root enhanced [wandb_run_id]
```
- Calls [run_inference.py](./inference/run_inference.py) to load the latest weights checkpoint, run rollouts on the test set, and save coordinates to `outputs/{encoder}/npz_files/`.
- Calls [compute_metrics.py](./inference/compute_metrics.py) to print errors, generate relative L2 error plots, and log curve data to WandB.
- Calls [visualize_predictions.py](./inference/visualize_predictions.py) to build 3-panel animated trajectory GIFs (displacement and local error maps) and log them to WandB.

#### Cross-Model Benchmarking (`run_comparison.sh`)
Compare baseline and enhanced model predictions locally without touching WandB using [run_comparison.sh](./run_comparison.sh):
```bash
./run_comparison.sh
```
- Calls [compare_models.py](./inference/compare_models.py) to overlay relative L2 curves for baseline, stats_only, and enhanced models, saving the comparison summary to `outputs/comparisons/`.

---

## 2. How to Create New Encoders

All custom geometric encoders are placed inside the [geo_encoders/](./geo_encoders) subfolder. To add a new encoder, follow these steps:

### 1. Implement the Encoder Class
Create your encoder class as a PyTorch module inheriting from `nn.Module`. Your encoder must conform to this signature:

```python
import torch
import torch.nn as nn

class MyNewEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, **kwargs) -> None:
        super().__init__()
        # 1. Define hidden_dim property so the model knows how many channels to append
        self.hidden_dim = hidden_dim
        
        # 2. Initialize layers (e.g. MLPs, PointNets, GNNs)
        self.network = nn.Sequential(...)

    def precompute(self, positions: torch.Tensor, part_id: torch.Tensor) -> list[torch.Tensor]:
        """
        Optional precompute step to compute neighborhood graphs, eigenvalues, 
        or distances in O(N²) once before training rather than inside the training loop.
        Should return a list/dict of PyTorch tensors.
        """
        # Calculate raw statistics or embeddings
        return [...]

    def forward(
        self,
        positions: torch.Tensor,                # (B, N, 3) XYZ coordinates
        part_id: torch.Tensor,                  # (N,) Long part IDs
        precomputed_stats: list[torch.Tensor] | None = None, # Cached objects
    ) -> torch.Tensor:
        """
        Output: a tensor of shape (B, N, hidden_dim) representing 
        per-node geometric context.
        """
        if precomputed_stats is not None:
            # Use cached representations for fast O(N) training forward passes
            features = precomputed_stats
        else:
            # Compute representation dynamically (used for inference / validation)
            features = ...
            
        return self.network(features)
```

### 2. Expose the Encoder
Add your class to the package initialization file [geo_encoders/\_\_init\_\_.py](./geo_encoders/__init__.py):
```python
from .my_new_encoder import MyNewEncoder

__all__ = ["GeometricEncoder", "StatsOnlyEncoder", "MyNewEncoder"]
```

### 3. Connect to the Base Model
The base model [EnhancedGeoTransolver](./geo_transolver_enhanced.py) is encoder-agnostic. It accepts any `nn.Module` encoder instance during initialization:
```python
model = EnhancedGeoTransolver(
    functional_dim=3,
    out_dim=out_dim,
    geometry_dim=3,
    geo_encoder=encoder_instance, # Appends encoder_instance.hidden_dim to geometry
    ...
)
```

### 4. Hook up to train.py and run_inference.py
Instantiate your encoder in [train.py](./train.py) and [run_inference.py](./inference/run_inference.py) by adding your choice to the CLI parser and building the instance:
```python
# Add CLI choice
parser.add_argument("--encoder", choices=["baseline", "stats_only", "enhanced", "my_new_encoder"])

# In main(): instantiate encoder
if args.encoder == "my_new_encoder":
    from geo_encoders import MyNewEncoder
    encoder = MyNewEncoder(hidden_dim=args.enc_hdim).to(device)
```

---

## 3. Core Ideas Behind Encoders

### `GeometricEncoder` (Stats + Part ID)
The enhanced [GeometricEncoder](./geo_encoders/geometric_encoder.py) combines spatial neighborhood structure with manufacturing metadata (part IDs).
- **Multi-Scale Statistical Queries**: Runs parallel ball queries at three spatial scales (default radii: `0.05`, `0.20`, `0.60`). For each scale, it computes **7 distinct geometric descriptors**:
  - `[0]` Normalized mean distance to local neighbors.
  - `[1]` Standard deviation of local neighbor distances.
  - `[2]` Local neighborhood density normalized by max neighbors.
  - `[3] - [5]` PCA eigenvalues ($e_0, e_1, e_2$) representing local structure shape (linear vs. planar vs. volumetric).
  - `[6]` Same-part fraction (what percentage of neighbors share this node's part ID).
- **Learned Part Embeddings**: Node part IDs are projected into a continuous latent space via an embedding table. This allows the model to learn a topological hierarchy of vehicle sections (e.g. stiffness of bumper vs. crushbox).
- **MLP Fusion**: The concatenated statistics and part embeddings are projected through a multi-layer perceptron into the final geometric context vectors.

### `StatsOnlyEncoder` (Ablation)
The [StatsOnlyEncoder](./geo_encoders/stats_only_encoder.py) is used for ablation analysis to measure the contribution of the categorical part-ID metadata.
- **Dropping Part IDs**: It removes the Same-Part Fraction descriptor (column 6) and completely drops the `nn.Embedding` table.
- **Pure Geometric Statistics**: Uses only the remaining **6 structural descriptors** (mean dist, std dist, density, and 3 PCA eigenvalues) per scale.
- **MLP Fusion**: Projects the combined 18 raw values ($6 \text{ stats} \times 3 \text{ scales}$) to the hidden context dimension. By comparing this against the full `GeometricEncoder`, we can quantify the performance gain provided by part segmentation metadata versus raw spatial layouts.
