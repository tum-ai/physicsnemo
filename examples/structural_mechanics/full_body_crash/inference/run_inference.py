"""
run_inference.py
================
Run model evaluation rollout on a raw test dataset containing VTKHDF files.
Loads the specified model type (baseline, stats_only, or enhanced), runs inference,
reconstructs absolute position trajectories, and saves them as .npz files.
"""

import sys
import os
import time
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path to load vtkhdf_reader and geo_transolver_enhanced
sys.path.insert(0, str(Path(__file__).parent.parent))
from vtkhdf_reader import load_simulation
from geo_transolver_enhanced import EnhancedGeoTransolver


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sim(name: str, data_root: Path, filter_parts_csv: str | None, pos_scale: float, device: torch.device) -> dict | None:
    vtkhdf = data_root / name / "field_trajectory.vtkhdf"
    if not vtkhdf.exists():
        return None

    print(f"  Loading {name} ...", end="", flush=True)
    t0   = time.perf_counter()
    data = load_simulation(str(vtkhdf))
    print(f" {time.perf_counter() - t0:.1f}s")

    pos  = data["positions"]     # (T, N, 3) float32
    pid  = data["part_id"]       # (N,)      int32

    # Filter parts if CSV is provided
    if filter_parts_csv is not None:
        import csv
        target_parts = set()
        with open(filter_parts_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "part_id" in row and row["part_id"]:
                    try:
                        # If 'selected' column exists, load only if True (case-insensitive)
                        if row.get("selected", "true").strip().lower() == "true":
                            target_parts.add(int(row["part_id"]))
                    except ValueError:
                        pass
        mask = np.isin(pid, list(target_parts))
        if not np.any(mask):
            print(f"Warning: No nodes matched part IDs from {filter_parts_csv} in {name}!")
            return None
        pos = pos[:, mask, :]
        pid = pid[mask]

    # Convert coordinates from mm to normalized scale
    pos = pos.astype(np.float32) / pos_scale

    n_nodes = pos.shape[1]
    _, pid_remap = np.unique(pid, return_inverse=True)       # 0-indexed contiguous

    # displacement target: pos[t] − pos[0] for t = 1…T-1
    disp      = pos[1:] - pos[0]                             # (T_PRED, N, 3)
    disp_flat = disp.transpose(1, 0, 2).reshape(n_nodes, -1)   # (N, T_PRED * 3)

    return {
        "name":    name,
        "pos0":    torch.tensor(pos[0]).unsqueeze(0).to(device),                  # (1, N, 3)
        "target":  torch.tensor(disp_flat).unsqueeze(0).to(device),              # (1, N, out_dim)
        "part_id": torch.tensor(pid_remap, dtype=torch.long).to(device),         # (N,)
        "n_parts": int(pid_remap.max()) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run inference using a trained GeoTransolver model.")
    # Configuration
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root path to the test set raw simulation directories.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["baseline", "stats_only", "enhanced"],
        required=True,
        help="Which encoder structure the model was trained with.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the model weights checkpoint (.pt file). If None, auto-selects the latest epoch checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root directory for saving output arrays.",
    )

    parser.add_argument("--filter_parts_csv", type=str, default=None, help="Path to CSV containing part IDs to filter/work with.")
    parser.add_argument("--pos_scale", type=float, default=1000.0, help="Divide positions by this.")

    parser.add_argument("--n_hidden", type=int, default=128, help="Backbone hidden dimensionality.")
    parser.add_argument("--n_layers", type=int, default=4, help="Backbone layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Backbone attention heads.")
    parser.add_argument("--slice_num", type=int, default=32, help="Backbone slice tokens count.")
    # Encoder params
    parser.add_argument("--enc_radii", nargs="+", type=float, default=[0.05, 0.20, 0.60])
    parser.add_argument("--enc_max_k", nargs="+", type=int, default=[8, 32, 64])
    parser.add_argument("--enc_hdim", type=int, default=32)
    parser.add_argument("--enc_part_edim", type=int, default=8)
    parser.add_argument("--enc_n_parts", type=int, default=300)

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: Ignoring unrecognized arguments forwarded to run_inference.py: {unknown}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)

    # ── Auto-resolve checkpoint if None ───────────────────────────────────
    if args.checkpoint is None:
        weights_pattern = os.path.join(args.output_dir, args.encoder, "weights", "model_epoch_*.pt")
        checkpoints = glob.glob(weights_pattern)
        if not checkpoints:
            raise RuntimeError(
                f"No checkpoints found in {weights_pattern}. Please specify --checkpoint explicitly."
            )
        # Sort by epoch number to find the latest
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
        args.checkpoint = checkpoints[-1]
        print(f"Auto-selected latest weights checkpoint: {args.checkpoint}")

    # ── Resolve outputs directories ───────────────────────────────────────
    output_root = Path(args.output_dir) / args.encoder
    npz_dir = output_root / "npz_files"
    npz_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover test directories ─────────────────────────────────────────
    test_dir = data_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected 'test' subdirectory under {data_root}")
        
    test_sim_dirs = sorted([d.name for d in test_dir.glob("neon_*") if d.is_dir()])
    if not test_sim_dirs:
        print(f"No simulations starting with 'neon_*' found in {test_dir}")
        return
    print(f"Found {len(test_sim_dirs)} test simulations for inference.")

    # ── Build modular encoder and model ───────────────────────────────────
    t_steps = 17
    n_pred = t_steps - 1
    out_dim = n_pred * 3

    if args.encoder == "baseline":
        encoder = None
    elif args.encoder == "stats_only":
        from geo_encoders import StatsOnlyEncoder
        encoder = StatsOnlyEncoder(
            radii=args.enc_radii,
            max_neighbors=args.enc_max_k,
            hidden_dim=args.enc_hdim,
        ).to(device)
    elif args.encoder == "enhanced":
        from geo_encoders import GeometricEncoder
        encoder = GeometricEncoder(
            radii=args.enc_radii,
            max_neighbors=args.enc_max_k,
            n_parts=args.enc_n_parts,
            part_embed_dim=args.enc_part_edim,
            hidden_dim=args.enc_hdim,
        ).to(device)

    model = EnhancedGeoTransolver(
        functional_dim=3,
        out_dim=out_dim,
        geometry_dim=3,
        global_dim=None,
        geo_encoder=encoder,
        n_layers=args.n_layers,
        n_hidden=args.n_hidden,
        n_head=args.n_head,
        slice_num=args.slice_num,
        use_te=False,
        include_local_features=False,
    ).to(device)

    # Load weights
    print(f"Loading weights from {args.checkpoint} ...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # ── Run Rollouts ──────────────────────────────────────────────────────
    print("\nRunning inference rollouts...")
    for sim_name in test_sim_dirs:
        sim = load_sim(sim_name, test_dir, args.filter_parts_csv, args.pos_scale, device)
        if sim is None:
            continue

        with torch.no_grad():
            if encoder is not None:
                # Precompute stats for inference speed
                stats = encoder.precompute(sim["pos0"], sim["part_id"])
                pred = model(
                    local_embedding=sim["pos0"],
                    geometry=sim["pos0"],
                    part_id=sim["part_id"],
                    precomputed_encoder_stats=stats,
                )
            else:
                pred = model(
                    local_embedding=sim["pos0"],
                    geometry=sim["pos0"],
                )

        # ── Reconstruct trajectories ──────────────────────────────────────
        # Target/Prediction coordinates shape: [1, N, 48]
        # Reshape to [1, N, N_PRED, 3] -> transpose to [N_PRED, N, 3]
        N = sim["pos0"].shape[1]
        pos0_np = sim["pos0"].squeeze(0).cpu().numpy()          # [N, 3]
        pred_disp = pred.squeeze(0).view(N, n_pred, 3).transpose(1, 0).cpu().numpy()  # [N_PRED, N, 3]
        exact_disp = sim["target"].squeeze(0).view(N, n_pred, 3).transpose(1, 0).cpu().numpy()  # [N_PRED, N, 3]

        pred_traj_norm = np.zeros((t_steps, N, 3), dtype=np.float32)
        exact_traj_norm = np.zeros((t_steps, N, 3), dtype=np.float32)

        # Timestep 0
        pred_traj_norm[0] = pos0_np
        exact_traj_norm[0] = pos0_np

        # Timestep 1...T-1
        for t in range(n_pred):
            pred_traj_norm[t+1] = pos0_np + pred_disp[t]
            exact_traj_norm[t+1] = pos0_np + exact_disp[t]

        # Denormalize to original coordinate scale (mm)
        pred_traj = pred_traj_norm * args.pos_scale
        exact_traj = exact_traj_norm * args.pos_scale

        # Save arrays
        npz_path = npz_dir / f"{sim_name}.npz"
        np.savez_compressed(
            npz_path,
            pred_pos=pred_traj,
            exact_pos=exact_traj
        )
        print(f"    Saved evaluation arrays -> {npz_path}")

    print("\nInference pipeline successfully completed.")


if __name__ == "__main__":
    main()
