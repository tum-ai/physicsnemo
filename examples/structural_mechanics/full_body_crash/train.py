"""
train.py
========
Train a singular GeoTransolver model (Baseline or Enhanced with modular geometric encoder)
on full-body crash simulations. Logs to WandB and saves checkpoints.
"""

import sys
import time
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

# Add current directory to path to load local modules
sys.path.insert(0, str(Path(__file__).parent))
from vtkhdf_reader import load_simulation
from geo_transolver_enhanced import EnhancedGeoTransolver


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sim(name: str, data_root: Path, filter_parts_csv: str | None, pos_scale: float, device: torch.device) -> dict | None:
    vtkhdf = data_root / name / "field_trajectory.vtkhdf"
    if not vtkhdf.exists():
        print(f"  WARNING: {name} not found in {data_root} — skipping")
        return None

    print(f"  {name} ...", end="", flush=True)
    t0   = time.perf_counter()
    data = load_simulation(str(vtkhdf))
    print(f" {time.perf_counter() - t0:.1f}s")

    pos  = data["positions"]     # (T, N, 3) float32
    pid  = data["part_id"]       # (N,)      int32

    # Filter parts if CSV is provided
    if filter_parts_csv is not None:
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
# Training / Validation step functions
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    train_sims: list[dict],
    model: nn.Module,
    optimizer: optim.Optimizer,
    encoder_active: bool,
    stats_cache: dict,
) -> float:
    model.train()
    losses = []
    for sim in train_sims:
        optimizer.zero_grad()
        if encoder_active:
            pred = model(
                local_embedding=sim["pos0"],
                geometry=sim["pos0"],
                part_id=sim["part_id"],
                precomputed_encoder_stats=stats_cache[sim["name"]],
            )
        else:
            pred = model(
                local_embedding=sim["pos0"],
                geometry=sim["pos0"],
            )
        loss = F.mse_loss(pred, sim["target"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(
    val_sims: list[dict],
    model: nn.Module,
    encoder_active: bool,
    stats_cache: dict,
) -> tuple[float, float]:
    model.eval()
    mses = []
    rel_l2s = []
    for val_sim in val_sims:
        if encoder_active:
            pred = model(
                local_embedding=val_sim["pos0"],
                geometry=val_sim["pos0"],
                part_id=val_sim["part_id"],
                precomputed_encoder_stats=stats_cache[val_sim["name"]],
            )
        else:
            pred = model(
                local_embedding=val_sim["pos0"],
                geometry=val_sim["pos0"],
            )
        target = val_sim["target"]
        mse    = F.mse_loss(pred, target).item()
        rel_l2 = (torch.norm(pred - target) / (torch.norm(target) + 1e-8)).item()
        mses.append(mse)
        rel_l2s.append(rel_l2)
    return float(np.mean(mses)), float(np.mean(rel_l2s))


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train a singular GeoTransolver model.")
    # Encoder type
    parser.add_argument(
        "--encoder",
        type=str,
        choices=["baseline", "stats_only", "enhanced"],
        default="enhanced",
        help="Which geometric encoder to run. baseline has no encoder.",
    )
    # Dataset and paths
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/1t/mit-project/Dataset/full_body",
        help="Root path to the full body crash dataset.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Path to save the training results. If None, defaults to outputs/<encoder>/results.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root directory for saving training outputs.",
    )

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--filter_parts_csv", type=str, default=None, help="Path to CSV containing part IDs to filter/work with.")
    parser.add_argument("--pos_scale", type=float, default=1000.0, help="Scale factor to divide positions by.")
    parser.add_argument("--log_every", type=int, default=10, help="Validate and log metrics every N epochs.")

    parser.add_argument(
        "--n_steps",
        type=int,
        default=25,
        help="Save model weights checkpoints every N epochs.",
    )
    # Backbone Hyperparameters
    parser.add_argument("--n_hidden", type=int, default=128, help="Backbone hidden dimensionality.")
    parser.add_argument("--n_layers", type=int, default=4, help="Backbone number of layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Backbone number of attention heads.")
    parser.add_argument("--slice_num", type=int, default=32, help="Backbone slice tokens count.")
    # Encoder Hyperparameters (if active)
    parser.add_argument(
        "--enc_radii",
        nargs="+",
        type=float,
        default=[0.05, 0.20, 0.60],
        help="List of radii for ball-queries in normalized units.",
    )
    parser.add_argument(
        "--enc_max_k",
        nargs="+",
        type=int,
        default=[8, 32, 64],
        help="List of max neighbor counts corresponding to radii.",
    )
    parser.add_argument("--enc_hdim", type=int, default=32, help="Hidden dimension of encoder projection.")
    parser.add_argument("--enc_part_edim", type=int, default=8, help="Part ID embedding dimension.")
    parser.add_argument("--enc_n_parts", type=int, default=300, help="Size of part-ID embedding table.")

    args = parser.parse_args()

    # ── setups ────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)

    # Resolve output directory: outputs/<encoder>/weights/
    output_root = Path(args.output_dir) / args.encoder
    weights_dir = output_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if args.out_csv is None:
        out_csv_path = output_root / "results.csv"
    else:
        out_csv_path = Path(args.out_csv)
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)



    # ── Initialize wandb ──────────────────────────────────────────────────
    wandb.init(
        project="geotransolver-crash",
        name=f"run_{args.encoder}_seed{args.seed}",
        config=vars(args),
    )

    # Save wandb run ID for resuming during inference
    wandb_id_file = output_root / "wandb_run_id.txt"
    with open(wandb_id_file, "w") as f:
        f.write(wandb.run.id)

    print(f"Device : {device}")
    print(f"Encoder: {args.encoder}")
    print(f"FILTER_CSV: {args.filter_parts_csv}  |  EPOCHS: {args.epochs}  |  LR: {args.lr}")

    # ── load data ─────────────────────────────────────────────────────────
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected 'train' and 'val' subdirectories under {data_root}"
        )
        
    train_sim_names = sorted([d.name for d in train_dir.glob("neon_*") if d.is_dir()])
    val_sim_names = sorted([d.name for d in val_dir.glob("neon_*") if d.is_dir()])
    
    if not train_sim_names:
        raise ValueError(f"No simulations starting with 'neon_*' found in {train_dir}")
    if not val_sim_names:
        raise ValueError(f"No simulations starting with 'neon_*' found in {val_dir}")

    print("\nLoading training simulations …")
    train_sims = [
        d for name in train_sim_names
        if (d := load_sim(name, train_dir, args.filter_parts_csv, args.pos_scale, device)) is not None
    ]
    
    print("Loading validation simulations …")
    val_sims = [
        d for name in val_sim_names
        if (d := load_sim(name, val_dir, args.filter_parts_csv, args.pos_scale, device)) is not None
    ]
    print(f"Loaded {len(train_sims)} training + {len(val_sims)} validation simulations")

    # ── build modular encoder & model ─────────────────────────────────────
    print("\nBuilding model & encoder …")
    
    t_steps = 17           # timesteps per simulation
    n_pred  = t_steps - 1  # predict t = 1 … T-1
    out_dim = n_pred * 3   # 48: all displacements flattened

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
    else:
        raise ValueError(f"Unknown encoder type: {args.encoder}")

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

    # Print param count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    wandb.summary[f"{args.encoder}/params"] = n_params

    # ── precompute geometric stats (one-time cost) ────────────────────────
    stats_cache = {}
    if encoder is not None:
        all_sims = train_sims + val_sims
        print("\nPrecomputing geometric stats (one-time O(N²) cost) …")
        for sim in all_sims:
            t0_pre = time.perf_counter()
            stats_cache[sim["name"]] = encoder.precompute(sim["pos0"], sim["part_id"])
            print(f"  {sim['name']}: {time.perf_counter() - t0_pre:.1f}s")

    # ── optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ── train loop ────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Training: {args.encoder.upper()}")
    print(f"{'='*62}")

    rows = []
    window_losses = []
    wandb_prefix = {
        "baseline": "baseline",
        "stats_only": "statsonly",
        "enhanced": "enhanced",
    }[args.encoder]

    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            train_sims, model, optimizer, (encoder is not None), stats_cache
        )
        window_losses.append(train_loss)

        # Validate & Log
        if epoch % args.log_every == 0:
            val_mse, val_rl2 = evaluate(
                val_sims, model, (encoder is not None), stats_cache
            )
            avg_train = float(np.mean(window_losses))
            window_losses.clear()

            rows.append({
                "epoch":      epoch,
                "train_mse":  avg_train,
                "val_mse":    val_mse,
                "val_rel_l2": val_rl2,
            })
            print(
                f"  epoch {epoch:3d}/{args.epochs}"
                f"  train_mse={avg_train:.5f}"
                f"  val_mse={val_mse:.5f}"
                f"  val_rel_l2={val_rl2:.4f}"
            )

            # Wandb log matching original prefix schema
            wandb.log(
                {
                    f"{wandb_prefix}/train_mse":  avg_train,
                    f"{wandb_prefix}/val_mse":    val_mse,
                    f"{wandb_prefix}/val_rel_l2": val_rl2,
                },
                step=epoch,
            )

        # Save Checkpoint
        if epoch % args.n_steps == 0:
            ckpt_path = weights_dir / f"model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved weights checkpoint to {ckpt_path}")

    print(f"Training completed in {(time.perf_counter() - t_start)/60:.1f} minutes.")

    # ── write CSV ─────────────────────────────────────────────────────────
    fieldnames = ["epoch", "train_mse", "val_mse", "val_rel_l2"]
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "epoch":      r["epoch"],
                "train_mse":  f"{r['train_mse']:.6f}",
                "val_mse":    f"{r['val_mse']:.6f}",
                "val_rel_l2": f"{r['val_rel_l2']:.6f}",
            })
    print(f"Saved training history to {out_csv_path}")

    # Log best summary metrics to WandB
    best_val_rl2 = min(r["val_rel_l2"] for r in rows)
    wandb.summary[f"{wandb_prefix}/best_val_rel_l2"] = best_val_rl2
    wandb.finish()


if __name__ == "__main__":
    main()
