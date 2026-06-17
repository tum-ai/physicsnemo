"""
Compute relative L2 position error from inference .npz files (VTKHDF pipeline).

Each .npz file (produced by inference.py) contains:
  pred_pos  : [T, N, 3]  predicted positions
  exact_pos : [T, N, 3]  ground-truth positions

Usage (single model):
    python compute_l2_error_npz.py \
        --npz_dir outputs/figconvnet_full/predicted_vtps/eval_arrays \
        --out outputs/figconvnet_full/l2_error.png \
        --csv outputs/figconvnet_full/l2_error.csv

Usage (comparison):
    python compute_l2_error_npz.py \
        --npz_dir outputs/figconvnet_full/predicted_vtps/eval_arrays \
                  outputs/geotransolver/predicted_vtps/eval_arrays \
        --labels FIGConvNet GeoTransolver \
        --out outputs/l2_comparison.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def relative_l2_position_error(pred: np.ndarray, exact: np.ndarray) -> np.ndarray:
    """
    Per-timestep relative L2 position error.
    pred, exact: [T, N, 3]
    Returns: [T] array of floats
    """
    assert pred.shape == exact.shape, f"Shape mismatch: {pred.shape} vs {exact.shape}"
    T = pred.shape[0]
    errors = np.zeros(T)
    for t in range(T):
        diff = (pred[t] - exact[t]).ravel()
        denom = exact[t].ravel()
        num = np.linalg.norm(diff, ord=2)
        den = np.linalg.norm(denom, ord=2)
        errors[t] = num / den if den > 1e-12 else 0.0
    return errors


def load_npz_dir(npz_dir: str):
    """Load all .npz files in a directory. Returns list of (T,) error arrays."""
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in: {npz_dir}")
    all_errors = []
    for f in files:
        data = np.load(f)
        if "pred_pos" not in data or "exact_pos" not in data:
            print(f"  Skipping {os.path.basename(f)} — missing pred_pos or exact_pos")
            continue
        err = relative_l2_position_error(data["pred_pos"], data["exact_pos"])
        all_errors.append(err)
        print(f"  {os.path.basename(f)}: mean L2 = {err.mean():.4f}")
    return all_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_dir",
        nargs="+",
        required=True,
        help="One or more directories containing inference .npz files",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (one per npz_dir)",
    )
    parser.add_argument("--out", default="l2_error.png", help="Output PNG path")
    parser.add_argument("--csv", default=None, help="Optional output CSV path")
    args = parser.parse_args()

    labels = args.labels or [
        os.path.basename(os.path.dirname(os.path.dirname(d))) for d in args.npz_dir
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (npz_dir, label) in enumerate(zip(args.npz_dir, labels)):
        print(f"\n{label} — loading from {npz_dir}")
        all_errors = load_npz_dir(npz_dir)
        if not all_errors:
            print(f"  No valid files found, skipping.")
            continue

        T = min(len(e) for e in all_errors)
        stack = np.stack([e[:T] for e in all_errors], axis=0)  # [num_sims, T]
        means = stack.mean(axis=0)
        stds = stack.std(axis=0)
        timesteps = np.arange(1, T + 1)
        color = colors[i % len(colors)]

        ax.plot(timesteps, means, color=color, linewidth=1.8, label=label)
        ax.fill_between(
            timesteps,
            np.clip(means - stds, 0, None),
            means + stds,
            color=color,
            alpha=0.2,
        )

        print(f"  Runs: {len(all_errors)}, Timesteps: {T}")
        print(f"  Mean L2 (all timesteps): {means.mean():.4f} ± {stds.mean():.4f}")

        if args.csv and i == 0:
            import csv
            with open(args.csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestep", "mean_l2", "std_l2", "num_sims"])
                for t, m, s in zip(timesteps, means, stds):
                    w.writerow([int(t), float(m), float(s), len(all_errors)])
            print(f"  Saved CSV: {args.csv}")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Relative $L^2$ Position Error")
    ax.set_title("Relative $L^2$ Position Error — Bumper Beam Benchmark")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot: {args.out}")


if __name__ == "__main__":
    main()
