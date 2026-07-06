"""
Integration test: load one full-body crash simulation, subsample nodes
(stratified by part_id), run GeometricEncoder, verify output.

Usage::

    python test_geometric_encoder.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# make sure the local modules are importable
sys.path.insert(0, str(Path(__file__).parent))
from vtkhdf_reader import load_simulation
from geo_encoders import GeometricEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

VTKHDF = "/mnt/1t/mit-project/Dataset/full_body/neon_full_frontal_ml_0002/field_trajectory.vtkhdf"

N_SUBSAMPLE = 4096      # nodes to keep after subsampling
TIMESTEP    = 0         # which timestep's positions to use

# Radii in mm.  After subsampling to 4096 nodes over a ~4400×4400 mm car,
# the average inter-node spacing is ~60 mm, so we pick three scales:
RADII           = [50.0, 200.0, 600.0]
MAX_NEIGHBORS   = [8, 32, 64]   # expected neighbour counts at each radius
HIDDEN_DIM      = 64


# ─────────────────────────────────────────────────────────────────────────────
# Stratified subsampling — at least one node per part
# ─────────────────────────────────────────────────────────────────────────────

def stratified_subsample(part_id: np.ndarray, n_total: int, seed: int = 42) -> np.ndarray:
    """
    Return n_total node indices sampled so that every part is represented
    by at least one node.  Remaining quota is distributed proportionally
    to part size.

    Parameters
    ----------
    part_id : (N,) int  — part ID per node
    n_total : int       — number of nodes to select
    seed    : int       — random seed for reproducibility

    Returns
    -------
    (n_total,) int  — selected node indices (unsorted)
    """
    rng = np.random.default_rng(seed)
    parts, inverse = np.unique(part_id, return_inverse=True)
    n_parts = len(parts)

    if n_total < n_parts:
        raise ValueError(
            f"n_total ({n_total}) < n_parts ({n_parts}); "
            "cannot give every part at least one node."
        )

    # count nodes per part, assign base quota of 1 each
    counts = np.bincount(inverse)
    quota  = np.ones(n_parts, dtype=int)

    # distribute remaining slots proportionally to part size
    remaining = n_total - n_parts
    prop  = counts / counts.sum()
    extra = np.floor(prop * remaining).astype(int)

    # fix rounding shortfall: give 1 extra to the largest under-represented parts
    shortfall = remaining - extra.sum()
    if shortfall > 0:
        order = np.argsort(-(counts - extra))   # biggest gap first
        extra[order[:shortfall]] += 1

    quota += extra   # (n_parts,) total nodes per part

    # sample from each part
    selected = []
    for i in range(n_parts):
        node_indices = np.where(inverse == i)[0]
        k = min(quota[i], len(node_indices))
        chosen = rng.choice(node_indices, size=k, replace=False)
        selected.append(chosen)

    return np.concatenate(selected)


# ─────────────────────────────────────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load simulation ────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1 — loading simulation …")
    t0 = time.perf_counter()
    data = load_simulation(VTKHDF)
    load_time = time.perf_counter() - t0

    positions_all = data["positions"]   # (T, N, 3)  numpy float32
    part_id_all   = data["part_id"]     # (N,)       numpy int32
    T, N, _ = positions_all.shape

    print(f"  sim_name      : {data['sim_name']}")
    print(f"  timesteps (T) : {T}")
    print(f"  nodes (N)     : {N:,}")
    print(f"  parts         : {len(np.unique(part_id_all))}")
    print(f"  load time     : {load_time:.1f} s")

    # ── 2. Stratified subsample ───────────────────────────────────────────
    print(f"\nStep 2 — stratified subsample → {N_SUBSAMPLE} nodes …")
    idx = stratified_subsample(part_id_all, n_total=N_SUBSAMPLE)

    pos_sub = positions_all[TIMESTEP][idx]    # (N_sub, 3)  numpy
    pid_sub = part_id_all[idx]                # (N_sub,)    numpy

    # remap part IDs to contiguous 0-indexed range (required by nn.Embedding)
    _, pid_remapped = np.unique(pid_sub, return_inverse=True)
    n_parts_sub = int(pid_remapped.max()) + 1

    print(f"  sampled nodes : {len(idx)}")
    print(f"  parts covered : {n_parts_sub}  (out of {len(np.unique(part_id_all))})")

    # count coverage per part
    orig_parts = np.unique(part_id_all)
    covered = len(np.intersect1d(np.unique(pid_sub), orig_parts))
    print(f"  part coverage : {covered}/{len(orig_parts)} parts represented")

    # per-part node count stats in the subsample
    counts_sub = np.bincount(pid_remapped)
    print(f"  nodes/part    : min={counts_sub.min()}  median={int(np.median(counts_sub))}  max={counts_sub.max()}")

    # ── 3. Convert to tensors and run encoder ─────────────────────────────
    print(f"\nStep 3 — running GeometricEncoder …")
    print(f"  radii         : {RADII} mm")
    print(f"  max_neighbors : {MAX_NEIGHBORS}")
    print(f"  hidden_dim    : {HIDDEN_DIM}")

    positions_t = torch.tensor(pos_sub, dtype=torch.float32).unsqueeze(0)  # (1, N_sub, 3)
    part_id_t   = torch.tensor(pid_remapped, dtype=torch.long)             # (N_sub,)

    encoder = GeometricEncoder(
        radii=RADII,
        max_neighbors=MAX_NEIGHBORS,
        n_parts=n_parts_sub,
        part_embed_dim=8,
        hidden_dim=HIDDEN_DIM,
    )
    encoder.eval()

    t1 = time.perf_counter()
    with torch.no_grad():
        context = encoder(positions_t, part_id_t)
    encode_time = time.perf_counter() - t1

    # ── 4. Check output ───────────────────────────────────────────────────
    print(f"\nStep 4 — output checks …")
    print(f"  context shape : {tuple(context.shape)}")
    print(f"  dtype         : {context.dtype}")
    print(f"  encode time   : {encode_time:.2f} s")

    has_nan = context.isnan().any().item()
    has_inf = context.isinf().any().item()
    print(f"  has NaN       : {has_nan}")
    print(f"  has Inf       : {has_inf}")
    print(f"  value range   : [{context.min().item():.4f}, {context.max().item():.4f}]")
    print(f"  mean / std    : {context.mean().item():.4f} / {context.std().item():.4f}")

    assert tuple(context.shape) == (1, N_SUBSAMPLE, HIDDEN_DIM), "shape mismatch"
    assert not has_nan, "NaN values in output"
    assert not has_inf, "Inf values in output"

    # ── 5. Per-scale density check ────────────────────────────────────────
    print(f"\nStep 5 — neighbour density at each radius (sampled on 256 nodes) …")
    sample256 = positions_t[:, :256, :]
    dists256  = torch.cdist(sample256, positions_t)[0]    # (256, N_sub)
    for r, k_max in zip(RADII, MAX_NEIGHBORS):
        n_nbrs = ((dists256 < r) & (dists256 > 0)).float().sum(-1)
        pct_empty = (n_nbrs == 0).float().mean().item() * 100
        print(
            f"  r={r:6.0f} mm  mean_nbrs={n_nbrs.mean():.1f}  "
            f"max_nbrs={n_nbrs.max():.0f}  isolated={pct_empty:.1f}%"
        )

    print("\n" + "=" * 60)
    print("All checks passed.")


if __name__ == "__main__":
    main()
