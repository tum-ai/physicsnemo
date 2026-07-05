"""
run_comparison.py
=================
Train Baseline GeoTransolver vs Enhanced GeoTransolver (+ geometric encoder)
on 5 full-body crash simulations.  Report validation loss every 10 epochs.
Save results to results.csv.

Key design decisions
--------------------
- Task: one-shot prediction of displacement (pos[t] - pos[0]) at all T-1
  future timesteps, given the initial node positions.
- N = 4096 nodes per sim (stratified subsample by part_id).
- Positions normalised by /1000 (mm → ~unit scale).
- Geometric stats (O(N²)) are precomputed once per sim before training;
  only the encoder MLP runs inside each training step (fast, O(N)).
- Both models share the same backbone architecture so parameter counts
  differ only by the encoder and the wider geometry projection.

Runtime: ~20-40 min on CPU · ~5-10 min on GPU
"""

import sys
import time
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from vtkhdf_reader import load_simulation
from geometric_encoder import GeometricEncoder, _ball_query_stats
from physicsnemo.experimental.models.geotransolver import GeoTransolver

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path("/mnt/1t/mit-project/Dataset/full_body")
OUT_CSV   = Path(__file__).parent / "results.csv"

TRAIN_SIMS = [
    "neon_full_frontal_ml_0002",
    "neon_full_frontal_ml_0003",
    "neon_full_frontal_ml_0005",
    "neon_full_frontal_ml_0007",
    "neon_full_frontal_ml_0008",
]
VAL_SIM = "neon_full_frontal_ml_0010"

N_SUB     = 4096      # nodes per simulation after subsampling
EPOCHS    = 200
LOG_EVERY = 10        # validate every N epochs
LR        = 1e-3
SEED      = 42

T_STEPS = 17           # timesteps per simulation
N_PRED  = T_STEPS - 1  # predict t = 1 … T-1
OUT_DIM = N_PRED * 3   # 48: all displacements flattened

POS_SCALE = 1000.0     # divide positions by this (mm → ~unit scale)

# Backbone architecture — shared by both models
N_HIDDEN  = 128
N_LAYERS  = 4
N_HEAD    = 8
SLICE_NUM = 32

# Geometric encoder (enhanced model only)
ENC_RADII     = [0.05, 0.20, 0.60]   # in normalised units (pos / POS_SCALE)
ENC_MAX_K     = [8, 32, 64]
ENC_HDIM      = 32                    # appended to geometry dim
ENC_PART_EDIM = 8
ENC_N_PARTS   = 300                   # must be > max part_id

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Stratified subsampling  (at least one node per part)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_subsample(part_id: np.ndarray, n_total: int, seed: int = 42) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    _, inv = np.unique(part_id, return_inverse=True)
    n_p    = inv.max() + 1
    counts = np.bincount(inv)
    quota  = np.ones(n_p, dtype=int)
    rem    = n_total - n_p
    extra  = np.floor(counts / counts.sum() * rem).astype(int)
    short  = rem - extra.sum()
    extra[np.argsort(-(counts - extra))[:short]] += 1
    quota += extra
    sel = []
    for i in range(n_p):
        idx = np.where(inv == i)[0]
        sel.append(rng.choice(idx, size=min(quota[i], len(idx)), replace=False))
    return np.concatenate(sel)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sim(name: str) -> dict | None:
    vtkhdf = DATA_ROOT / name / "field_trajectory.vtkhdf"
    if not vtkhdf.exists():
        print(f"  WARNING: {name} not found — skipping")
        return None

    print(f"  {name} ...", end="", flush=True)
    t0   = time.perf_counter()
    data = load_simulation(str(vtkhdf))
    print(f" {time.perf_counter() - t0:.1f}s")

    pos  = data["positions"]     # (T, N, 3) float32
    pid  = data["part_id"]       # (N,)      int32

    idx  = stratified_subsample(pid, N_SUB, seed=SEED)
    pos  = pos[:, idx, :].astype(np.float32) / POS_SCALE   # (T, N_sub, 3) normalised
    pid  = pid[idx]

    _, pid_remap = np.unique(pid, return_inverse=True)       # 0-indexed contiguous

    # displacement target: pos[t] − pos[0] for t = 1…T-1
    disp      = pos[1:] - pos[0]                             # (N_PRED, N, 3)
    disp_flat = disp.transpose(1, 0, 2).reshape(N_SUB, -1)    # (N, N_PRED * 3)

    return {
        "name":    name,
        "pos0":    torch.tensor(pos[0]).unsqueeze(0).to(DEVICE),                  # (1, N, 3)
        "target":  torch.tensor(disp_flat).unsqueeze(0).to(DEVICE),              # (1, N, 48)
        "part_id": torch.tensor(pid_remap, dtype=torch.long).to(DEVICE),         # (N,)
        "n_parts": int(pid_remap.max()) + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────────────────

def _backbone_kwargs(geometry_dim: int) -> dict:
    return dict(
        functional_dim=3,
        out_dim=OUT_DIM,
        geometry_dim=geometry_dim,
        global_dim=None,
        n_layers=N_LAYERS,
        n_hidden=N_HIDDEN,
        n_head=N_HEAD,
        slice_num=SLICE_NUM,
        use_te=False,
        include_local_features=False,
    )


def build_baseline() -> GeoTransolver:
    return GeoTransolver(**_backbone_kwargs(geometry_dim=3)).to(DEVICE)


def build_enhanced() -> tuple[GeoTransolver, GeometricEncoder]:
    backbone = GeoTransolver(**_backbone_kwargs(geometry_dim=3 + ENC_HDIM)).to(DEVICE)
    encoder  = GeometricEncoder(
        radii=ENC_RADII,
        max_neighbors=ENC_MAX_K,
        n_parts=ENC_N_PARTS,
        part_embed_dim=ENC_PART_EDIM,
        hidden_dim=ENC_HDIM,
    ).to(DEVICE)
    return backbone, encoder


# ─────────────────────────────────────────────────────────────────────────────
# Stats-only encoder (ablation: no part_id, no same_part_frac feature)
# ─────────────────────────────────────────────────────────────────────────────

class StatsOnlyEncoder(nn.Module):
    """
    Ablation encoder: uses only the 6 geometric statistics per scale
    (mean_dist, std_dist, density, pca_eig_0, pca_eig_1, pca_eig_2).
    Drops same_part_frac (index 6) and uses no part_id embedding.

    Input : list of n_scales × (1, N, 7) cached raw stats
    Output: (1, N, hidden_dim)
    """
    N_STATS = 6   # columns 0-5; column 6 (same_part_frac) is excluded

    def __init__(self, n_scales: int, hidden_dim: int) -> None:
        super().__init__()
        mlp_in = self.N_STATS * n_scales
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cached_stats: list[torch.Tensor]) -> torch.Tensor:
        trimmed = [s[..., : self.N_STATS] for s in cached_stats]  # drop col 6
        return self.mlp(torch.cat(trimmed, dim=-1))                # (1, N, hidden_dim)


def build_stats_only() -> tuple[GeoTransolver, StatsOnlyEncoder]:
    backbone = GeoTransolver(**_backbone_kwargs(geometry_dim=3 + ENC_HDIM)).to(DEVICE)
    encoder  = StatsOnlyEncoder(n_scales=len(ENC_RADII), hidden_dim=ENC_HDIM).to(DEVICE)
    return backbone, encoder


# ─────────────────────────────────────────────────────────────────────────────
# Precompute ball-query statistics  (O(N²), done once before training)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_stats(sims: list[dict], encoder: GeometricEncoder) -> dict:
    """
    For each simulation, compute raw per-node ball-query statistics.
    Positions don't change within a simulation, so this only runs once.
    Returns: {sim_name: [list of (1, N, 7) tensors, one per radius scale]}
    """
    print("\nPrecomputing geometric stats (one-time O(N²) cost) …")
    cache = {}
    encoder.eval()
    for sim in sims:
        t0 = time.perf_counter()
        with torch.no_grad():
            cache[sim["name"]] = [
                _ball_query_stats(sim["pos0"], sim["part_id"], r, k)
                for r, k in zip(encoder.radii, encoder.max_neighbors)
            ]
        print(f"  {sim['name']}: {time.perf_counter() - t0:.1f}s")
    encoder.train()
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# Forward passes
# ─────────────────────────────────────────────────────────────────────────────

def fwd_baseline(model: GeoTransolver, sim: dict) -> torch.Tensor:
    return model(local_embedding=sim["pos0"], geometry=sim["pos0"])


def fwd_enhanced(
    backbone: GeoTransolver,
    encoder: GeometricEncoder,
    sim: dict,
    stats_cache: dict,
) -> torch.Tensor:
    """Stats + part_id: run encoder MLP on cached stats + part embedding."""
    cached   = stats_cache[sim["name"]]                              # list of (1,N,7)
    part_emb = encoder.part_embed(sim["part_id"]).unsqueeze(0)       # (1, N, part_dim)
    feats    = torch.cat(cached + [part_emb], dim=-1)                # (1, N, 7*n_scales + part_dim)
    geo_ctx  = encoder.mlp(feats)                                    # (1, N, ENC_HDIM)
    enh_geom = torch.cat([sim["pos0"], geo_ctx], dim=-1)             # (1, N, 3 + ENC_HDIM)
    return backbone(local_embedding=sim["pos0"], geometry=enh_geom)


def fwd_stats_only(
    backbone: GeoTransolver,
    encoder: StatsOnlyEncoder,
    sim: dict,
    stats_cache: dict,
) -> torch.Tensor:
    """Stats only (ablation): no part_id, same_part_frac dropped."""
    geo_ctx  = encoder(stats_cache[sim["name"]])                     # (1, N, ENC_HDIM)
    enh_geom = torch.cat([sim["pos0"], geo_ctx], dim=-1)
    return backbone(local_embedding=sim["pos0"], geometry=enh_geom)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    train_sims: list[dict],
    optimizer: optim.Optimizer,
    fwd_fn,           # callable(sim) -> pred tensor
) -> float:
    losses = []
    for sim in train_sims:
        optimizer.zero_grad()
        pred = fwd_fn(sim)
        loss = F.mse_loss(pred, sim["target"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(val_sim: dict, fwd_fn) -> tuple[float, float]:
    pred   = fwd_fn(val_sim)
    target = val_sim["target"]
    mse    = F.mse_loss(pred, target).item()
    rel_l2 = (torch.norm(pred - target) / (torch.norm(target) + 1e-8)).item()
    return mse, rel_l2


def run_training(
    label: str,
    train_sims: list[dict],
    val_sim: dict,
    optimizer: optim.Optimizer,
    fwd_fn,                      # forward function (same signature for train & eval)
    set_train_mode,              # callable: switch model(s) to train mode
    set_eval_mode,               # callable: switch model(s) to eval mode
) -> list[dict]:
    """
    Train for EPOCHS, log every LOG_EVERY epochs.
    Returns list of result dicts.
    """
    print(f"\n{'='*62}")
    print(f"  {label}")
    print(f"{'='*62}")

    rows = []
    window_losses = []           # accumulate train losses within each log window

    for epoch in range(1, EPOCHS + 1):
        set_train_mode()
        train_loss = train_one_epoch(train_sims, optimizer, fwd_fn)
        window_losses.append(train_loss)

        if epoch % LOG_EVERY == 0:
            set_eval_mode()
            val_mse, val_rl2 = evaluate(val_sim, fwd_fn)
            avg_train = float(np.mean(window_losses))
            window_losses.clear()

            rows.append({
                "epoch":      epoch,
                "train_mse":  avg_train,
                "val_mse":    val_mse,
                "val_rel_l2": val_rl2,
            })
            print(
                f"  epoch {epoch:3d}/{EPOCHS}"
                f"  train_mse={avg_train:.5f}"
                f"  val_mse={val_mse:.5f}"
                f"  val_rel_l2={val_rl2:.4f}"
            )

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    wandb.init(
        project="geotransolver-crash",
        config={
            "n_sub":       N_SUB,
            "epochs":      EPOCHS,
            "lr":          LR,
            "n_hidden":    N_HIDDEN,
            "n_layers":    N_LAYERS,
            "n_head":      N_HEAD,
            "slice_num":   SLICE_NUM,
            "enc_radii":   ENC_RADII,
            "enc_max_k":   ENC_MAX_K,
            "enc_hdim":    ENC_HDIM,
            "enc_part_edim": ENC_PART_EDIM,
            "enc_n_parts": ENC_N_PARTS,
            "pos_scale":   POS_SCALE,
            "train_sims":  TRAIN_SIMS,
            "val_sim":     VAL_SIM,
            "device":      str(DEVICE),
        },
    )

    print(f"Device : {DEVICE}")
    print(f"N_SUB  : {N_SUB}  |  EPOCHS: {EPOCHS}  |  LR: {LR}")

    # ── load data ─────────────────────────────────────────────────────────
    print("\nLoading simulations …")
    train_sims = [d for name in TRAIN_SIMS if (d := load_sim(name)) is not None]
    val_sim    = load_sim(VAL_SIM)
    if val_sim is None:
        raise RuntimeError(f"Validation simulation {VAL_SIM} not found in {DATA_ROOT}")
    print(f"Loaded {len(train_sims)} training + 1 validation simulation")

    # ── build models ──────────────────────────────────────────────────────
    print("\nBuilding models …")
    baseline              = build_baseline()
    so_backbone, so_enc   = build_stats_only()
    enh_backbone, enh_enc = build_enhanced()

    def n_params(*modules):
        return sum(sum(p.numel() for p in m.parameters()) for m in modules)

    n_base = n_params(baseline)
    n_so   = n_params(so_backbone, so_enc)
    n_enh  = n_params(enh_backbone, enh_enc)
    print(f"  baseline         params : {n_base:,}")
    print(f"  stats only       params : {n_so:,}  ({n_so - n_base:+,} vs baseline)")
    print(f"  stats + part_id  params : {n_enh:,}  ({n_enh - n_base:+,} vs baseline)")

    # ── precompute geometric stats (shared by both enhanced models) ───────
    all_sims    = train_sims + [val_sim]
    stats_cache = precompute_stats(all_sims, enh_enc)  # uses enh_enc for radii/max_k

    # ── optimizers ────────────────────────────────────────────────────────
    base_opt = optim.Adam(baseline.parameters(), lr=LR)
    so_opt   = optim.Adam(list(so_backbone.parameters()) + list(so_enc.parameters()), lr=LR)
    enh_opt  = optim.Adam(list(enh_backbone.parameters()) + list(enh_enc.parameters()), lr=LR)

    # ── train baseline ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    base_rows = run_training(
        label          = "Baseline  (XYZ only)",
        train_sims     = train_sims,
        val_sim        = val_sim,
        optimizer      = base_opt,
        fwd_fn         = lambda sim: fwd_baseline(baseline, sim),
        set_train_mode = baseline.train,
        set_eval_mode  = baseline.eval,
    )
    print(f"  training time: {(time.perf_counter()-t0)/60:.1f} min")

    # ── train stats-only (ablation) ───────────────────────────────────────
    t0 = time.perf_counter()
    so_rows = run_training(
        label          = "Stats only  (XYZ + geometric stats, no part_id)",
        train_sims     = train_sims,
        val_sim        = val_sim,
        optimizer      = so_opt,
        fwd_fn         = lambda sim: fwd_stats_only(so_backbone, so_enc, sim, stats_cache),
        set_train_mode = lambda: (so_backbone.train(), so_enc.train()),
        set_eval_mode  = lambda: (so_backbone.eval(), so_enc.eval()),
    )
    print(f"  training time: {(time.perf_counter()-t0)/60:.1f} min")

    # ── train stats + part_id ─────────────────────────────────────────────
    t0 = time.perf_counter()
    enh_rows = run_training(
        label          = "Stats + part_id  (XYZ + geometric stats + part embedding)",
        train_sims     = train_sims,
        val_sim        = val_sim,
        optimizer      = enh_opt,
        fwd_fn         = lambda sim: fwd_enhanced(enh_backbone, enh_enc, sim, stats_cache),
        set_train_mode = lambda: (enh_backbone.train(), enh_enc.train()),
        set_eval_mode  = lambda: (enh_backbone.eval(), enh_enc.eval()),
    )
    print(f"  training time: {(time.perf_counter()-t0)/60:.1f} min")

    # ── write CSV ─────────────────────────────────────────────────────────
    fieldnames = [
        "epoch",
        "baseline_train_mse",  "baseline_val_mse",  "baseline_val_rel_l2",
        "statsonly_train_mse", "statsonly_val_mse", "statsonly_val_rel_l2",
        "enhanced_train_mse",  "enhanced_val_mse",  "enhanced_val_rel_l2",
    ]
    base_by_e = {r["epoch"]: r for r in base_rows}
    so_by_e   = {r["epoch"]: r for r in so_rows}
    enh_by_e  = {r["epoch"]: r for r in enh_rows}
    epochs    = sorted(base_by_e)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in epochs:
            b, s, en = base_by_e[e], so_by_e[e], enh_by_e[e]
            writer.writerow({
                "epoch":                e,
                "baseline_train_mse":   f"{b['train_mse']:.6f}",
                "baseline_val_mse":     f"{b['val_mse']:.6f}",
                "baseline_val_rel_l2":  f"{b['val_rel_l2']:.6f}",
                "statsonly_train_mse":  f"{s['train_mse']:.6f}",
                "statsonly_val_mse":    f"{s['val_mse']:.6f}",
                "statsonly_val_rel_l2": f"{s['val_rel_l2']:.6f}",
                "enhanced_train_mse":   f"{en['train_mse']:.6f}",
                "enhanced_val_mse":     f"{en['val_mse']:.6f}",
                "enhanced_val_rel_l2":  f"{en['val_rel_l2']:.6f}",
            })

    # ── log all epochs to wandb (aligned step = epoch) ───────────────────
    for e in epochs:
        b, s, en = base_by_e[e], so_by_e[e], enh_by_e[e]
        wandb.log(
            {
                "baseline/train_mse":    b["train_mse"],
                "baseline/val_mse":      b["val_mse"],
                "baseline/val_rel_l2":   b["val_rel_l2"],
                "statsonly/train_mse":   s["train_mse"],
                "statsonly/val_mse":     s["val_mse"],
                "statsonly/val_rel_l2":  s["val_rel_l2"],
                "enhanced/train_mse":    en["train_mse"],
                "enhanced/val_mse":      en["val_mse"],
                "enhanced/val_rel_l2":   en["val_rel_l2"],
            },
            step=e,
        )

    wandb.summary["baseline/best_val_rel_l2"]  = min(r["val_rel_l2"] for r in base_rows)
    wandb.summary["statsonly/best_val_rel_l2"]  = min(r["val_rel_l2"] for r in so_rows)
    wandb.summary["enhanced/best_val_rel_l2"]   = min(r["val_rel_l2"] for r in enh_rows)
    wandb.summary["baseline/params"]  = n_base
    wandb.summary["statsonly/params"] = n_so
    wandb.summary["enhanced/params"]  = n_enh

    wandb.finish()

    # ── final table ───────────────────────────────────────────────────────
    W = 88
    print(f"\nResults saved to {OUT_CSV}")
    print("\n" + "─" * W)
    print(f"{'Epoch':>5} │ {'Baseline':^18} │ {'Stats only':^18} │ {'Stats+part_id':^18} │")
    print(f"{'':>5} │ {'train    val_rl2':^18} │ {'train    val_rl2':^18} │ {'train    val_rl2':^18} │")
    print("─" * W)
    for e in epochs:
        b, s, en = base_by_e[e], so_by_e[e], enh_by_e[e]
        print(
            f"{e:>5} │"
            f" {b['train_mse']:.4f}  {b['val_rel_l2']:.4f} │"
            f" {s['train_mse']:.4f}  {s['val_rel_l2']:.4f} │"
            f" {en['train_mse']:.4f}  {en['val_rel_l2']:.4f} │"
            f"  Δso={s['val_rel_l2']-b['val_rel_l2']:+.4f}"
            f"  Δenh={en['val_rel_l2']-b['val_rel_l2']:+.4f}"
        )
    print("─" * W)
    print("Negative Δ = better than baseline.\n")


if __name__ == "__main__":
    main()
