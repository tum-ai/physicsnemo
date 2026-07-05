"""
Standalone geometric context encoder for unstructured mesh simulations.

For each node, computes statistical descriptors from ball-query neighborhoods
at three spatial scales, then projects everything through an MLP to a
fixed-size context vector that can be injected into GeoTransolver.

Usage::

    encoder = GeometricEncoder(radii=[0.1, 0.25, 1.0], hidden_dim=64, n_parts=1200)
    context = encoder(positions, part_id)   # (B, N, 64)

Memory note
-----------
Distance computation is O(N²).  Keep N ≤ ~8 000 per batch element, or
subsample the point cloud before calling this encoder.  For the full-body
crash dataset (N ~ 1M) you will always want to subsample first.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Statistics computed per node per scale
# ─────────────────────────────────────────────────────────────────────────────

_STATS_PER_SCALE = 7
# [0] mean_dist       mean distance to neighbors, normalised by radius
# [1] std_dist        std of distances, normalised by radius
# [2] density         neighbour count / max_neighbors
# [3] pca_eig_0       largest PCA eigenvalue (fraction of total variance)
# [4] pca_eig_1
# [5] pca_eig_2
# [6] same_part_frac  fraction of neighbours sharing this node's part_id


def _pca_eigenvalues(
    rel_masked: torch.Tensor,   # (B, N, N, 3)  relative positions, non-neighbors zeroed
    safe_count: torch.Tensor,   # (B, N)        neighbour count, clamped ≥ 1
) -> torch.Tensor:              # (B, N, 3)     normalised eigenvalues, descending
    """Eigenvalues of the per-node neighbour covariance matrix."""
    # covariance: (B, N, 3, 3)  via batched outer-product sum
    cov = torch.einsum("bnkd,bnke->bnde", rel_masked, rel_masked)
    cov = cov / safe_count[:, :, None, None]

    # eigvalsh returns ascending order on a symmetric matrix
    eigs = torch.linalg.eigvalsh(cov).flip(-1)          # (B, N, 3) descending

    # normalise so eigenvalues sum to 1 (share of total variance)
    trace = eigs.sum(-1, keepdim=True).clamp(min=1e-8)
    return eigs / trace


def _ball_query_stats(
    positions: torch.Tensor,    # (B, N, 3)
    part_id: torch.Tensor,      # (N,)   long
    radius: float,
    max_neighbors: int,
) -> torch.Tensor:              # (B, N, _STATS_PER_SCALE)
    """Compute all per-node statistics for one radius scale."""
    B, N, _ = positions.shape
    device = positions.device

    # ── pairwise distances ────────────────────────────────────────────────
    dists = torch.cdist(positions, positions)              # (B, N, N)

    # exclude self-distance; keep only in-radius neighbours
    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    in_radius = (dists < radius) & ~eye                    # (B, N, N) bool

    count = in_radius.float().sum(dim=-1)                  # (B, N)
    has_nbrs = count > 0                                   # (B, N)
    safe_count = count.clamp(min=1)

    # ── mean and std distance ─────────────────────────────────────────────
    masked_d = dists * in_radius.float()
    mean_d = masked_d.sum(-1) / safe_count                 # (B, N)

    sq_diff = (dists - mean_d.unsqueeze(-1)).pow(2) * in_radius.float()
    std_d = (sq_diff.sum(-1) / safe_count).sqrt()          # (B, N)

    # ── PCA eigenvalues of relative neighbour positions ───────────────────
    # rel[b, i, j] = position[b, j] - position[b, i]  (vector from i to j)
    rel = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)
    rel_masked = rel * in_radius.float().unsqueeze(-1)     # zero non-neighbours
    eigs = _pca_eigenvalues(rel_masked, safe_count)        # (B, N, 3)

    # ── part-aware: fraction of neighbours with the same part_id ─────────
    pid = part_id.unsqueeze(0)                             # (1, N)
    same_part = (pid.unsqueeze(-1) == pid.unsqueeze(-2))   # (1, N, N) bool
    same_count = (same_part & in_radius).float().sum(-1)   # (B, N)
    same_frac = same_count / safe_count                    # (B, N)

    # ── assemble, normalise, and zero isolated nodes ──────────────────────
    stats = torch.stack([
        mean_d / radius,
        std_d  / radius,
        count  / max_neighbors,
        eigs[..., 0],
        eigs[..., 1],
        eigs[..., 2],
        same_frac,
    ], dim=-1)                                             # (B, N, 7)

    # nodes with no neighbours at this scale get all-zero stats
    return stats * has_nbrs.float().unsqueeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class GeometricEncoder(nn.Module):
    """
    Per-node geometric context encoder.

    Runs ball queries at multiple radii in parallel, collects seven
    statistical descriptors per scale (see _STATS_PER_SCALE above),
    concatenates them with a learned part-ID embedding, and projects
    through an MLP to produce a context vector of shape (B, N, hidden_dim).

    Parameters
    ----------
    radii : list[float]
        Ball-query radii (one scale per entry).
    max_neighbors : int or list[int]
        Expected maximum neighbour count per radius, used to normalise the
        density descriptor.  A single int is broadcast to all radii.
    n_parts : int
        Size of the part-ID embedding table.  Must be > largest part_id.
    part_embed_dim : int
        Embedding dimension for part IDs.
    hidden_dim : int
        Output feature dimension.

    Inputs
    ------
    positions : (B, N, 3)  float  — absolute node XYZ coordinates
    part_id   : (N,)       long   — integer part ID per node, 0-indexed

    Output
    ------
    context : (B, N, hidden_dim)
    """

    def __init__(
        self,
        radii: list[float] | None = None,
        max_neighbors: list[int] | int = 32,
        n_parts: int = 1200,
        part_embed_dim: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.radii = [0.1, 0.25, 1.0] if radii is None else list(radii)

        if isinstance(max_neighbors, int):
            self.max_neighbors = [max_neighbors] * len(self.radii)
        else:
            if len(max_neighbors) != len(self.radii):
                raise ValueError("max_neighbors must be an int or have the same length as radii")
            self.max_neighbors = list(max_neighbors)

        # part-ID embedding: integer → dense vector
        self.part_embed = nn.Embedding(n_parts, part_embed_dim)

        # MLP: concatenated stats + part embedding → hidden_dim
        mlp_in = _STATS_PER_SCALE * len(self.radii) + part_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        positions: torch.Tensor,   # (B, N, 3)
        part_id: torch.Tensor,     # (N,)
    ) -> torch.Tensor:             # (B, N, hidden_dim)

        B = positions.shape[0]

        # statistical descriptors at each scale: each (B, N, 7)
        scale_stats = [
            _ball_query_stats(positions, part_id, r, k)
            for r, k in zip(self.radii, self.max_neighbors)
        ]

        # part embedding, broadcast over batch dimension
        part_emb = self.part_embed(part_id)                    # (N, part_embed_dim)
        part_emb = part_emb.unsqueeze(0).expand(B, -1, -1)    # (B, N, part_embed_dim)

        # concatenate all features and project
        feats = torch.cat(scale_stats + [part_emb], dim=-1)   # (B, N, mlp_in)
        return self.mlp(feats)                                 # (B, N, hidden_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    B, N = 1, 512
    n_parts = 50
    hidden_dim = 64

    positions = torch.randn(B, N, 3)
    part_id   = torch.randint(0, n_parts, (N,))

    encoder = GeometricEncoder(
        radii=[0.5, 1.5, 3.0],     # use larger radii for random unit-scale data
        max_neighbors=32,
        n_parts=n_parts,
        part_embed_dim=8,
        hidden_dim=hidden_dim,
    )

    context = encoder(positions, part_id)
    print(f"positions : {tuple(positions.shape)}")
    print(f"part_id   : {tuple(part_id.shape)}  (unique: {part_id.unique().numel()})")
    print(f"context   : {tuple(context.shape)}")

    # quick checks
    assert context.shape == (B, N, hidden_dim), "output shape mismatch"
    assert not context.isnan().any(), "NaN in output"
    print("OK — no NaNs, shape correct.")

    # per-scale density sanity check
    print("\nNeighbour counts at each radius (mean over nodes):")
    for r in [0.5, 1.5, 3.0]:
        dists = torch.cdist(positions[0:1], positions[0:1])[0]
        n_nbrs = ((dists < r) & (dists > 0)).float().sum(-1).mean()
        print(f"  r={r:.1f}  mean_neighbours={n_nbrs:.1f}")
