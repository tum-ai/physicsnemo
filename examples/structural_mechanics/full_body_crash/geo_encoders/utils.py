import torch

_STATS_PER_SCALE = 7

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
