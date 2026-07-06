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
    chunk_size: int = 1024,     # Process query nodes in chunks to avoid CUDA OOM on large meshes
) -> torch.Tensor:
    """Compute all per-node statistics for one radius scale in memory-efficient chunks."""
    B, N, _ = positions.shape
    device = positions.device
    dtype = positions.dtype

    stats_out = torch.zeros(B, N, _STATS_PER_SCALE, device=device, dtype=dtype)

    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        chunk_len = end_idx - start_idx

        # ── query slice ───────────────────────────────────────────────────
        q_pos = positions[:, start_idx:end_idx, :]             # (B, chunk_len, 3)

        # ── pairwise distances (chunk to all) ─────────────────────────────
        dists = torch.cdist(q_pos, positions)                  # (B, chunk_len, N)

        # exclude self-distance by comparing global indices
        cols = torch.arange(N, device=device).unsqueeze(0).unsqueeze(0)               # (1, 1, N)
        rows = torch.arange(start_idx, end_idx, device=device).unsqueeze(0).unsqueeze(2) # (1, chunk_len, 1)
        is_self = (cols == rows)                               # (1, chunk_len, N) bool

        in_radius = (dists < radius) & ~is_self                 # (B, chunk_len, N) bool

        count = in_radius.float().sum(dim=-1)                  # (B, chunk_len)
        has_nbrs = count > 0                                   # (B, chunk_len)
        safe_count = count.clamp(min=1)

        # ── mean and std distance ─────────────────────────────────────────
        masked_d = dists * in_radius.float()
        mean_d = masked_d.sum(-1) / safe_count                 # (B, chunk_len)

        sq_diff = (dists - mean_d.unsqueeze(-1)).pow(2) * in_radius.float()
        std_d = (sq_diff.sum(-1) / safe_count).sqrt()          # (B, chunk_len)

        # ── PCA eigenvalues of relative neighbour positions ───────────────
        # rel[b, i, j] = positions[b, j] - q_pos[b, i]
        rel = positions.unsqueeze(1) - q_pos.unsqueeze(2)      # (B, chunk_len, N, 3)
        rel_masked = rel * in_radius.float().unsqueeze(-1)     # zero non-neighbours
        eigs = _pca_eigenvalues(rel_masked, safe_count)        # (B, chunk_len, 3)

        # ── part-aware: fraction of neighbours with the same part_id ──────
        q_pid = part_id[start_idx:end_idx].unsqueeze(0).unsqueeze(2)  # (1, chunk_len, 1)
        all_pid = part_id.unsqueeze(0).unsqueeze(1)                  # (1, 1, N)
        same_part = (q_pid == all_pid)                               # (1, chunk_len, N) bool
        same_count = (same_part & in_radius).float().sum(-1)         # (B, chunk_len)
        same_frac = same_count / safe_count                          # (B, chunk_len)

        # ── assemble, normalise, and zero isolated nodes ──────────────────
        stats = torch.stack([
            mean_d / radius,
            std_d  / radius,
            count  / max_neighbors,
            eigs[..., 0],
            eigs[..., 1],
            eigs[..., 2],
            same_frac,
        ], dim=-1)                                             # (B, chunk_len, 7)

        # store chunk and apply neighbor mask
        stats_out[:, start_idx:end_idx, :] = stats * has_nbrs.float().unsqueeze(-1)

    return stats_out
