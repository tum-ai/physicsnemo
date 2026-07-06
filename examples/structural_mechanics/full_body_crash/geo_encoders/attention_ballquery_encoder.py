import torch
import torch.nn as nn

from physicsnemo.nn import BQWarp


class AttentionBallQueryEncoder(nn.Module):
    """
    Per-node geometric context encoder using attention-weighted ball-query
    aggregation instead of fixed statistical descriptors (mean/std/PCA).

    For each radius scale, BQWarp finds up to `max_neighbors` neighbors per
    node (including the node itself). Neighbor relative positions/distances
    become the Key/Value input to a multi-head attention block; the Query
    is derived from the node's own part-ID embedding. Padded neighbor slots
    (when a node has fewer than `max_neighbors` real neighbors) are masked
    out of the softmax using a distance check (BQWarp pads by duplicating
    an existing point, so ``dist <= radius`` reliably marks real slots).

    Scale outputs are concatenated with the part embedding and projected
    through an MLP to the final context vector, matching the interface of
    ``GeometricEncoder`` / ``StatsOnlyEncoder``.
    """

    def __init__(
        self,
        radii: list[float] | None = None,
        max_neighbors: list[int] | int = 32,
        n_parts: int = 1200,
        part_embed_dim: int = 8,
        hidden_dim: int = 64,
        n_heads: int = 4,
    ) -> None:
        super().__init__()

        self.radii = [0.1, 0.25, 1.0] if radii is None else list(radii)

        if isinstance(max_neighbors, int):
            self.max_neighbors = [max_neighbors] * len(self.radii)
        else:
            if len(max_neighbors) != len(self.radii):
                raise ValueError("max_neighbors must be an int or have the same length as radii")
            self.max_neighbors = list(max_neighbors)

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.head_dim = hidden_dim // n_heads

        self.bq_warps = nn.ModuleList([
            BQWarp(radius=r, neighbors_in_radius=k)
            for r, k in zip(self.radii, self.max_neighbors)
        ])

        # part-ID embedding: integer -> dense vector (also used to build the attention Query)
        self.part_embed = nn.Embedding(n_parts, part_embed_dim)
        self.query_proj = nn.Linear(part_embed_dim, hidden_dim)

        # Key/Value input per neighbor: [rel_pos (3) / radius, dist (1) / radius] = 4 dims
        self.kv_proj = nn.ModuleList([
            nn.Linear(4, hidden_dim * 2) for _ in self.radii
        ])

        # per-scale attended features + part embedding -> hidden_dim
        mlp_in = hidden_dim * len(self.radii) + part_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _neighbor_features(
        self, positions: torch.Tensor, scale_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run BQWarp for one scale and build (rel_feat, valid_mask)."""
        radius = self.radii[scale_idx]
        bq_warp = self.bq_warps[scale_idx]

        _, neighbor_pos = bq_warp(positions, positions)          # (B, N, K, 3)
        query = positions.unsqueeze(2)                            # (B, N, 1, 3)
        rel_pos = neighbor_pos - query                            # (B, N, K, 3)
        dist = rel_pos.norm(dim=-1, keepdim=True)                 # (B, N, K, 1)

        valid = (dist.squeeze(-1) <= radius + 1e-4)               # (B, N, K)  BQWarp pads by
                                                                    # duplicating an existing point;
                                                                    # padded slots fall outside radius
                                                                    # when fewer than K real neighbors exist.

        rel_feat = torch.cat([rel_pos / radius, dist / radius], dim=-1)  # (B, N, K, 4)
        return rel_feat, valid

    def precompute(self, positions: torch.Tensor, part_id: torch.Tensor) -> list[torch.Tensor]:
        """
        Precompute the (static) ball-query neighbor geometry once per simulation.
        Only the neighbor search runs here; the learnable attention runs in forward().
        """
        self.eval()
        with torch.no_grad():
            cached = []
            for scale_idx in range(len(self.radii)):
                rel_feat, valid = self._neighbor_features(positions, scale_idx)
                cached.append(rel_feat)
                cached.append(valid)
        return cached

    def forward(
        self,
        positions: torch.Tensor,   # (B, N, 3)
        part_id: torch.Tensor,     # (N,)
        precomputed_stats: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:             # (B, N, hidden_dim)

        B, N, _ = positions.shape
        H, D = self.n_heads, self.head_dim

        part_emb = self.part_embed(part_id)                      # (N, part_embed_dim)
        part_emb = part_emb.unsqueeze(0).expand(B, -1, -1)        # (B, N, part_embed_dim)
        query = self.query_proj(part_emb)                         # (B, N, hidden_dim)
        query = query.view(B, N, H, D).transpose(1, 2)            # (B, H, N, D)

        scale_outputs = []
        for scale_idx in range(len(self.radii)):
            if precomputed_stats is not None:
                rel_feat = precomputed_stats[2 * scale_idx]
                valid = precomputed_stats[2 * scale_idx + 1]
            else:
                rel_feat, valid = self._neighbor_features(positions, scale_idx)

            K = rel_feat.shape[2]
            kv = self.kv_proj[scale_idx](rel_feat)                 # (B, N, K, 2*hidden_dim)
            k, v = kv.chunk(2, dim=-1)                             # each (B, N, K, hidden_dim)
            k = k.view(B, N, K, H, D).permute(0, 3, 1, 2, 4)       # (B, H, N, K, D)
            v = v.view(B, N, K, H, D).permute(0, 3, 1, 2, 4)       # (B, H, N, K, D)

            # scaled dot-product attention: query (B,H,N,D) attends over K neighbors
            attn_logits = torch.einsum("bhnd,bhnkd->bhnk", query, k) * (D ** -0.5)
            attn_logits = attn_logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
            # nodes with zero valid neighbors would produce an all -inf row; guard with a safe fallback
            no_valid = (~valid).all(dim=-1, keepdim=True).unsqueeze(1)  # (B,1,N,1)... broadcast fix below
            attn_logits = torch.where(
                no_valid.expand_as(attn_logits), torch.zeros_like(attn_logits), attn_logits
            )
            attn_weights = torch.softmax(attn_logits, dim=-1)      # (B, H, N, K)

            attended = torch.einsum("bhnk,bhnkd->bhnd", attn_weights, v)  # (B, H, N, D)
            attended = attended.transpose(1, 2).reshape(B, N, H * D)      # (B, N, hidden_dim)
            scale_outputs.append(attended)

        feats = torch.cat(scale_outputs + [part_emb], dim=-1)
        return self.mlp(feats)
