import torch
import torch.nn as nn
from .utils import _ball_query_stats

class GeometricEncoder(nn.Module):
    """
    Per-node geometric context encoder.

    Runs ball queries at multiple radii in parallel, collects seven
    statistical descriptors per scale, concatenates them with a learned 
    part-ID embedding, and projects through an MLP.
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

        self.hidden_dim = hidden_dim

        # part-ID embedding: integer → dense vector
        self.part_embed = nn.Embedding(n_parts, part_embed_dim)

        # MLP: concatenated stats + part embedding → hidden_dim
        # 7 statistical descriptors per scale
        mlp_in = 7 * len(self.radii) + part_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def precompute(self, positions: torch.Tensor, part_id: torch.Tensor) -> list[torch.Tensor]:
        """
        Precompute the ball-query statistics for one-time O(N^2) training speedup.
        """
        self.eval()
        with torch.no_grad():
            stats = [
                _ball_query_stats(positions, part_id, r, k)
                for r, k in zip(self.radii, self.max_neighbors)
            ]
        return stats

    def forward(
        self,
        positions: torch.Tensor,   # (B, N, 3)
        part_id: torch.Tensor,     # (N,)
        precomputed_stats: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:             # (B, N, hidden_dim)

        B = positions.shape[0]

        # statistical descriptors at each scale: each (B, N, 7)
        if precomputed_stats is not None:
            scale_stats = precomputed_stats
        else:
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
