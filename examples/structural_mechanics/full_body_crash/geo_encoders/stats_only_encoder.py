import torch
import torch.nn as nn
from .utils import _ball_query_stats

class StatsOnlyEncoder(nn.Module):
    """
    Ablation encoder: uses only the first 6 geometric statistics per scale
    (mean_dist, std_dist, density, pca_eig_0, pca_eig_1, pca_eig_2).
    Drops same_part_frac (index 6) and uses no part_id embedding.
    """
    N_STATS = 6

    def __init__(
        self,
        radii: list[float] | None = None,
        max_neighbors: list[int] | int = 32,
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

        mlp_in = self.N_STATS * len(self.radii)
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
        positions: torch.Tensor,
        part_id: torch.Tensor,
        precomputed_stats: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if precomputed_stats is not None:
            scale_stats = precomputed_stats
        else:
            scale_stats = [
                _ball_query_stats(positions, part_id, r, k)
                for r, k in zip(self.radii, self.max_neighbors)
            ]

        # Trim the same_part_frac (column 6) out of each scale's stats
        trimmed = [s[..., :self.N_STATS] for s in scale_stats]
        return self.mlp(torch.cat(trimmed, dim=-1))
