"""DECEncoder — per-node geometric context from DEC shape features.

Drop-in alternative to GeometricEncoder / StatsOnlyEncoder: computes seven
static per-node channels on the undeformed (t=0) shell mesh via the cotangent
Laplacian (signed mean curvature, curvature magnitude, unit normal, log lumped
node area, open-boundary flag) and projects them through an MLP to
(B, N, hidden_dim). The result is concatenated onto the geometry tensor by
EnhancedGeoTransolver, enriching the ContextProjector slice tokens.

Unlike the ball-query encoders, the raw features need mesh connectivity, which
only exists at data-loading time — so this encoder is precompute-only:
`precompute(positions, part_id, cells=...)` must be called once per simulation
and its result passed to forward() as `precomputed_stats`.

Normalization: curvature scales as 1/length and log-area is unbounded, so the
continuous channels (h_signed, h_mag, log_area) are z-scored per simulation
over the nodes where they are defined, then clipped to ±clip_sigma to tame
sliver-element outliers. Normals (unit) and the boundary flag (binary) pass
through unchanged; undefined entries stay 0 with the flag channel = 1.
"""

import numpy as np
import torch
import torch.nn as nn

from .dec_features import DEC_NUM_CHANNELS, dec_point_features_flat


def _zscore_masked(x: np.ndarray, mask: np.ndarray, clip: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    if mask.any():
        mu = float(x[mask].mean())
        sd = float(x[mask].std()) + 1e-8
        out[mask] = np.clip((x[mask] - mu) / sd, -clip, clip)
    return out


class DECEncoder(nn.Module):
    """
    Per-node DEC (Discrete Exterior Calculus) context encoder.

    Computes curvature/normal/area descriptors on the t=0 mesh once per
    simulation (`precompute`), then projects the normalized 7-channel
    features through an MLP at forward time.
    """

    # train.py checks this to pass cell connectivity into precompute()
    needs_cells = True

    def __init__(self, hidden_dim: int = 32, clip_sigma: float = 5.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.clip_sigma = clip_sigma

        # MLP: 7 DEC channels → hidden_dim (mirrors GeometricEncoder's head)
        self.mlp = nn.Sequential(
            nn.Linear(DEC_NUM_CHANNELS, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def precompute(
        self,
        positions: torch.Tensor,                       # (B, N, 3), B == 1
        part_id: torch.Tensor | None = None,           # unused (interface parity)
        cells: tuple[np.ndarray, np.ndarray] | None = None,  # (conn, offsets)
    ) -> list[torch.Tensor]:
        """One-time DEC feature computation for a simulation.

        `cells` is the flat VTK-style (connectivity, offsets) pair from the
        reader, with node indices matching `positions` (i.e. already filtered
        and remapped if a part filter is active).
        """
        if cells is None:
            raise ValueError(
                "DECEncoder.precompute needs cells=(connectivity, offsets); "
                "load the simulation with with_cells=True."
            )
        coords = positions[0].detach().cpu().numpy().astype(np.float64)
        conn, offsets = cells
        f = dec_point_features_flat(coords, conn, offsets)

        bnd = f["dec_is_boundary"] > 0.5
        curv_valid = ~bnd
        # normals are unit on the shell surface, exactly zero elsewhere
        on_surface = np.linalg.norm(f["dec_normal"], axis=1) > 0.5

        feats = np.concatenate(
            [
                _zscore_masked(f["dec_h_signed"], curv_valid, self.clip_sigma)[:, None],
                _zscore_masked(f["dec_h_mag"], curv_valid, self.clip_sigma)[:, None],
                f["dec_normal"],
                _zscore_masked(f["dec_log_area"], on_surface, self.clip_sigma)[:, None],
                f["dec_is_boundary"][:, None],
            ],
            axis=1,
        ).astype(np.float32)                            # (N, 7)

        return [torch.from_numpy(feats).unsqueeze(0).to(positions.device)]

    def forward(
        self,
        positions: torch.Tensor,                       # (B, N, 3)
        part_id: torch.Tensor | None = None,           # unused (interface parity)
        precomputed_stats: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:                                 # (B, N, hidden_dim)
        if precomputed_stats is None:
            raise ValueError(
                "DECEncoder requires precomputed_stats: mesh connectivity is "
                "not available at forward time. Call precompute() per "
                "simulation and pass its result through."
            )
        feats = precomputed_stats[0]
        B = positions.shape[0]
        if feats.shape[0] != B:
            feats = feats.expand(B, -1, -1)
        return self.mlp(feats)
