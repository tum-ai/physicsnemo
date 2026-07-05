"""
EnhancedGeoTransolver — GeoTransolver with an additional per-node
geometric context stream from GeometricEncoder.

How it works
------------
GeometricEncoder runs ball queries on the node positions and part IDs to
produce a per-node context vector (B, N, encoder_hidden_dim).  This vector
is concatenated onto the existing geometry tensor before it enters
GeoTransolver's ContextProjector, so the slice tokens that are cross-attended
in every GALE block carry both the original geometry features *and* our
richer statistical / part-aware descriptors — without touching any existing
file.

                                              ┌─ GeometricEncoder ──────────┐
positions (B,N,3) ──────────────────────────►│ ball-query stats + part emb │
part_id   (N,)   ──────────────────────────►│ → (B, N, encoder_hidden_dim)│
                                              └─────────────────────────────┘
                                                            │ cat
geometry  (B,N,C_geo) ──────────────────────────────────────┘
                                              → enhanced_geometry (B, N, C_geo + encoder_hidden_dim)
                                                            │
                                              ┌─ GeoTransolver ─────────────┐
local_embedding (B,N,C) ─────────────────►   │  ContextProjector           │
enhanced_geometry (B,N,C_geo+enc_dim) ───►   │  → slice tokens (B,H,S,D)  │
global_embedding (B,N_g,C_g) ────────────►   │  → GALE × n_layers         │
                                              │  → output (B, N, out_dim)   │
                                              └─────────────────────────────┘

The forward signature is identical to GeoTransolver.forward() plus one
optional keyword argument: `part_id (N,) long`.  Existing call-sites that
don't pass part_id get a fallback of all-zeros (one virtual "unknown" part).

Usage::

    model = EnhancedGeoTransolver(
        functional_dim=3,
        out_dim=500,
        geometry_dim=3,
        global_dim=3,
        n_layers=6,
        slice_num=128,
        use_te=False,
        encoder_radii=[50.0, 200.0, 600.0],
        encoder_max_neighbors=[8, 32, 64],
        encoder_n_parts=300,
        encoder_hidden_dim=32,
    )

    output = model(
        local_embedding=fx,          # (B, N, functional_dim)
        local_positions=coords,      # (B, N, 3)
        geometry=coords,             # (B, N, geometry_dim)
        global_embedding=global_emb, # (B, 1, global_dim)  or None
        part_id=part_id,             # (N,)                or None
    )
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ── local modules (same directory) ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from geometric_encoder import GeometricEncoder

# ── GeoTransolver from the existing physicsnemo codebase ─────────────────────
from physicsnemo.experimental.models.geotransolver import GeoTransolver


class EnhancedGeoTransolver(nn.Module):
    """
    GeoTransolver augmented with per-node geometric context.

    All GeoTransolver constructor arguments are accepted unchanged.
    Three groups of new arguments (prefixed ``encoder_``) control the
    GeometricEncoder that produces the extra context stream.

    Parameters — GeoTransolver (pass-through)
    -----------------------------------------
    functional_dim : int
        Input feature dimension (e.g. 3 for XYZ coords only).
    out_dim : int
        Output dimension.
    geometry_dim : int | None
        Dimension of the raw geometry tensor passed at forward time.
        The actual ``geometry_dim`` seen by GeoTransolver will be
        ``geometry_dim + encoder_hidden_dim`` (or ``encoder_hidden_dim``
        when geometry_dim is None).
    global_dim : int | None
        Dimension of the global embedding.
    **gt_kwargs
        Any remaining GeoTransolver constructor arguments
        (n_layers, n_hidden, slice_num, use_te, …).

    Parameters — GeometricEncoder (new)
    ------------------------------------
    encoder_radii : list[float]
        Ball-query radii.  Units must match the positions passed at forward
        time.  Default: [0.1, 0.25, 1.0].
    encoder_max_neighbors : int | list[int]
        Expected max neighbour count per radius (used for density
        normalization).  Default: 32.
    encoder_n_parts : int
        Size of the part-ID embedding table.  Must be > the largest
        part_id value you will pass.  Default: 1200.
    encoder_part_embed_dim : int
        Part-ID embedding dimension.  Default: 8.
    encoder_hidden_dim : int
        Output dim of GeometricEncoder; this many channels are appended
        to the geometry tensor.  Default: 32.

    Forward inputs
    --------------
    local_embedding : (B, N, functional_dim)
    local_positions : (B, N, 3) | None
        Node XYZ coordinates used for GeoTransolver's internal ball query
        (when include_local_features=True).
    global_embedding : (B, N_g, global_dim) | None
    geometry : (B, N, geometry_dim) | None
        Raw geometry features.  The encoder output is *concatenated* here
        before passing to GeoTransolver.
    part_id : (N,) long | None
        Integer part IDs per node.  Falls back to all-zeros when None.
    time : not yet implemented (passed through to GeoTransolver).
    return_embedding_states : bool
        When True, also return the (B, H, S, D) embedding states.

    Forward output
    --------------
    (B, N, out_dim)  or  ((B, N, out_dim), embedding_states)
    """

    def __init__(
        self,
        # ── GeoTransolver args (required) ────────────────────────────────
        functional_dim: int,
        out_dim: int,
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        # ── GeometricEncoder args (new) ──────────────────────────────────
        encoder_radii: list[float] | None = None,
        encoder_max_neighbors: list[int] | int = 32,
        encoder_n_parts: int = 1200,
        encoder_part_embed_dim: int = 8,
        encoder_hidden_dim: int = 32,
        # ── remaining GeoTransolver kwargs ───────────────────────────────
        **gt_kwargs,
    ) -> None:
        super().__init__()

        self.encoder_hidden_dim = encoder_hidden_dim
        self._geometry_dim_raw = geometry_dim   # original dim, before augmentation

        # ── geometric encoder ─────────────────────────────────────────────
        self.geo_encoder = GeometricEncoder(
            radii=encoder_radii,
            max_neighbors=encoder_max_neighbors,
            n_parts=encoder_n_parts,
            part_embed_dim=encoder_part_embed_dim,
            hidden_dim=encoder_hidden_dim,
        )

        # ── compute augmented geometry_dim seen by GeoTransolver ──────────
        if geometry_dim is not None:
            enhanced_geometry_dim = geometry_dim + encoder_hidden_dim
        else:
            # no raw geometry tensor provided at runtime — encoder output only
            enhanced_geometry_dim = encoder_hidden_dim

        # ── GeoTransolver (unchanged, just with a wider geometry_dim) ─────
        self.backbone = GeoTransolver(
            functional_dim=functional_dim,
            out_dim=out_dim,
            geometry_dim=enhanced_geometry_dim,
            global_dim=global_dim,
            **gt_kwargs,
        )

    def forward(
        self,
        local_embedding: torch.Tensor,              # (B, N, functional_dim)
        local_positions: torch.Tensor | None = None,   # (B, N, 3)
        global_embedding: torch.Tensor | None = None,  # (B, N_g, global_dim)
        geometry: torch.Tensor | None = None,           # (B, N, geometry_dim)
        part_id: torch.Tensor | None = None,            # (N,)  long
        time: torch.Tensor | None = None,
        return_embedding_states: bool = False,
    ) -> torch.Tensor:

        # ── resolve positions for the geometric encoder ───────────────────
        # prefer local_positions; fall back to geometry (which is often XYZ)
        if local_positions is not None:
            pos = local_positions
        elif geometry is not None:
            pos = geometry[..., :3]
        else:
            raise ValueError(
                "EnhancedGeoTransolver needs either local_positions or geometry "
                "to compute ball-query statistics."
            )

        B = pos.shape[0]

        # ── part_id fallback ──────────────────────────────────────────────
        if part_id is None:
            # all nodes treated as the same "unknown" part
            part_id = torch.zeros(pos.shape[1], dtype=torch.long, device=pos.device)

        # ── run geometric encoder ─────────────────────────────────────────
        geo_context = self.geo_encoder(pos, part_id)   # (B, N, encoder_hidden_dim)

        # ── augment geometry tensor ───────────────────────────────────────
        if geometry is not None:
            enhanced_geometry = torch.cat([geometry, geo_context], dim=-1)
        else:
            enhanced_geometry = geo_context             # encoder output only

        # ── call GeoTransolver with the augmented geometry ────────────────
        return self.backbone(
            local_embedding=local_embedding,
            local_positions=local_positions,
            global_embedding=global_embedding,
            geometry=enhanced_geometry,
            time=time,
            return_embedding_states=return_embedding_states,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    B, N = 1, 512
    functional_dim = 3
    geometry_dim   = 3
    global_dim     = 3
    out_dim        = 10
    n_parts        = 50

    positions = torch.randn(B, N, 3)
    geometry  = positions.clone()
    features  = positions.clone()
    global_emb = torch.randn(B, 1, global_dim)
    part_id    = torch.randint(0, n_parts, (N,))

    model = EnhancedGeoTransolver(
        functional_dim=functional_dim,
        out_dim=out_dim,
        geometry_dim=geometry_dim,
        global_dim=global_dim,
        n_layers=2,
        n_hidden=64,
        n_head=4,
        slice_num=16,
        use_te=False,
        include_local_features=False,
        encoder_radii=[0.5, 1.5, 3.0],
        encoder_max_neighbors=16,
        encoder_n_parts=n_parts,
        encoder_hidden_dim=16,
    )
    model.eval()

    print("EnhancedGeoTransolver")
    print(f"  backbone geometry_dim : {model.backbone.context_builder.geometry_tokenizer.in_project_x.in_features}")

    with torch.no_grad():
        output = model(
            local_embedding=features,
            local_positions=positions,
            geometry=geometry,
            global_embedding=global_emb,
            part_id=part_id,
        )

    print(f"  input  shape : {tuple(features.shape)}")
    print(f"  output shape : {tuple(output.shape)}")
    assert output.shape == (B, N, out_dim)
    assert not output.isnan().any()
    print("OK — correct shape, no NaNs.")

    # verify fallback: no part_id passed
    with torch.no_grad():
        output2 = model(
            local_embedding=features,
            local_positions=positions,
            geometry=geometry,
            global_embedding=global_emb,
            part_id=None,
        )
    assert output2.shape == (B, N, out_dim)
    print("OK — part_id=None fallback works.")
