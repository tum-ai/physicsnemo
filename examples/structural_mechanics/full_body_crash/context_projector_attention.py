# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Context Projector for GeoTransolver model (duplicated + attention variant).

This is a duplicate of
``physicsnemo/experimental/models/geotransolver/context_projector.py``,
kept local to ``examples/structural_mechanics/crash`` so the attention-based
ball-query variant can be tried out without touching the shared core library.

Everything below is identical to the original file EXCEPT for the new
``AttentionGeometricFeatureProcessor`` class, added directly after
``GeometricFeatureProcessor``. It has the exact same constructor/forward
signature as ``GeometricFeatureProcessor`` (drop-in replacement), so it can
be swapped in wherever a ``GeometricFeatureProcessor`` instance is built
(e.g. inside ``MultiScaleFeatureExtractor``).

Classes
-------
ContextProjector
    Projects context features onto physical state slices.
StructuredContextProjector
    Context projector with Conv2d/Conv3d geometry encoding on structured grids.
GeometricFeatureProcessor
    Original: ball-query neighbors -> concatenate -> MLP.
AttentionGeometricFeatureProcessor
    NEW: ball-query neighbors -> attention-weighted aggregation.
MultiScaleFeatureExtractor
    Multi-scale geometric feature extraction with minimal complexity.
GlobalContextBuilder
    Orchestrates all context construction for the GeoTransolver model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core.version_check import check_version_spec
from physicsnemo.nn import BQWarp
from physicsnemo.nn import Mlp
from physicsnemo.nn.module.physics_attention import (
    _compute_slices_from_projections,
    _project_input,
)

from physicsnemo.nn import ConcreteDropout

# Check optional dependency availability
TE_AVAILABLE = check_version_spec("transformer_engine", "0.1.0", hard_fail=False)
if TE_AVAILABLE:
    import transformer_engine.pytorch as te


def _structured_grid_to_conv_input(
    x: Float[torch.Tensor, "batch tokens channels"],
    batch: int,
    tokens: int,
    channels: int,
    ndim: int,
    spatial_shape: tuple[int, ...],
) -> Float[torch.Tensor, "batch channels ..."]:
    r"""Reshape flat token tensor to spatial layout for Conv2d/Conv3d.

    Converts :math:`(B, N, C)` to :math:`(B, C, H, W)` for 2D or
    :math:`(B, C, H, W, D)` for 3D so that structured context projectors
    can apply spatial convolutions. Validates that :math:`N` matches the
    grid size.
    """
    if ndim == 2:
        H, W = spatial_shape
        if tokens != H * W:
            raise ValueError(
                f"Expected N={H * W} tokens for 2D grid, got N={tokens}"
            )
        return x.view(batch, H, W, channels).permute(0, 3, 1, 2)
    H, W, D = spatial_shape
    if tokens != H * W * D:
        raise ValueError(
            f"Expected N={H * W * D} tokens for 3D grid, got N={tokens}"
        )
    return x.view(batch, H, W, D, channels).permute(0, 4, 1, 2, 3)


class _SliceToContextMixin:
    r"""Internal mixin providing shared slice-to-context init and slice aggregation."""

    def _init_slice_components(
        self,
        dim_head: int,
        slice_num: int,
        heads: int,
        use_te: bool,
        plus: bool,
    ) -> None:
        linear_layer = te.Linear if (use_te and TE_AVAILABLE) else nn.Linear
        self.in_project_slice = linear_layer(dim_head, slice_num)
        self.temperature = nn.Parameter(torch.ones([1, 1, heads, 1]) * 0.5)
        if plus:
            self.proj_temperature = nn.Sequential(
                linear_layer(dim_head, slice_num),
                nn.GELU(),
                linear_layer(slice_num, 1),
                nn.GELU(),
            )

    def _compute_slices(
        self,
        slice_projections: Float[torch.Tensor, "batch tokens heads slices"],
        fx: Float[torch.Tensor, "batch tokens heads dim"],
    ) -> tuple[
        Float[torch.Tensor, "batch tokens heads slices"],
        Float[torch.Tensor, "batch heads slices dim"],
    ]:
        proj_temp = getattr(self, "proj_temperature", None) if self.plus else None
        return _compute_slices_from_projections(
            slice_projections,
            fx,
            self.temperature,
            self.plus,
            proj_temperature=proj_temp,
        )


class ContextProjector(_SliceToContextMixin, nn.Module):
    r"""Projects context features onto physical state space. (Unchanged from core.)"""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.plus = plus
        self.scale = dim_head**-0.5
        self.use_te = use_te

        linear_layer = te.Linear if (use_te and TE_AVAILABLE) else nn.Linear

        self.in_project_x = linear_layer(dim, inner_dim)
        if not plus:
            self.in_project_fx = linear_layer(dim, inner_dim)

        self.softmax = nn.Softmax(dim=-1)

        self._init_slice_components(dim_head, slice_num, heads, use_te, plus)

        if concrete_dropout:
            self.output_dropout = ConcreteDropout(
                in_features=dim_head,
                init_p=max(dropout, 0.05),
            )
        else:
            self.output_dropout = None

    def project_input_onto_slices(
        self, x: Float[torch.Tensor, "batch tokens channels"]
    ) -> (
        Float[torch.Tensor, "batch tokens heads dim"]
        | tuple[
            Float[torch.Tensor, "batch tokens heads dim"],
            Float[torch.Tensor, "batch tokens heads dim"],
        ]
    ):
        fx = None if self.plus else self.in_project_fx
        return _project_input(
            x, self.in_project_x, self.heads, self.dim_head,
            "B N (H D) -> B N H D", project_fx=fx,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch tokens channels"]
    ) -> Float[torch.Tensor, "batch heads slices dim"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3D input tensor (B, N, C), "
                    f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
                )

        if self.plus:
            projected_x = self.project_input_onto_slices(x)
            feature_projection = projected_x
        else:
            projected_x, feature_projection = self.project_input_onto_slices(x)

        slice_projections = self.in_project_slice(projected_x)

        _, slice_tokens = self._compute_slices(
            slice_projections, feature_projection
        )

        if self.output_dropout is not None:
            slice_tokens = self.output_dropout(slice_tokens)

        return slice_tokens


class StructuredContextProjector(_SliceToContextMixin, nn.Module):
    r"""Context projector with Conv2d/Conv3d geometry encoding on structured grids.
    (Unchanged from core.)"""

    def __init__(
        self,
        dim: int,
        spatial_shape: tuple[int, ...],
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        kernel: int = 3,
        use_te: bool = True,
        plus: bool = False,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()
        if len(spatial_shape) not in (2, 3):
            raise ValueError(
                f"StructuredContextProjector expects spatial_shape of length 2 or 3, got {spatial_shape!r}"
            )
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.plus = plus
        self.use_te = use_te
        self.spatial_shape = tuple(int(s) for s in spatial_shape)
        self._nd = len(self.spatial_shape)
        pad = kernel // 2
        if self._nd == 2:
            H, W = self.spatial_shape
            self.H, self.W = H, W
            self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, pad)
            if not plus:
                self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, pad)
        else:
            H, W, D_ = self.spatial_shape
            self.H, self.W, self.D = H, W, D_
            self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, pad)
            if not plus:
                self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, pad)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self._init_slice_components(dim_head, slice_num, heads, use_te, plus)

        if concrete_dropout:
            self.output_dropout = ConcreteDropout(
                in_features=dim_head,
                init_p=max(dropout, 0.05),
            )
        else:
            self.output_dropout = None

    def _grid_project(
        self, x: Float[torch.Tensor, "batch tokens channels"]
    ) -> (
        Float[torch.Tensor, "batch tokens heads dim"]
        | tuple[
            Float[torch.Tensor, "batch tokens heads dim"],
            Float[torch.Tensor, "batch tokens heads dim"],
        ]
    ):
        B, N, C = x.shape
        grid = _structured_grid_to_conv_input(
            x, B, N, C, self._nd, self.spatial_shape
        )
        pattern = (
            "B (H D) h w -> B (h w) H D"
            if self._nd == 2
            else "B (H D) h w d -> B (h w d) H D"
        )
        fx = None if self.plus else self.in_project_fx
        return _project_input(
            grid, self.in_project_x, self.heads, self.dim_head,
            pattern, project_fx=fx,
        )

    def forward(
        self, x: Float[torch.Tensor, "batch tokens channels"]
    ) -> Float[torch.Tensor, "batch heads slices dim"]:
        if not torch.compiler.is_compiling():
            if x.ndim != 3:
                raise ValueError(
                    f"Expected 3D input (B, N, C), got {x.ndim}D shape {tuple(x.shape)}"
                )
        if self.plus:
            projected_x = self._grid_project(x)
            feature_projection = projected_x
        else:
            projected_x, feature_projection = self._grid_project(x)
        slice_projections = self.in_project_slice(projected_x)
        _, slice_tokens = self._compute_slices(
            slice_projections, feature_projection
        )

        if self.output_dropout is not None:
            slice_tokens = self.output_dropout(slice_tokens)

        return slice_tokens


class GeometricFeatureProcessor(nn.Module):
    r"""Processes geometric features at a single spatial scale using BQWarp.

    ORIGINAL VERSION (unchanged from core physicsnemo). Ball-query neighbors,
    then flatten + MLP. Kept here for reference/comparison — see
    ``AttentionGeometricFeatureProcessor`` below for the attention variant.

    Forward
    -------
    query_points : torch.Tensor
        Query coordinates of shape :math:`(B, N, 3)`.
    key_features : torch.Tensor
        Features to query from of shape :math:`(B, N, C)`.

    Outputs
    -------
    torch.Tensor
        Processed features of shape :math:`(B, N, D)` where :math:`D` is ``hidden_dim``.
    """

    def __init__(
        self,
        radius: float,
        neighbors_in_radius: int,
        feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        # Ball query for neighbor search within radius
        self.bq_warp = BQWarp(radius=radius, neighbors_in_radius=neighbors_in_radius)

        # MLP to process flattened neighbor features
        self.mlp = Mlp(
            in_features=feature_dim * neighbors_in_radius,
            hidden_features=[hidden_dim, hidden_dim // 2],
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(
        self,
        query_points: Float[torch.Tensor, "batch points spatial_dim"],
        key_features: Float[torch.Tensor, "batch points features"],
    ) -> Float[torch.Tensor, "batch points hidden_dim"]:
        if not torch.compiler.is_compiling():
            if query_points.ndim != 3:
                raise ValueError(
                    f"Expected 3D query_points tensor (B, N, 3), "
                    f"got {query_points.ndim}D tensor with shape {tuple(query_points.shape)}"
                )
            if key_features.ndim != 3:
                raise ValueError(
                    f"Expected 3D key_features tensor (B, N, C), "
                    f"got {key_features.ndim}D tensor with shape {tuple(key_features.shape)}"
                )

        # Query neighbors within radius: (B, N, K, C)
        _, neighbors = self.bq_warp(query_points, key_features)

        # Flatten neighbor features for MLP: (B, N, K, C) -> (B, N, K*C)
        neighbors_flat = rearrange(neighbors, "b n k c -> b n (k c)")

        # Process through MLP with tanh activation for bounded output
        return torch.nn.functional.tanh(self.mlp(neighbors_flat))


class AttentionGeometricFeatureProcessor(nn.Module):
    r"""Attention-based variant of GeometricFeatureProcessor.

    Same constructor/forward signature as ``GeometricFeatureProcessor`` (drop-in
    replacement) and the exact same ``BQWarp`` neighbor search. The only change:
    instead of flattening all K neighbors into one vector and pushing it through
    an MLP (every neighbor counted equally), a query derived from the node's own
    position attends over its neighbors' relative positions + distances, so
    nearby/important neighbors are weighted more than others.

    Forward
    -------
    query_points : torch.Tensor
        Query coordinates of shape :math:`(B, N, 3)`.
    key_features : torch.Tensor
        Features to query from of shape :math:`(B, N, C)`. Note ``BQWarp``
        requires the last dim to be 3, so in practice C == feature_dim == 3
        (raw coordinates), same as the original class.

    Outputs
    -------
    torch.Tensor
        Processed features of shape :math:`(B, N, D)` where :math:`D` is ``hidden_dim``.
    """

    def __init__(
        self,
        radius: float,
        neighbors_in_radius: int,
        feature_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
    ) -> None:
        super().__init__()

        # Ball query for neighbor search within radius -- identical to GeometricFeatureProcessor
        self.bq_warp = BQWarp(radius=radius, neighbors_in_radius=neighbors_in_radius)

        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Query: derived from the node's own position (no separate identity
        # signal like part_id is available at this level of GeoTransolver).
        self.query_proj = nn.Linear(feature_dim, hidden_dim)
        # Key/Value: derived from each neighbor's relative position + distance,
        # not its raw absolute coordinate.
        self.kv_proj = nn.Linear(feature_dim + 1, hidden_dim * 2)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        query_points: Float[torch.Tensor, "batch points spatial_dim"],
        key_features: Float[torch.Tensor, "batch points features"],
    ) -> Float[torch.Tensor, "batch points hidden_dim"]:
        if not torch.compiler.is_compiling():
            if query_points.ndim != 3:
                raise ValueError(
                    f"Expected 3D query_points tensor (B, N, 3), "
                    f"got {query_points.ndim}D tensor with shape {tuple(query_points.shape)}"
                )
            if key_features.ndim != 3:
                raise ValueError(
                    f"Expected 3D key_features tensor (B, N, C), "
                    f"got {key_features.ndim}D tensor with shape {tuple(key_features.shape)}"
                )

        # Query neighbors within radius: (B, N, K, C) -- unchanged
        _, neighbors = self.bq_warp(query_points, key_features)

        B, N, K, _ = neighbors.shape
        H, D = self.n_heads, self.head_dim

        # Relative position + distance instead of raw absolute neighbor coordinates
        rel_pos = neighbors - query_points.unsqueeze(2)              # (B, N, K, C)
        dist = rel_pos.norm(dim=-1, keepdim=True)                    # (B, N, K, 1)
        kv_input = torch.cat([rel_pos, dist], dim=-1)                # (B, N, K, C+1)

        # Query from the node's own position
        q = self.query_proj(query_points).view(B, N, H, D)          # (B, N, H, D)

        kv = self.kv_proj(kv_input).view(B, N, K, H, 2 * D)         # (B, N, K, H, 2D)
        k, v = kv.chunk(2, dim=-1)                                    # each (B, N, K, H, D)

        # Importance score per neighbor (replaces "concatenate + MLP")
        attn_logits = torch.einsum("bnhd,bnkhd->bnhk", q, k) * (D ** -0.5)  # (B, N, H, K)
        attn_weights = torch.softmax(attn_logits, dim=-1)                   # (B, N, H, K)

        # Weighted sum over neighbors (replaces flattening all neighbors together)
        attended = torch.einsum("bnhk,bnkhd->bnhd", attn_weights, v)        # (B, N, H, D)
        attended = attended.reshape(B, N, H * D)                           # (B, N, hidden_dim)

        return torch.nn.functional.tanh(self.out_proj(attended))


class MultiScaleFeatureExtractor(nn.Module):
    r"""Multi-scale geometric feature extraction with minimal complexity.

    Manages multiple GeometricFeatureProcessor instances for different radii.
    Provides both tokenized context and concatenated local features.

    NOTE: uses ``GeometricFeatureProcessor`` (the original) by default, exactly
    like core physicsnemo. To try the attention variant, swap the class used
    in ``self.processors`` below for ``AttentionGeometricFeatureProcessor``.
    """

    def __init__(
        self,
        geometry_dim: int,
        radii: list[float],
        neighbors_in_radius: list[int],
        hidden_dim: int,
        n_head: int,
        dim_head: int,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.num_scales = len(radii)

        # One processor per scale for geometric feature extraction
        self.processors = nn.ModuleList(
            [
                GeometricFeatureProcessor(
                    radii[i], neighbors_in_radius[i], geometry_dim, hidden_dim
                )
                for i in range(self.num_scales)
            ]
        )

        # One tokenizer per scale for projecting to context space
        self.tokenizers = nn.ModuleList(
            [
                ContextProjector(
                    hidden_dim,
                    n_head,
                    dim_head,
                    dropout,
                    slice_num,
                    use_te,
                    plus,
                    concrete_dropout=concrete_dropout,
                )
                for _ in range(self.num_scales)
            ]
        )

    def extract_context_features(
        self,
        spatial_coords: Float[torch.Tensor, "batch points spatial_dim"],
        geometry: Float[torch.Tensor, "batch points geometry_dim"],
    ) -> list[Float[torch.Tensor, "batch heads slices dim"]]:
        return [
            tokenizer(processor(spatial_coords, geometry))
            for processor, tokenizer in zip(self.processors, self.tokenizers)
        ]

    def extract_local_features(
        self,
        spatial_coords: Float[torch.Tensor, "batch points spatial_dim"],
        geometry: Float[torch.Tensor, "batch points geometry_dim"],
    ) -> Float[torch.Tensor, "batch points total_hidden"]:
        return torch.cat(
            [processor(geometry, spatial_coords) for processor in self.processors],
            dim=-1,
        )


class GlobalContextBuilder(nn.Module):
    r"""Orchestrates all context construction with a clean, simple interface.
    (Unchanged from core.)"""

    def __init__(
        self,
        functional_dims: tuple[int, ...],
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        radii: list[float] | None = None,
        neighbors_in_radius: list[int] | None = None,
        n_hidden_local: int = 32,
        n_hidden: int = 256,
        n_head: int = 8,
        dropout: float = 0.0,
        slice_num: int = 32,
        use_te: bool = True,
        plus: bool = False,
        include_local_features: bool = False,
        structured_shape: tuple[int, ...] | None = None,
        concrete_dropout: bool = False,
    ) -> None:
        super().__init__()

        if radii is None:
            radii = [0.05, 0.25]
        if neighbors_in_radius is None:
            neighbors_in_radius = [8, 32]

        dim_head = n_hidden // n_head
        context_dim = 0
        self.structured_shape = structured_shape

        use_local_bq = (
            geometry_dim is not None
            and include_local_features
            and structured_shape is None
        )

        if use_local_bq:
            self.local_extractors = nn.ModuleList(
                [
                    MultiScaleFeatureExtractor(
                        geometry_dim,
                        radii,
                        neighbors_in_radius,
                        n_hidden_local,
                        n_head,
                        dim_head,
                        dropout,
                        slice_num,
                        use_te,
                        plus,
                        concrete_dropout=concrete_dropout,
                    )
                    for _ in functional_dims
                ]
            )
            context_dim += dim_head * len(radii) * len(functional_dims)
        else:
            self.local_extractors = None

        if geometry_dim is not None:
            if structured_shape is not None:
                self.geometry_tokenizer = StructuredContextProjector(
                    geometry_dim,
                    structured_shape,
                    n_head,
                    dim_head,
                    dropout,
                    slice_num,
                    use_te=use_te,
                    plus=plus,
                    concrete_dropout=concrete_dropout,
                )
            else:
                self.geometry_tokenizer = ContextProjector(
                    geometry_dim, n_head, dim_head, dropout, slice_num, use_te, plus=plus,
                    concrete_dropout=concrete_dropout,
                )
            context_dim += dim_head
        else:
            self.geometry_tokenizer = None

        if global_dim is not None:
            self.global_tokenizer = ContextProjector(
                global_dim,
                n_head,
                dim_head,
                dropout,
                slice_num,
                use_te,
                plus,
                concrete_dropout=concrete_dropout,
            )
            context_dim += dim_head
        else:
            self.global_tokenizer = None

        self._context_dim = context_dim

    def get_context_dim(self) -> int:
        return self._context_dim

    def build_context(
        self,
        local_embeddings: tuple[Float[torch.Tensor, "batch tokens features"], ...],
        local_positions: (
            tuple[Float[torch.Tensor, "batch tokens spatial_dim"], ...] | None
        ),
        geometry: Float[torch.Tensor, "batch tokens geometry_dim"] | None = None,
        global_embedding: Float[torch.Tensor, "batch global_tokens global_dim"]
        | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch heads slices context_dim"] | None,
        list[Float[torch.Tensor, "batch tokens local_features"]] | None,
        Float[torch.Tensor, "batch heads slices dim_head"] | None,
    ]:
        if not torch.compiler.is_compiling():
            if len(local_embeddings) == 0:
                raise ValueError("Expected non-empty tuple of local embeddings")
            for i, emb in enumerate(local_embeddings):
                if emb.ndim != 3:
                    raise ValueError(
                        f"Expected 3D local_embedding tensor (B, N, C) at index {i}, "
                        f"got {emb.ndim}D tensor with shape {tuple(emb.shape)}"
                    )

        context_parts = []
        local_features = None
        geometry_context_detached: torch.Tensor | None = None

        if local_positions is None and self.local_extractors is not None:
            raise ValueError(
                "Local positions are required if local features are enabled."
            )

        if self.local_extractors is not None and geometry is not None:
            local_features = []
            for i, embedding in enumerate(local_embeddings):
                spatial_coords = local_positions[i]

                context_feats = self.local_extractors[i].extract_context_features(
                    spatial_coords, geometry
                )
                context_parts.extend(context_feats)

                local_feats = self.local_extractors[i].extract_local_features(
                    spatial_coords, geometry
                )
                local_features.append(local_feats)

        if self.geometry_tokenizer is not None and geometry is not None:
            geometry_context = self.geometry_tokenizer(geometry)
            geometry_context_detached = geometry_context.detach()
            context_parts.append(geometry_context)

        if self.global_tokenizer is not None and global_embedding is not None:
            context_parts.append(self.global_tokenizer(global_embedding))

        context = torch.cat(context_parts, dim=-1) if context_parts else None

        return context, local_features, geometry_context_detached
