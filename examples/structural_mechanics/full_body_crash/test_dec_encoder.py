"""Smoke test: DECEncoder standalone + plugged into EnhancedGeoTransolver.

Builds a synthetic sphere shell mesh (quads, flat conn/offsets like the
reader), precomputes DEC features, and runs a full forward pass.

    python3 test_dec_encoder.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from geo_encoders import DECEncoder
from geo_encoders.dec_features import _sphere_cells, _to_flat, run_tests
from geo_transolver_enhanced import EnhancedGeoTransolver


def main() -> int:
    torch.manual_seed(0)

    print("── DEC feature math self-tests ─────────────────────────")
    if not run_tests():
        return 1

    print("\n── DECEncoder precompute + forward ─────────────────────")
    X, cells = _sphere_cells(r=2.0)
    conn, offsets = _to_flat(cells)
    N = len(X)
    positions = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # (1, N, 3)
    part_id = torch.zeros(N, dtype=torch.long)

    encoder = DECEncoder(hidden_dim=16)
    stats = encoder.precompute(positions, part_id, cells=(conn, offsets))
    feats = stats[0]
    assert feats.shape == (1, N, 7), feats.shape
    assert not feats.isnan().any()
    # normalized channels are bounded; boundary flag is binary
    assert feats[..., 0].abs().max() <= encoder.clip_sigma + 1e-5
    assert set(feats[..., 6].unique().tolist()) <= {0.0, 1.0}
    print(f"  precomputed feats : {tuple(feats.shape)}, "
          f"boundary frac = {feats[..., 6].mean():.3f}")

    out = encoder(positions, part_id, precomputed_stats=stats)
    assert out.shape == (1, N, 16), out.shape
    assert not out.isnan().any()
    print(f"  encoder output    : {tuple(out.shape)}  OK")

    print("\n── EnhancedGeoTransolver end-to-end ────────────────────")
    model = EnhancedGeoTransolver(
        functional_dim=3,
        out_dim=48,
        geometry_dim=3,
        global_dim=None,
        geo_encoder=encoder,
        n_layers=2,
        n_hidden=64,
        n_head=4,
        slice_num=16,
        use_te=False,
        include_local_features=False,
    )
    model.eval()
    with torch.no_grad():
        pred = model(
            local_embedding=positions,
            geometry=positions,
            part_id=part_id,
            precomputed_encoder_stats=stats,
        )
    assert pred.shape == (1, N, 48), pred.shape
    assert not pred.isnan().any()
    print(f"  prediction        : {tuple(pred.shape)}  OK")

    # one optimizer step to confirm gradients flow through the DEC MLP
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pred = model(
        local_embedding=positions,
        geometry=positions,
        part_id=part_id,
        precomputed_encoder_stats=stats,
    )
    loss = torch.nn.functional.mse_loss(pred, torch.zeros_like(pred))
    loss.backward()
    grad = encoder.mlp[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0
    opt.step()
    print(f"  train step        : loss={loss.item():.4f}, DEC-MLP grads flow  OK")

    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
