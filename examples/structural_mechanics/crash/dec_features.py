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

"""DEC (Discrete Exterior Calculus) shape features for crash surrogate inputs.

Computes per-node geometric-context features on the *undeformed* (t=0) shell
mesh: signed mean curvature via the cotangent Laplacian, unit normals, lumped
node area, and an open-boundary flag. These are static input features — they
are non-trivial on the reference mesh (unlike the deformation gradient, which
is identically I there), invariant/equivariant under rigid motion, and
mesh-density corrected by the barycentric mass matrix (the DEC Hodge star_0).

They enter the model through the datapipe's ``static_features`` mechanism:
``vtkhdf_reader.Reader(dec_features=True)`` injects them into each sample's
``point_data`` under the keys in ``DEC_FEATURE_KEYS``; the experiment config
lists those keys under ``datapipe.static_features`` and bumps
``model.functional_dim`` by ``DEC_NUM_CHANNELS``.

Self-test (flat plane / sphere / rigid-motion invariance):
    python3 dec_features.py --test

Origin: prototyped and validated against the CarCrashNet yaris sample in the
standalone DEC_testing lab; this is the dependency-free (numpy+scipy) port.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

EPS = 1e-12

# point_data keys produced by dec_point_features, and their channel widths.
DEC_FEATURE_CHANNELS = {
    "dec_h_signed": 1,   # signed mean curvature Hvec . n_hat  (convex vs concave)
    "dec_h_mag": 1,      # |Hvec| = 2H, zeroed on open boundaries (undefined there)
    "dec_normal": 3,     # area-weighted unit vertex normal
    "dec_log_area": 1,   # log lumped (barycentric) node area — mesh-density cue
    "dec_is_boundary": 1,  # 1.0 where curvature is undefined (open edge / no shell)
}
DEC_FEATURE_KEYS = list(DEC_FEATURE_CHANNELS)
DEC_NUM_CHANNELS = sum(DEC_FEATURE_CHANNELS.values())  # = 7


def triangulate_cells(cells) -> np.ndarray:
    """[M,3] triangle array from mixed polygonal cells (3-node kept, 4-node split).

    Cells with other sizes (beams, solids) are skipped — DEC surface operators
    are only defined on the shell submesh.
    """
    tris = []
    for cell in cells:
        m = len(cell)
        if m == 3:
            tris.append(cell)
        elif m == 4:
            a, b, c, d = cell
            tris.append((a, b, c))
            tris.append((a, c, d))
    return np.asarray(tris, dtype=np.int64) if tris else np.zeros((0, 3), np.int64)


def cotan_laplacian(X: np.ndarray, tris: np.ndarray):
    """Sparse cotangent Laplacian (PSD, L = D - W) + barycentric lumped mass."""
    n = len(X)
    i0, i1, i2 = tris[:, 0], tris[:, 1], tris[:, 2]
    e0, e1, e2 = X[i2] - X[i1], X[i0] - X[i2], X[i1] - X[i0]  # edge opposite corner
    dbl_area = np.maximum(np.linalg.norm(np.cross(e1, -e2), axis=1), EPS)
    cot0 = np.einsum("ij,ij->i", -e1, e2) / dbl_area
    cot1 = np.einsum("ij,ij->i", -e2, e0) / dbl_area
    cot2 = np.einsum("ij,ij->i", -e0, e1) / dbl_area
    # clamp for robustness to slivers/obtuse triangles (keeps L an M-matrix)
    cot0, cot1, cot2 = (np.clip(c, 0.0, 1e6) for c in (cot0, cot1, cot2))

    rows = np.concatenate([i1, i2, i2, i0, i0, i1])
    cols = np.concatenate([i2, i1, i0, i2, i1, i0])
    w = 0.5 * np.concatenate([cot0, cot0, cot1, cot1, cot2, cot2])
    W = sp.coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W

    m = np.zeros(n)
    np.add.at(m, tris.ravel(), np.repeat(dbl_area / 6.0, 3))  # area/3 per corner
    return L, np.maximum(m, EPS)


def boundary_nodes(tris: np.ndarray, n: int) -> np.ndarray:
    """Bool mask of nodes on an open boundary (edge with one incident triangle)."""
    edges = np.sort(
        np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1
    )
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    b = np.zeros(n, dtype=bool)
    b[uniq[counts == 1].ravel()] = True
    return b


def vertex_normals(X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    fn = np.cross(X[tris[:, 1]] - X[tris[:, 0]], X[tris[:, 2]] - X[tris[:, 0]])
    vn = np.zeros_like(X)
    for k in range(3):
        np.add.at(vn, tris[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), EPS)


def dec_point_features(coords0: np.ndarray, cells) -> dict:
    """Per-node DEC features on the undeformed mesh, keyed by DEC_FEATURE_KEYS.

    Args:
        coords0: [N,3] t=0 node coordinates (the merged reader mesh).
        cells:   iterable of per-cell node-id lists (mixed sizes OK; only
                 3/4-node shell cells contribute — others get zeros + flag).

    Returns:
        {key: float32 array [N] or [N,3]} — safe to place into point_data.
    """
    X = np.asarray(coords0, dtype=np.float64)
    n = len(X)
    tris = triangulate_cells(cells)
    out = {
        "dec_h_signed": np.zeros(n, np.float32),
        "dec_h_mag": np.zeros(n, np.float32),
        "dec_normal": np.zeros((n, 3), np.float32),
        "dec_log_area": np.zeros(n, np.float32),
        "dec_is_boundary": np.ones(n, np.float32),  # default: feature undefined
    }
    if len(tris) == 0:
        return out

    L, m = cotan_laplacian(X, tris)
    Hvec = (L @ X) / m[:, None]  # mean-curvature vector, |Hvec| = 2H (sphere r -> 2/r)
    normals = vertex_normals(X, tris)
    on_surface = np.zeros(n, dtype=bool)
    on_surface[np.unique(tris)] = True
    bnd = boundary_nodes(tris, n) | ~on_surface  # curvature invalid on open edges
    valid = ~bnd

    out["dec_h_signed"][valid] = np.einsum("ij,ij->i", Hvec, normals)[valid]
    out["dec_h_mag"][valid] = np.linalg.norm(Hvec, axis=1)[valid]
    out["dec_normal"][on_surface] = normals[on_surface].astype(np.float32)
    out["dec_log_area"][on_surface] = np.log(m[on_surface]).astype(np.float32)
    out["dec_is_boundary"] = bnd.astype(np.float32)
    return out


# ----------------------------------------------------------------------------
# Self-tests (flat plane, sphere, rigid-motion invariance)
# ----------------------------------------------------------------------------

def _grid_cells(nx=20, ny=20):
    gx, gy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    X = np.stack([gx.ravel(), gy.ravel(), np.zeros(nx * ny)], axis=1)
    idx = np.arange(nx * ny).reshape(ny, nx)
    quads = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(),
                      idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], axis=1)
    return X, [list(q) for q in quads]


def _sphere_cells(r=2.0, nt=40, nph=80):
    th = np.linspace(np.pi / 6, 5 * np.pi / 6, nt)
    ph = np.linspace(0, 2 * np.pi, nph, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    X = r * np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    X = X.reshape(-1, 3)
    idx = np.arange(nt * nph).reshape(nt, nph)
    a = idx[:-1, :]; b = idx[:-1, np.r_[1:nph, 0]]
    c = idx[1:, np.r_[1:nph, 0]]; d = idx[1:, :]
    quads = np.stack([a.ravel(), b.ravel(), c.ravel(), d.ravel()], axis=1)
    return X, [list(q) for q in quads]


def run_tests():
    ok = True
    X, cells = _grid_cells()
    f = dec_point_features(X, cells)
    interior = f["dec_is_boundary"] == 0
    v = float(np.abs(f["dec_h_mag"][interior]).max())
    print(f"[test] plane: max interior |H| = {v:.2e} (want ~0)",
          "PASS" if v < 1e-8 else "FAIL")
    ok &= v < 1e-8

    r = 2.0
    Xs, cs = _sphere_cells(r=r)
    fs = dec_point_features(Xs, cs)
    inter = fs["dec_is_boundary"] == 0
    med = float(np.median(fs["dec_h_mag"][inter]))
    err = abs(med - 2.0 / r) / (2.0 / r)
    print(f"[test] sphere r={r}: median |H| = {med:.4f} (want {2/r:.4f}, err {err:.1%})",
          "PASS" if err < 0.05 else "FAIL")
    ok &= err < 0.05
    # signed curvature has one sign on a sphere (normal orientation fixes which)
    sgn = fs["dec_h_signed"][inter]
    same_sign = max((sgn > 0).mean(), (sgn < 0).mean())
    print(f"[test] sphere: signed H single-signed on {same_sign:.1%} of nodes",
          "PASS" if same_sign > 0.99 else "FAIL")
    ok &= same_sign > 0.99

    ang = 0.7
    Rot = np.array([[np.cos(ang), -np.sin(ang), 0],
                    [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]])
    f2 = dec_point_features(Xs @ Rot.T + np.array([5.0, -3.0, 11.0]), cs)
    dmax = float(np.abs(f2["dec_h_mag"] - fs["dec_h_mag"]).max())
    print(f"[test] rigid motion: max |H| change = {dmax:.2e} (want ~0)",
          "PASS" if dmax < 1e-6 else "FAIL")
    ok &= dmax < 1e-6

    print("[test] ALL PASS" if ok else "[test] FAILURES PRESENT")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
