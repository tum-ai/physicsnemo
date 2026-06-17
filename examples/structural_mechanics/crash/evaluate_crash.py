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

"""Direct (no-VTP) evaluation of a crash prediction -- numpy + PIL only.

Reads the per-run ``.npz`` written by ``inference.py`` (keys: pred_pos, exact_pos,
pred_field_*, exact_field_*; positions in mm, fields in normalized units) and
produces:

  * a relative-L2 position-error curve vs timestep (PNG + CSV), and
  * a ground-truth vs predicted vs error GIF (top view, colored by von Mises and
    by per-node position error).

Position error is reported two ways per timestep:
    abs_mm      mean over nodes of ||pred - exact||                 (millimetres)
    rel_l2_pos  ||pred - exact||_F / ||exact - exact[t=0]||_F       (displacement-relative)

Run:
    python evaluate_crash.py outputs/predicted_vtps/eval_arrays/sim_00001.npz
    python evaluate_crash.py <npz> --stride 1 --fps 15 --out-prefix results/sim_00001
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visualize_crash as vz  # noqa: E402  (reuse colorize / render_panel / compose_frame)
from PIL import Image, ImageDraw  # noqa: E402


def rel_l2_curves(pred_pos, exact_pos):
    """Per-timestep position errors. Returns dict of [T] arrays."""
    err = pred_pos - exact_pos                       # [T, N, 3]
    node_err = np.linalg.norm(err, axis=2)           # [T, N]
    num = np.sqrt((node_err ** 2).sum(axis=1))       # [T]
    disp = exact_pos - exact_pos[0:1]                # motion relative to frame 0
    den = np.sqrt((np.linalg.norm(disp, axis=2) ** 2).sum(axis=1))  # [T]
    rel = num / np.maximum(den, 1e-8)
    rel[0] = 0.0  # t=0 is the reference frame (zero displacement) -> ratio undefined
    return {
        "abs_mm": node_err.mean(axis=1),
        "rel_l2_pos": rel,
        "node_err": node_err,                        # [T, N], for the error panel
    }


def field_rel_l2(pred, exact):
    """Per-timestep relative L2 of a scalar field, [T, N(,1)] -> [T]."""
    p = pred.reshape(pred.shape[0], -1)
    e = exact.reshape(exact.shape[0], -1)
    num = np.sqrt(((p - e) ** 2).sum(axis=1))
    den = np.sqrt((e ** 2).sum(axis=1))
    out = num / np.maximum(den, 1e-8)
    out[0] = 0.0  # initial frame reference ~ 0 -> relative error undefined
    return out


def line_plot(series, T, out_path, title):
    """Minimal multi-series line plot (no matplotlib). series: list of (label, y, rgb)."""
    W, H = 900, 360
    ml, mr, mt, mb = 70, 20, 34, 40
    img = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    x0, x1 = ml, W - mr
    y0, y1 = H - mb, mt
    # robust y-max: fields that start at ~0 produce huge early relative-L2 spikes;
    # scale to the 95th percentile and clip so the plot stays readable.
    allvals = np.concatenate([np.asarray(y, float) for _, y, _ in series])
    allvals = allvals[np.isfinite(allvals)]
    ymax = float(np.percentile(allvals, 95)) or 1.0
    d.text((ml, 8), title, fill=(20, 20, 20))
    # axes
    d.rectangle([x0, y1, x1, y0], outline=(120, 120, 120))
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = y0 + (y1 - y0) * frac
        d.line([x0, gy, x1, gy], fill=(225, 225, 225))
        d.text((6, gy - 6), f"{ymax * frac:.3g}", fill=(80, 80, 80))
    d.text((x0, y0 + 6), "0", fill=(80, 80, 80))
    d.text((x1 - 24, y0 + 6), f"{T - 1}", fill=(80, 80, 80))
    d.text(((x0 + x1) // 2 - 30, H - 14), "timestep", fill=(80, 80, 80))

    def px(t):
        return x0 + (x1 - x0) * (t / max(1, T - 1))

    def py(v):
        return y0 + (y1 - y0) * (min(v, ymax) / ymax)  # clip spikes to the box

    for li, (label, y, rgb) in enumerate(series):
        pts = [(px(t), py(y[t])) for t in range(len(y))]
        d.line(pts, fill=rgb, width=2)
        d.text((x1 - 150, mt + 4 + li * 14), label, fill=rgb)
    img.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("npz", help="per-run npz from inference.py (eval_arrays/<run>.npz)")
    ap.add_argument("--stride", type=int, default=2, help="render every Nth timestep in the GIF")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out-prefix", default=None, help="output path prefix (default: ./eval_<run>)")
    args = ap.parse_args()

    d = np.load(args.npz)
    run = os.path.splitext(os.path.basename(args.npz))[0]
    prefix = args.out_prefix or os.path.join(os.getcwd(), f"eval_{run}")
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    pred_pos = d["pred_pos"]                          # [T, N, 3] mm
    exact_pos = d["exact_pos"]
    T, N, _ = pred_pos.shape
    cur = rel_l2_curves(pred_pos, exact_pos)

    # field curves (normalized units)
    field_curves = {}
    for key in ("stress_vm", "effective_plastic_strain"):
        pk, ek = f"pred_field_{key}", f"exact_field_{key}"
        if pk in d.files and ek in d.files:
            field_curves[key] = field_rel_l2(d[pk], d[ek])

    # ---- CSV ----
    csv_path = f"{prefix}_error.csv"
    cols = ["timestep", "abs_mm", "rel_l2_pos"] + [f"rel_l2_{k}" for k in field_curves]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for t in range(T):
            row = [t, f"{cur['abs_mm'][t]:.6g}", f"{cur['rel_l2_pos'][t]:.6g}"]
            row += [f"{field_curves[k][t]:.6g}" for k in field_curves]
            w.writerow(row)

    # ---- summary ----
    print(f"{run}: T={T}, N={N}")
    print(f"  position abs error (mm):  mean={cur['abs_mm'].mean():.4g}  "
          f"final={cur['abs_mm'][-1]:.4g}  max={cur['abs_mm'].max():.4g}")
    print(f"  position rel L2:          mean={np.nanmean(cur['rel_l2_pos'][1:]):.4g}  "
          f"final={cur['rel_l2_pos'][-1]:.4g}")
    for k, v in field_curves.items():
        print(f"  {k} rel L2:             mean={v.mean():.4g}  final={v[-1]:.4g}")

    # ---- line plot ----
    series = [("rel L2 position", cur["rel_l2_pos"], (200, 30, 30))]
    for k, v in field_curves.items():
        series.append((f"rel L2 {k}", v, (30, 90, 200) if "stress" in k else (30, 160, 60)))
    plot_path = line_plot(series, T, f"{prefix}_error.png", f"{run}: relative L2 error vs timestep")

    # ---- GIF: ground truth | predicted | position error ----
    p_stress = d.get("pred_field_stress_vm")
    e_stress = d.get("exact_field_stress_vm")
    have_stress = p_stress is not None and e_stress is not None
    if have_stress:
        p_stress = p_stress.reshape(T, N)
        e_stress = e_stress.reshape(T, N)
        s_vmax = float(np.percentile(e_stress, 99)) or 1.0

    # shared extents over GT + prediction so deformation is comparable
    both = np.concatenate([exact_pos, pred_pos], axis=1)
    hext, vext = vz._extents(both, 1, 0)  # y horizontal, x vertical (top view)
    frames_idx = list(range(0, T, max(1, args.stride)))
    err_vmax = float(np.percentile(cur["node_err"][frames_idx], 99)) or 1.0

    images = []
    for t in frames_idx:
        panels = []
        if have_stress:
            panels.append(vz.render_panel(exact_pos[t], e_stress[t], 1, 0, hext, vext,
                                          "inferno", 0.0, s_vmax, "ground truth | von Mises"))
            panels.append(vz.render_panel(pred_pos[t], p_stress[t], 1, 0, hext, vext,
                                          "inferno", 0.0, s_vmax, "predicted | von Mises"))
        panels.append(vz.render_panel(pred_pos[t], cur["node_err"][t], 1, 0, hext, vext,
                                      "inferno", 0.0, err_vmax, "position error (mm)"))
        head = (f"{run}   t={t:>3}/{T - 1}   "
                f"abs={cur['abs_mm'][t]:.2f} mm   relL2={cur['rel_l2_pos'][t]:.3g}")
        images.append(vz.compose_frame(panels, head))

    gif_path = f"{prefix}.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:],
                   duration=int(1000 / max(1, args.fps)), loop=0, optimize=True)

    print(f"saved:\n  {csv_path}\n  {plot_path}\n  {gif_path}  "
          f"({len(images)} frames, {images[0].size[0]}x{images[0].size[1]})")


if __name__ == "__main__":
    main()
