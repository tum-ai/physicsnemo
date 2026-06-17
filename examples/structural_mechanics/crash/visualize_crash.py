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

"""Animate one bumper-beam crash as a GIF -- no matplotlib required.

Loads a single simulation through ``vtkhdf_reader.load_vtkhdf_file`` (the same
tensors the datapipe feeds the model: coords [T, N, 3] plus the per-step
``stress_vm`` and ``effective_plastic_strain`` targets) and renders an animated
GIF using only numpy + PIL. Three panels per frame, with fixed world extents and
shared robust color scales so the crush and stress hotspots are comparable across
time:

    1. top view  (x-y), colored by von Mises stress
    2. side view (x-z), colored by von Mises stress   -- shows out-of-plane buckling
    3. top view  (x-y), colored by effective plastic strain

Run:
    python visualize_crash.py [sim_id] [--field ...] [--stride N] [--fps F] [--out path]

Examples:
    python visualize_crash.py                      # auto-pick a high-velocity sim
    python visualize_crash.py sim_00002
    python visualize_crash.py sim_00002 --stride 1 --fps 20 --out crash.gif
"""

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtkhdf_reader as R  # noqa: E402

CRASH_DIR = os.path.dirname(os.path.abspath(__file__))
# repo layout: .../tumai/physicsnemo/examples/structural_mechanics/crash
WORKSPACE = os.path.abspath(os.path.join(CRASH_DIR, "..", "..", "..", ".."))


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_data_dir(cli):
    return cli or os.environ.get("BUMPER_DATA_DIR") or _first_existing(
        os.path.join(WORKSPACE, "data", "simulations"),
        os.path.join(WORKSPACE, "simulations"),
    ) or os.path.join(WORKSPACE, "data", "simulations")


def resolve_master_csv(cli):
    return cli or os.environ.get("BUMPER_MASTER_CSV") or _first_existing(
        os.path.join(WORKSPACE, "data", "metadata", "bumper_beam_master_with_split.csv"),
        os.path.join(WORKSPACE, "bumper_beam_master_with_split.csv"),
    )


# ----------------------------------------------------------------------------
# Color maps (small anchor LUTs interpolated to 256 entries -- avoids matplotlib)
# ----------------------------------------------------------------------------
_INFERNO = np.array([
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
], dtype=np.float64)
_VIRIDIS = np.array([
    [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
], dtype=np.float64)


def _build_lut(anchors, n=256):
    xp = np.linspace(0.0, 1.0, len(anchors))
    xs = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(xs, xp, anchors[:, c]) for c in range(3)], axis=1)


_LUTS = {"inferno": _build_lut(_INFERNO), "viridis": _build_lut(_VIRIDIS)}


def colorize(values, vmin, vmax, lut_name):
    """Map a 1-D array of scalars to [N, 3] uint8 RGB via a 256-entry LUT."""
    lut = _LUTS[lut_name]
    if vmax <= vmin:
        vmax = vmin + 1e-9
    t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (t * (len(lut) - 1)).astype(np.int64)
    return lut[idx].astype(np.uint8)


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------
BG = (16, 16, 20)
DATA_H = 360          # pixels of plot area height per panel
TITLE_H = 22
BAR_H = 26
GAP = 10
PAD_FRAC = 0.04       # world-space padding around the data extents
MARKER_R = 1          # marker radius in pixels (0 = single pixel)


def _extents(coords, ax_h, ax_v):
    """Fixed (min,max) for two axes across all timesteps, with padding."""
    h = coords[..., ax_h]
    v = coords[..., ax_v]
    hmin, hmax = float(h.min()), float(h.max())
    vmin, vmax = float(v.min()), float(v.max())
    hpad = (hmax - hmin) * PAD_FRAC or 1.0
    vpad = (vmax - vmin) * PAD_FRAC or 1.0
    return (hmin - hpad, hmax + hpad), (vmin - vpad, vmax + vpad)


def _panel_width(hrange, vrange):
    """Width that preserves equal aspect (mm/pixel) for fixed DATA_H."""
    w = int(round(DATA_H * (hrange / vrange))) if vrange > 0 else DATA_H
    return max(120, min(w, 900))


def _splat(img, px, py, colors):
    """Draw markers into an [H, W, 3] uint8 buffer (high values assumed last)."""
    H, W = img.shape[:2]
    r = MARKER_R
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            xx = px + dx
            yy = py + dy
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            img[yy[m], xx[m]] = colors[m]


def render_panel(coords_t, vals_t, ax_h, ax_v, hext, vext, lut_name,
                 vmin, vmax, title, flip_v=True):
    """Render one timestep of one view into a titled, color-barred RGB image."""
    hr = hext[1] - hext[0]
    vr = vext[1] - vext[0]
    W = _panel_width(hr, vr)
    H = DATA_H

    plot = np.zeros((H, W, 3), dtype=np.uint8)
    plot[:] = BG

    # draw order: low values first so hotspots land on top
    order = np.argsort(vals_t)
    h = coords_t[order, ax_h]
    v = coords_t[order, ax_v]
    colors = colorize(vals_t[order], vmin, vmax, lut_name)

    px = ((h - hext[0]) / hr * (W - 1)).astype(np.int64)
    py = ((v - vext[0]) / vr * (H - 1)).astype(np.int64)
    if flip_v:
        py = (H - 1) - py
    _splat(plot, px, py, colors)

    # assemble title strip + plot + colorbar strip
    panel = Image.new("RGB", (W, TITLE_H + H + BAR_H), BG)
    panel.paste(Image.fromarray(plot), (0, TITLE_H))
    draw = ImageDraw.Draw(panel)
    draw.text((4, 5), title, fill=(230, 230, 230))

    # colorbar: horizontal gradient with min/max labels
    bar = np.zeros((BAR_H, W, 3), dtype=np.uint8)
    grad = colorize(np.linspace(vmin, vmax, W), vmin, vmax, lut_name)
    bar[: BAR_H - 12] = grad[None, :, :]
    bar_img = Image.fromarray(bar)
    panel.paste(bar_img, (0, TITLE_H + H))
    draw.text((2, TITLE_H + H + BAR_H - 11), f"{vmin:.3g}", fill=(200, 200, 200))
    draw.text((W - 42, TITLE_H + H + BAR_H - 11), f"{vmax:.3g}", fill=(200, 200, 200))
    return panel


def compose_frame(panels, header):
    total_w = sum(p.width for p in panels) + GAP * (len(panels) - 1)
    head_h = 20
    frame = Image.new("RGB", (total_w, head_h + panels[0].height), BG)
    x = 0
    for p in panels:
        frame.paste(p, (x, head_h))
        x += p.width + GAP
    ImageDraw.Draw(frame).text((6, 5), header, fill=(255, 255, 255))
    return frame


def velocity_for(master_csv, sim_id):
    if not master_csv or not os.path.isfile(master_csv):
        return None
    with open(master_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sim_name") == sim_id:
                try:
                    return float(row["velocity_kmh"])
                except (KeyError, ValueError):
                    return None
    return None


def pick_sim(data_dir, master_csv):
    files = R.find_sim_files(data_dir)
    if not files:
        raise SystemExit(f"No VTKHDF sims found under {data_dir}")
    best, best_v = None, -1.0
    for p in files:
        sid = os.path.splitext(os.path.basename(p))[0]
        v = velocity_for(master_csv, sid) or 0.0
        if v > best_v:
            best, best_v = sid, v
    return best or os.path.splitext(os.path.basename(files[0]))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sim_id", nargs="?", default=None,
                    help="e.g. sim_00002 (default: auto-pick highest-velocity sim)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--master-csv", default=None)
    ap.add_argument("--field", choices=["stress_vm", "effective_plastic_strain", "both"],
                    default="both", help="which target(s) to color by")
    ap.add_argument("--stride", type=int, default=2, help="render every Nth timestep")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--exclude-rwall", action="store_true",
                    help="drop the rigid wall/pole part (RWALL_1)")
    ap.add_argument("--out", default=None, help="output .gif path")
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    master_csv = resolve_master_csv(args.master_csv)
    sim_id = args.sim_id or pick_sim(data_dir, master_csv)

    path = os.path.join(data_dir, sim_id, f"{sim_id}.vtkhdf")
    if not os.path.isfile(path):
        raise SystemExit(f"Not found: {path}")

    exclude = ("RWALL_1",) if args.exclude_rwall else ()
    coords, _src, _dst, targets = R.load_vtkhdf_file(path, exclude_parts=exclude)
    T, N, _ = coords.shape
    v_kmh = velocity_for(master_csv, sim_id)
    print(f"{sim_id}: coords {coords.shape}, {T} steps, "
          f"velocity {v_kmh if v_kmh is not None else '?'} km/h")

    # which views: (label, ax_h, ax_v, field_key, lut)
    vm = targets["stress_vm"]
    eps = targets["effective_plastic_strain"]
    # y is the long (car-width) axis -> keep it horizontal so panels stay wide
    views = []
    if args.field in ("stress_vm", "both"):
        views.append(("top y-x | von Mises", 1, 0, vm, "inferno"))
        views.append(("front y-z | von Mises", 1, 2, vm, "inferno"))
    if args.field in ("effective_plastic_strain", "both"):
        views.append(("top y-x | plastic strain", 1, 0, eps, "viridis"))

    frames_idx = list(range(0, T, max(1, args.stride)))
    # precompute fixed extents and robust shared color scales over rendered frames
    sampled = np.asarray(frames_idx)
    precomp = []
    for label, ax_h, ax_v, field, lut in views:
        hext, vext = _extents(coords, ax_h, ax_v)
        vmax = float(np.percentile(field[sampled], 99)) or 1.0
        precomp.append((label, ax_h, ax_v, field, lut, hext, vext, vmax))

    images = []
    for t in frames_idx:
        panels = [
            render_panel(coords[t], field[t], ax_h, ax_v, hext, vext, lut,
                         0.0, vmax, label)
            for (label, ax_h, ax_v, field, lut, hext, vext, vmax) in precomp
        ]
        head = f"{sim_id}   t={t:>3}/{T - 1}"
        if v_kmh is not None:
            head += f"   impact {v_kmh:.0f} km/h"
        head += f"   N={N}"
        images.append(compose_frame(panels, head))

    out = args.out or os.path.join(os.getcwd(), f"crash_{sim_id}.gif")
    duration_ms = int(1000 / max(1, args.fps))
    images[0].save(out, save_all=True, append_images=images[1:],
                   duration=duration_ms, loop=0, optimize=True)
    print(f"saved {out}  ({len(images)} frames, {images[0].size[0]}x{images[0].size[1]})")


if __name__ == "__main__":
    main()
