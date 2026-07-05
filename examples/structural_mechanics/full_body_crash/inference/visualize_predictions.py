"""
visualize_predictions.py
========================
Generate 3-panel animated GIFs showing ground truth displacement, predicted displacement, 
and position error heatmap over time. Logs the GIFs to WandB.
"""

import argparse
import os
import glob
import numpy as np
import wandb
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
# Color maps (interpolated to 256 entries)
# ─────────────────────────────────────────────────────────────────────────────
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


_LUTS = {
    "inferno": _build_lut(_INFERNO),
    "viridis": _build_lut(_VIRIDIS)
}


def colorize(values, vmin, vmax, lut_name):
    """Map a 1-D array of scalars to [N, 3] uint8 RGB via a 256-entry LUT."""
    lut = _LUTS[lut_name]
    if vmax <= vmin:
        vmax = vmin + 1e-9
    t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (t * (len(lut) - 1)).astype(np.int64)
    return lut[idx].astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Rendering Helpers
# ─────────────────────────────────────────────────────────────────────────────
BG = (16, 16, 20)
DATA_H = 360          # pixels of plot area height per panel
TITLE_H = 22
BAR_H = 26
GAP = 10
PAD_FRAC = 0.04       # world-space padding around the data extents
MARKER_R = 1          # marker radius in pixels


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
    """Draw markers into an [H, W, 3] uint8 buffer."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate GIFs for test set crash predictions.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the predicted .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save the generated GIFs.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="geotransolver-crash-eval",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="WandB run ID to resume/log to.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Render every N-th timestep.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for output GIFs.",
    )
    parser.add_argument(
        "--ax_h",
        type=int,
        default=1,
        help="Horizontal axis coordinate (default 1 for Y / lateral width).",
    )
    parser.add_argument(
        "--ax_v",
        type=int,
        default=0,
        help="Vertical axis coordinate (default 0 for X / longitudinal length).",
    )
    args = parser.parse_args()

    # Find npz files
    search_path = os.path.join(args.data_dir, "*.npz")
    npz_files = glob.glob(search_path)
    if not npz_files:
        print(f"Error: No .npz files found in {args.data_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating visualizations for {len(npz_files)} files...")

    # Start WandB session
    if args.wandb_run_id:
        wandb.init(
            project="geotransolver-crash",
            id=args.wandb_run_id,
            resume="must",
        )
    else:
        wandb.init(project=args.project, name=f"visualizations_{os.path.basename(args.data_dir)}")

    for path in sorted(npz_files):
        run_name = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path)
        
        if "pred_pos" not in data or "exact_pos" not in data:
            continue
            
        pred_pos = data["pred_pos"]    # [T, N, 3]
        exact_pos = data["exact_pos"]  # [T, N, 3]
        T, N, _ = pred_pos.shape

        print(f"  Processing {run_name} (T={T}, N={N})...")

        # 1. Compute displacements (crumple wave) and errors
        gt_disp = np.linalg.norm(exact_pos - exact_pos[0:1], axis=2)     # [T, N]
        pred_disp = np.linalg.norm(pred_pos - pred_pos[0:1], axis=2)     # [T, N]
        node_err = np.linalg.norm(pred_pos - exact_pos, axis=2)          # [T, N]

        # 2. Precompute robust color limits (99th percentile across time to ignore noise)
        max_disp = float(np.percentile(gt_disp, 99)) or 1.0
        max_err = float(np.percentile(node_err, 99)) or 1.0

        # Shared coordinate boundary mapping
        both_coords = np.concatenate([exact_pos, pred_pos], axis=1)      # [T, 2N, 3]
        hext, vext = _extents(both_coords, args.ax_h, args.ax_v)

        frames_idx = list(range(0, T, max(1, args.stride)))
        images = []

        for t in frames_idx:
            panels = [
                render_panel(
                    exact_pos[t], gt_disp[t], args.ax_h, args.ax_v, hext, vext, 
                    "viridis", 0.0, max_disp, "ground truth | displacement"
                ),
                render_panel(
                    pred_pos[t], pred_disp[t], args.ax_h, args.ax_v, hext, vext, 
                    "viridis", 0.0, max_disp, "predicted | displacement"
                ),
                render_panel(
                    pred_pos[t], node_err[t], args.ax_h, args.ax_v, hext, vext, 
                    "inferno", 0.0, max_err, "absolute error (mm)"
                )
            ]
            
            # Mean error at this step
            step_mean_err = node_err[t].mean()
            header = f"{run_name}   t={t:>3}/{T - 1}   mean_err={step_mean_err:.2f} mm"
            images.append(compose_frame(panels, header))

        # Save GIF locally
        gif_path = os.path.join(args.output_dir, f"{run_name}_crash.gif")
        duration_ms = int(1000 / max(1, args.fps))
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=duration_ms, loop=0, optimize=True
        )
        print(f"    Saved GIF locally: {gif_path}")

        # Log GIF directly to WandB as media
        wandb.log({
            f"media/{run_name}_trajectory": wandb.Video(gif_path, fps=args.fps, format="gif")
        })

    wandb.finish()
    print("All visualizations created and uploaded to WandB.")


if __name__ == "__main__":
    main()
