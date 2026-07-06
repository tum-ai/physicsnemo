"""
compare_models.py
=================
Compare multiple models by loading their predicted .npz files, computing relative L2 
displacement errors over time, printing a summary table, and saving an overlay plot 
of the error curves. Does not log to WandB.
"""

import argparse
import os
import glob
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
# tabulate is imported locally in main() to allow clean fallback if not present


def compute_rel_l2(pred_pos, exact_pos):
    """
    Compute relative L2 position error per timestep.
    pred_pos:  [T, N, 3]
    exact_pos: [T, N, 3]
    """
    err = pred_pos - exact_pos                           # [T, N, 3]
    node_err = np.linalg.norm(err, axis=2)               # [T, N]
    num = np.sqrt((node_err ** 2).sum(axis=1))           # [T]
    disp = exact_pos - exact_pos[0:1]                    # [T, N, 3]
    den = np.sqrt((np.linalg.norm(disp, axis=2) ** 2).sum(axis=1))  # [T]
    
    rel_l2 = num / np.maximum(den, 1e-8)
    rel_l2[0] = 0.0  # reference frame t=0 has no displacement
    return rel_l2


def multi_line_plot(series, T, out_path, title):
    """
    Minimal multi-series line plot (no matplotlib).
    series: list of (label, y_array, rgb_tuple)
    """
    W, H = 900, 360
    ml, mr, mt, mb = 70, 240, 34, 40  # Large right margin for legend names
    img = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    x0, x1 = ml, W - mr
    y0, y1 = H - mb, mt
    
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
        return y0 + (y1 - y0) * (min(v, ymax) / ymax)

    # Plot each series
    for li, (label, y, rgb) in enumerate(series):
        pts = [(px(t), py(y[t])) for t in range(len(y))]
        d.line(pts, fill=rgb, width=2)
        
        # Legend drawing
        lx = x1 + 15
        ly = mt + 10 + li * 22
        d.rectangle([lx, ly + 2, lx + 18, ly + 12], fill=rgb)
        d.text((lx + 25, ly), label, fill=(30, 30, 30))
        
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Compare relative L2 curves across multiple models.")
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        required=True,
        help="List of model directories (e.g. outputs/baseline outputs/enhanced).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Custom labels corresponding to each model directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/comparisons",
        help="Directory to save the comparison plot.",
    )
    args = parser.parse_args()

    # Resolve labels
    if args.labels:
        if len(args.labels) != len(args.model_dirs):
            raise ValueError("Number of --labels must match number of --model_dirs")
        labels = args.labels
    else:
        labels = [os.path.basename(os.path.abspath(d)) for d in args.model_dirs]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_data = []
    comparison_table = []
    
    T = None

    print("Comparing models:")
    for model_dir, label in zip(args.model_dirs, labels):
        npz_dir = Path(model_dir)
        if (npz_dir / "npz_files").is_dir():
            npz_dir = npz_dir / "npz_files"
        npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
        
        if not npz_files:
            print(f"  Warning: No .npz files found in {npz_dir}. Skipping.")
            continue

        print(f"  Processing {label} ({len(npz_files)} files)...")

        all_rel_curves = []
        for path in npz_files:
            data = np.load(path)
            if "pred_pos" not in data or "exact_pos" not in data:
                continue
            pred_pos = data["pred_pos"]
            exact_pos = data["exact_pos"]
            rel_l2 = compute_rel_l2(pred_pos, exact_pos)
            all_rel_curves.append(rel_l2)

        if not all_rel_curves:
            continue

        # Average relative L2 curve across all files
        avg_curve = np.mean(all_rel_curves, axis=0)
        if T is None:
            T = len(avg_curve)
            
        series_data.append((label, avg_curve))

        # Calculate metrics
        mean_rel = float(np.nanmean(avg_curve[1:]))
        max_rel = float(avg_curve.max())
        final_rel = float(avg_curve[-1])
        
        comparison_table.append([
            label,
            f"{mean_rel:.5f}",
            f"{max_rel:.5f}",
            f"{final_rel:.5f}"
        ])

    if not series_data:
        print("Error: No data available for comparison.")
        return

    # Define color palette for overlay curves
    colors = [
        (200, 30, 30),   # Red
        (30, 90, 200),   # Blue
        (30, 160, 60),   # Green
        (180, 130, 50),  # Brown
        (130, 50, 180),  # Purple
    ]

    # Generate plotting series
    plot_series = []
    for i, (label, curve) in enumerate(series_data):
        color = colors[i % len(colors)]
        plot_series.append((label, curve, color))

    # Save plot locally
    plot_path = output_dir / "relative_l2_comparison.png"
    multi_line_plot(plot_series, T, plot_path, "Model Comparison: Average Relative L2 Error vs Timestep")
    print(f"Saved comparison plot to {plot_path}")

    # Output table
    headers = ["Model / Encoder", "Mean Rel L2", "Max Rel L2", "Final Rel L2"]
    
    try:
        from tabulate import tabulate
        table_str = tabulate(comparison_table, headers=headers, tablefmt="grid")
    except ImportError:
        row_format = "{:<30} | {:<12} | {:<12} | {:<12}"
        lines = [row_format.format(*headers), "-" * 75]
        for row in comparison_table:
            lines.append(row_format.format(*row))
        table_str = "\n".join(lines)

    # Print to terminal
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(table_str)
    print("=" * 80 + "\n")

    # Save local summary file
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  MODEL COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(table_str + "\n")
        f.write("=" * 80 + "\n")
    print(f"Saved comparison text summary to {summary_path}")


if __name__ == "__main__":
    main()
