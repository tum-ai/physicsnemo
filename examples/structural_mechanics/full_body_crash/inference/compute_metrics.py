"""
compute_metrics.py
==================
Compute quantitative metrics (relative L2 position error and absolute error in mm) 
across all evaluation npz files in a directory. Prints results to terminal, saves local 
summaries/plots, and logs to WandB.
"""

import argparse
import os
import glob
from pathlib import Path
import numpy as np
import wandb
from PIL import Image, ImageDraw
# tabulate is imported locally in main() to allow clean fallback if not present


def compute_errors(pred_pos, exact_pos):
    """
    Compute per-timestep position errors.
    pred_pos:  [T, N, 3]
    exact_pos: [T, N, 3]
    """
    # Absolute error in original unit (mm)
    err = pred_pos - exact_pos                           # [T, N, 3]
    node_err = np.linalg.norm(err, axis=2)               # [T, N]
    abs_err_curve = node_err.mean(axis=1)               # [T]

    # Relative L2 displacement error
    num = np.sqrt((node_err ** 2).sum(axis=1))           # [T]
    disp = exact_pos - exact_pos[0:1]                    # [T, N, 3]
    den = np.sqrt((np.linalg.norm(disp, axis=2) ** 2).sum(axis=1))  # [T]
    
    rel_l2_curve = num / np.maximum(den, 1e-8)
    rel_l2_curve[0] = 0.0  # reference frame t=0 has no displacement

    return {
        "abs_err": abs_err_curve,
        "rel_l2": rel_l2_curve,
    }


def line_plot(series, T, out_path, title):
    """Minimal multi-series line plot (no matplotlib). series: list of (label, y, rgb)."""
    W, H = 900, 360
    ml, mr, mt, mb = 70, 20, 34, 40
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

    for li, (label, y, rgb) in enumerate(series):
        pts = [(px(t), py(y[t])) for t in range(len(y))]
        d.line(pts, fill=rgb, width=2 if li == 0 else 1)
        d.text((x1 - 200, mt + 4 + li * 14), label, fill=rgb)
        
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics on inference outputs.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the predicted .npz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save local metrics logs and plots.",
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
    args = parser.parse_args()

    # Find all .npz files in the directory
    search_path = os.path.join(args.data_dir, "*.npz")
    npz_files = glob.glob(search_path)
    if not npz_files:
        print(f"Error: No .npz files found in {args.data_dir}")
        return

    print(f"Found {len(npz_files)} simulation outputs in {args.data_dir}")

    # Accumulate metrics
    run_curves = {}
    run_stats = {}
    
    all_abs_curves = []
    all_rel_curves = []

    for path in sorted(npz_files):
        run_name = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path)
        
        if "pred_pos" not in data or "exact_pos" not in data:
            print(f"Warning: Skipping {path} - missing pred_pos or exact_pos.")
            continue
            
        pred_pos = data["pred_pos"]    # [T, N, 3]
        exact_pos = data["exact_pos"]  # [T, N, 3]
        
        curves = compute_errors(pred_pos, exact_pos)
        
        # Calculate summary statistics
        stats = {
            "mean_abs": float(curves["abs_err"].mean()),
            "max_abs":  float(curves["abs_err"].max()),
            "final_abs": float(curves["abs_err"][-1]),
            "mean_rel": float(np.nanmean(curves["rel_l2"][1:])), # Skip t=0 (0.0)
            "max_rel":  float(curves["rel_l2"].max()),
            "final_rel": float(curves["rel_l2"][-1]),
        }
        
        run_curves[run_name] = curves
        run_stats[run_name] = stats
        
        all_abs_curves.append(curves["abs_err"])
        all_rel_curves.append(curves["rel_l2"])

    if not run_stats:
        print("No valid evaluation arrays were processed.")
        return

    # Compute averages across all runs
    T = len(all_abs_curves[0])
    avg_abs_curve = np.mean(all_abs_curves, axis=0)
    avg_rel_curve = np.mean(all_rel_curves, axis=0)
    
    avg_stats = {
        "mean_abs": float(avg_abs_curve.mean()),
        "max_abs":  float(avg_abs_curve.max()),
        "final_abs": float(avg_abs_curve[-1]),
        "mean_rel": float(np.nanmean(avg_rel_curve[1:])),
        "max_rel":  float(avg_rel_curve.max()),
        "final_rel": float(avg_rel_curve[-1]),
    }

    # ── Format results table ──────────────────────────────────────────────
    headers = [
        "Simulation", 
        "Mean Abs (mm)", "Max Abs (mm)", "Final Abs (mm)", 
        "Mean Rel L2", "Max Rel L2", "Final Rel L2"
    ]
    table_data = []
    for run_name, stats in run_stats.items():
        table_data.append([
            run_name,
            f"{stats['mean_abs']:.4f}", f"{stats['max_abs']:.4f}", f"{stats['final_abs']:.4f}",
            f"{stats['mean_rel']:.4f}", f"{stats['max_rel']:.4f}", f"{stats['final_rel']:.4f}",
        ])
    # Add separator and average row
    table_data.append(["---"] * len(headers))
    table_data.append([
        "AVERAGE",
        f"{avg_stats['mean_abs']:.4f}", f"{avg_stats['max_abs']:.4f}", f"{avg_stats['final_abs']:.4f}",
        f"{avg_stats['mean_rel']:.4f}", f"{avg_stats['max_rel']:.4f}", f"{avg_stats['final_rel']:.4f}",
    ])

    try:
        from tabulate import tabulate
        table_str = tabulate(table_data, headers=headers, tablefmt="grid")
    except ImportError:
        # Fallback to simple manual formatting
        row_format = "{:<25} | {:<12} | {:<12} | {:<12} | {:<11} | {:<11} | {:<11}"
        lines = [row_format.format(*headers), "-" * 110]
        for row in table_data:
            lines.append(row_format.format(*row))
        table_str = "\n".join(lines)

    # Print to terminal
    print("\n" + "=" * 90)
    print("  EVALUATION SUMMARY METRICS")
    print("=" * 90)
    print(table_str)
    print("=" * 90 + "\n")

    # ── Save local summaries and plots ────────────────────────────────────
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write summary table
        txt_path = out_dir / "metrics_summary.txt"
        with open(txt_path, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("  EVALUATION SUMMARY METRICS\n")
            f.write("=" * 90 + "\n")
            f.write(table_str + "\n")
            f.write("=" * 90 + "\n")
        print(f"Saved local summary text to {txt_path}")
        
        # 2. Generate and save offline line plot
        plot_path = out_dir / "relative_error_curves.png"
        series = [("Average (All Runs)", avg_rel_curve, (200, 30, 30))]
        for i, (run_name, curves) in enumerate(run_curves.items()):
            # Cycle through a few colors for individual runs
            colors = [(100, 100, 100), (80, 130, 190), (80, 170, 100), (180, 130, 50)]
            color = colors[i % len(colors)]
            series.append((run_name, curves["rel_l2"], color))
            
        line_plot(series, T, plot_path, "Relative L2 position error vs Timestep")
        print(f"Saved local relative L2 error curves to {plot_path}")

    # ── Log plots & tables to WandB ───────────────────────────────────────
    if args.wandb_run_id:
        wandb.init(
            project="geotransolver-crash",
            id=args.wandb_run_id,
            resume="must",
        )
    else:
        wandb.init(project=args.project, name=f"evaluation_{os.path.basename(args.data_dir)}")
    
    # 1. Log curves per timestep
    print("Uploading evaluation curves to WandB...")
    for t in range(T):
        log_dict = {"timestep": t}
        for run_name, curves in run_curves.items():
            log_dict[f"eval/{run_name}/rel_l2"] = curves["rel_l2"][t]
            log_dict[f"eval/{run_name}/abs_err_mm"] = curves["abs_err"][t]
        
        # Add average curves
        log_dict["eval/average/rel_l2"] = avg_rel_curve[t]
        log_dict["eval/average/abs_err_mm"] = avg_abs_curve[t]
        
        wandb.log(log_dict)

    # 2. Log summary spreadsheet table
    wb_table = wandb.Table(columns=headers)
    for row in table_data:
        if row[0] == "---":
            continue
        wb_table.add_data(
            row[0],
            float(row[1]), float(row[2]), float(row[3]),
            float(row[4]), float(row[5]), float(row[6])
        )
    wandb.log({"eval/summary_table": wb_table})
    
    # Save overall summary metrics to run summary
    for k, v in avg_stats.items():
        wandb.summary[f"eval/average/{k}"] = v

    wandb.finish()
    print("Evaluation completed and logged to WandB.")


if __name__ == "__main__":
    main()
