"""
Plot training loss from TensorBoard event files.
Usage:
    python plot_training_loss.py --logdir outputs/figconvnet/tensorboard_logs --out loss_figconvnet.png
    python plot_training_loss.py --logdir outputs/geotransolver/tensorboard_logs --out loss_geotransolver.png
    python plot_training_loss.py \
        --logdir outputs/figconvnet/tensorboard_logs outputs/geotransolver/tensorboard_logs \
        --labels FIGConvNet GeoTransolver \
        --out loss_comparison.png
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(logdir: str, tag: str = "train_loss") -> tuple:
    """Return (steps, values) arrays from a TensorBoard log directory."""
    pattern = os.path.join(logdir, "events.out.tfevents.*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {logdir}")

    steps, values = [], []
    for f in files:
        ea = EventAccumulator(f)
        ea.Reload()
        available = ea.Tags().get("scalars", [])
        if tag not in available:
            continue
        for event in ea.Scalars(tag):
            steps.append(event.step)
            values.append(event.value)

    if not steps:
        raise ValueError(
            f"Tag '{tag}' not found in {logdir}. Available: {available}"
        )

    steps = np.array(steps)
    values = np.array(values)
    order = np.argsort(steps)
    return steps[order], values[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        nargs="+",
        required=True,
        help="One or more TensorBoard log directories",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Legend labels (one per logdir). Defaults to directory name.",
    )
    parser.add_argument(
        "--tag", default="loss", help="Scalar tag to plot (default: loss)"
    )
    parser.add_argument("--out", default="loss_plot.png", help="Output PNG path")
    parser.add_argument(
        "--smooth", type=int, default=1, help="Rolling-average window (default: 1 = off)"
    )
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(d)) for d in args.logdir]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (logdir, label) in enumerate(zip(args.logdir, labels)):
        steps, values = load_scalar(logdir, args.tag)
        color = colors[i % len(colors)]

        if args.smooth > 1:
            kernel = np.ones(args.smooth) / args.smooth
            smoothed = np.convolve(values, kernel, mode="valid")
            smooth_steps = steps[args.smooth - 1 :]
            ax.plot(steps, values, alpha=0.25, color=color, linewidth=0.8)
            ax.plot(smooth_steps, smoothed, color=color, linewidth=1.8, label=label)
        else:
            ax.plot(steps, values, color=color, linewidth=1.5, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(args.tag.replace("_", " ").title())
    ax.set_title("Training Loss — Bumper Beam Benchmark")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
