"""Animated visualization of wealth distribution over time."""

from __future__ import annotations

import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.stats import StatsCollector


def animate_wealth_distribution(stats: StatsCollector, config: dict) -> None:
    """Create animated histogram showing wealth distribution evolution."""
    viz_cfg = config.get("viz", {})
    fps = viz_cfg.get("fps", 30)
    bins = viz_cfg.get("bins", 50)
    x_max = viz_cfg.get("x_max", 500)
    output = viz_cfg.get("output", None)

    # Sample snapshots to keep animation manageable (~30 sec)
    total_ticks = len(stats.history)
    target_frames = fps * 30
    every_n = max(1, total_ticks // target_frames)
    snapshots = stats.get_wealth_snapshots(every_n=every_n)

    if not snapshots:
        print("No data to animate.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    def update(frame_idx: int):
        ax.clear()
        tick, wealths = snapshots[frame_idx]
        wealths_arr = np.array(wealths)

        # Histogram
        ax.hist(
            wealths_arr, bins=bins, range=(0, x_max), density=True,
            color="steelblue", edgecolor="white", alpha=0.8,
        )

        # Theoretical exponential overlay
        mean_w = np.mean(wealths_arr) if len(wealths_arr) > 0 else 1.0
        if mean_w > 0:
            x = np.linspace(0, x_max, 300)
            ax.plot(
                x, (1.0 / mean_w) * np.exp(-x / mean_w),
                "r-", lw=2, label=f"Exp(mean={mean_w:.1f})",
            )

        # Look up gini from history
        gini = stats.history[tick]["gini"] if tick < len(stats.history) else 0.0

        ax.set_xlabel("Wealth")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Wealth Distribution  |  Tick {tick}  |  Gini = {gini:.3f}"
        )
        ax.legend(loc="upper right")
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 0.05)

    ani = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=1000 / fps, repeat=False,
    )

    if output:
        outdir = os.path.dirname(output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        ani.save(output, writer="ffmpeg", fps=fps)
        print(f"Saved animation to {output}")
    else:
        plt.show()
