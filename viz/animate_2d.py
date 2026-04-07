"""2D geography animation: spatial wealth map (top) + histogram (bottom)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.collections as mc
import matplotlib.patches as mpatches
import numpy as np


def animate_2d(
    snapshots: list[dict],
    grid_size: int = 200,
    x_max: float = 600.0,
    bins: int = 50,
    fps: int = 20,
    output: str | None = None,
    mining_center: tuple[int, int] | None = None,
    mining_radius: int | None = None,
) -> None:
    """Animate spatial wealth map alongside wealth histogram.

    Each snapshot dict has:
        tick   : int
        agents : [(row, col, wealth), ...]
        pairs  : [(r1, c1, r2, c2), ...]  — active pairs this tick
    """
    fig, (ax_map, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 10),
        gridspec_kw={"height_ratios": [1.2, 1]}
    )
    fig.tight_layout(pad=3.0)

    # --- Spatial map setup ---
    ax_map.set_xlim(-0.5, grid_size - 0.5)
    ax_map.set_ylim(-0.5, grid_size - 0.5)
    ax_map.set_aspect("equal")
    ax_map.set_facecolor("#111111")
    ax_map.set_xlabel("col")
    ax_map.set_ylabel("row")

    scat = ax_map.scatter([], [], s=6, c=[], cmap="inferno",
                          vmin=0, vmax=x_max, alpha=0.9, zorder=3)
    cbar = fig.colorbar(scat, ax=ax_map, fraction=0.03, pad=0.02)
    cbar.set_label("wealth")

    # LineCollection for pair connections (drawn below agents)
    pair_lc = mc.LineCollection([], colors="limegreen", linewidths=0.8,
                                alpha=0.6, zorder=2)
    ax_map.add_collection(pair_lc)

    # Mining zone circle overlay (col=x, row=y)
    if mining_center is not None and mining_radius is not None:
        mine_circle = mpatches.Circle(
            (mining_center[1], mining_center[0]),  # (x=col, y=row)
            mining_radius,
            fill=False, edgecolor="gold", linewidth=1.5, linestyle="--",
            alpha=0.8, zorder=4,
        )
        ax_map.add_patch(mine_circle)

    # --- Histogram setup ---
    ax_hist.set_xlim(0, x_max)
    ax_hist.set_xlabel("wealth")
    ax_hist.set_ylabel("count")
    bar_edges = np.linspace(0, x_max, bins + 1)
    bar_width = bar_edges[1] - bar_edges[0]
    bar_centers = (bar_edges[:-1] + bar_edges[1:]) / 2
    bars = ax_hist.bar(bar_centers, np.zeros(bins), width=bar_width * 0.9,
                       color="steelblue", alpha=0.7, label="simulation")
    exp_line, = ax_hist.plot([], [], "r-", lw=2, label="theory (exp)")
    ax_hist.legend(loc="upper right", fontsize=8)
    title = fig.suptitle("", fontsize=11)

    def _gini(values: list[float]) -> float:
        if not values or sum(values) == 0:
            return 0.0
        sv = sorted(values)
        n = len(sv)
        weighted = sum((i + 1) * v for i, v in enumerate(sv))
        return (2.0 * weighted) / (n * sum(sv)) - (n + 1.0) / n

    def update(frame_idx: int):
        snap = snapshots[frame_idx]
        tick = snap["tick"]
        agents_data = snap["agents"]
        pairs_data = snap["pairs"]

        if not agents_data:
            return

        cols = [d[1] for d in agents_data]
        rows = [d[0] for d in agents_data]
        wealths = [d[2] for d in agents_data]

        # Update agent scatter (col=x, row=y)
        scat.set_offsets(np.column_stack([cols, rows]))
        scat.set_array(np.array(wealths))

        # Update pair lines: skip pairs that wrap across the torus boundary
        # (their raw coordinate difference would exceed half the grid size)
        half = grid_size // 2
        segments = [
            [[c1, r1], [c2, r2]]
            for r1, c1, r2, c2 in pairs_data
            if abs(c2 - c1) <= half and abs(r2 - r1) <= half
        ]
        pair_lc.set_segments(segments)

        # Update histogram
        counts, _ = np.histogram(wealths, bins=bar_edges)
        for bar, h in zip(bars, counts):
            bar.set_height(h)
        ax_hist.set_ylim(0, max(counts.max() * 1.15, 10))

        # Theoretical exponential overlay
        mean_w = sum(wealths) / len(wealths)
        n_agents = len(wealths)
        if mean_w > 0:
            xs = np.linspace(0, x_max, 300)
            lam = 1.0 / mean_w
            ys = lam * n_agents * bar_width * np.exp(-lam * xs)
            exp_line.set_data(xs, ys)

        gini = _gini(wealths)
        n_pairs = len(pairs_data)
        title.set_text(
            f"Tick {tick}  |  n={len(wealths)}  |  pairs={n_pairs}  "
            f"|  mean={mean_w:.1f}  |  Gini={gini:.3f}"
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=1000 // fps,
        blit=False,
    )

    if output:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(output, writer=writer)
        print(f"Saved: {output}")
    else:
        plt.show()

    plt.close(fig)
