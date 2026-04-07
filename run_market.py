"""Market simulation: necessity ↔ coin bilateral trade with spatial farm production."""

from __future__ import annotations

import argparse
import math
import os

import yaml
import random
import numpy as np

from src.agent import Agent, CoinAgent, Portfolio
from src.environment import (
    Environment,
    GridMovementEvent,
    NecessityLifecycleEvent,
    StarvationWall,
)
from src.interaction import NecessityTradeInteraction
from src.matcher import Grid2DSelector, Matcher, WeightedSumCombiner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/market.yaml")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    rng = random.Random(seed)
    Agent.reset_id_counter()

    sim_cfg   = config["simulation"]
    max_ticks = sim_cfg["max_ticks"]
    grid_size = sim_cfg["grid_size"]

    agent_cfg    = config["agents"]
    n_agents     = agent_cfg["count"]
    initial_coin = agent_cfg["initial_coin"]
    initial_nec  = agent_cfg["initial_nec"]

    eco_cfg          = config["economy"]
    consumption_rate = eco_cfg["consumption_rate"]
    min_reserve      = eco_cfg["min_reserve"]

    farm_cfg     = config["farms"]
    farm_centers = [tuple(c) for c in farm_cfg["centers"]]
    farm_radius  = farm_cfg["radius"]

    radius = config.get("matcher", {}).get("radius", 5)

    viz_cfg      = config.get("viz", {})
    sample_every = viz_cfg.get("sample_every", 5)
    fps          = viz_cfg.get("fps", 20)
    output       = viz_cfg.get("output", "output/market.mp4")

    # Theoretical equilibrium price = total_coin / total_nec
    total_coin = n_agents * initial_coin
    total_nec  = n_agents * initial_nec
    eq_price   = total_coin / total_nec
    print(f"Theoretical equilibrium price: {eq_price:.3f} coin/nec")

    # Farm coverage info
    farm_area = len(farm_centers) * math.pi * farm_radius ** 2
    farm_frac = farm_area / grid_size ** 2
    print(f"Farm coverage: {farm_frac*100:.1f}% of map, expected {n_agents*farm_frac:.0f} agents on farms")

    # Build agents
    agents = []
    for _ in range(n_agents):
        portfolio = Portfolio({"coin": initial_coin, "nec": initial_nec})
        a = CoinAgent.__new__(CoinAgent)
        Agent.__init__(a, portfolio, rng)
        a.grid_pos = (rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1))
        agents.append(a)

    # Events (tick order): move → [consume + farm produce] → starvation wall
    events = [
        GridMovementEvent(grid_size=grid_size),
        NecessityLifecycleEvent(
            farm_centers=farm_centers,
            radius=farm_radius,
            grid_size=grid_size,
            consumption_rate=consumption_rate,
        ),
        StarvationWall(),
    ]

    selector    = Grid2DSelector(grid_size=grid_size, radius=radius)
    matcher     = Matcher(factors=[], combiner=WeightedSumCombiner(), selector=selector)
    interaction = NecessityTradeInteraction(
        consumption_rate=consumption_rate,
        min_reserve=min_reserve,
    )

    env = Environment(matcher=matcher, interaction=interaction, events=events)
    env.add_agents(agents)

    # History for stats and viz
    price_history:    list[float] = []   # avg trade price per tick
    starving_history: list[int]   = []   # number of starving agents per tick
    trade_history:    list[int]   = []   # number of trades per tick

    snapshots: list[dict] = []           # for animation

    print(f"\nRunning {max_ticks} ticks | {n_agents} agents | {grid_size}×{grid_size} | R={radius}")
    print(f"{'tick':>6} | {'trades':>6} | {'avg_price':>9} | {'starving':>8} | "
          f"{'mean_coin':>9} | {'mean_nec':>8} | {'nec_total':>9}")
    print("-" * 80)

    for tick in range(max_ticks):
        env.process_events(tick, rng)
        pairs   = env.do_matching(rng)
        results = env.do_interactions(pairs, rng)
        env.do_lifecycle()

        # Collect stats
        alive = [a for a in env.agents if a.alive]
        trades     = [r for r in results if r.get("trade")]
        n_trades   = len(trades)
        n_starving = sum(1 for a in alive if a.portfolio.get("nec") == 0)
        avg_price  = (sum(r["price"] for r in trades) / n_trades) if trades else float("nan")
        mean_coin  = np.mean([a.portfolio.get("coin") for a in alive]) if alive else 0
        mean_nec   = np.mean([a.portfolio.get("nec")  for a in alive]) if alive else 0
        total_nec_now = sum(a.portfolio.get("nec") for a in alive)

        price_history.append(avg_price)
        starving_history.append(n_starving)
        trade_history.append(n_trades)

        if (tick + 1) % 100 == 0:
            print(f"{tick+1:>6} | {n_trades:>6} | {avg_price:>9.3f} | {n_starving:>8} | "
                  f"{mean_coin:>9.1f} | {mean_nec:>8.2f} | {total_nec_now:>9.1f}")

        if tick % sample_every == 0:
            snap_agents = [
                (a.grid_pos[0], a.grid_pos[1],
                 a.portfolio.get("coin"), a.portfolio.get("nec"))
                for a in alive if a.grid_pos is not None
            ]
            snap_pairs = [
                (a.grid_pos[0], a.grid_pos[1], b.grid_pos[0], b.grid_pos[1])
                for a, b in pairs
                if a.grid_pos and b.grid_pos
            ]
            snapshots.append({
                "tick": tick,
                "agents": snap_agents,
                "pairs": snap_pairs,
                "avg_price": avg_price,
                "eq_price": eq_price,
            })

    # Final summary
    valid_prices = [p for p in price_history if not math.isnan(p)]
    if valid_prices:
        last200 = valid_prices[-200:]
        print(f"\n=== Price convergence ===")
        print(f"  Theoretical eq price : {eq_price:.3f}")
        print(f"  First 100 ticks avg  : {np.mean(valid_prices[:100]):.3f}")
        print(f"  Last 200 ticks avg   : {np.mean(last200):.3f}")
        print(f"  Last 200 ticks std   : {np.std(last200):.3f}")

    print(f"\n=== Final state ===")
    alive = [a for a in env.agents if a.alive]
    coins = [a.portfolio.get("coin") for a in alive]
    necs  = [a.portfolio.get("nec")  for a in alive]
    print(f"  Agents alive : {len(alive)}")
    print(f"  Total coin   : {sum(coins):.1f}  (initial: {total_coin:.1f})")
    print(f"  Total nec    : {sum(necs):.1f}   (initial: {total_nec:.1f})")
    print(f"  Starving now : {sum(1 for n in necs if n == 0)}")

    if not args.no_viz:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        _animate_market(snapshots, grid_size, farm_centers, farm_radius, eq_price, fps, output)


def _animate_market(snapshots, grid_size, farm_centers, farm_radius, eq_price, fps, output):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.collections as mc
    import matplotlib.patches as mpatches
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_coin, ax_nec = axes[0]
    ax_price, ax_hist = axes[1]
    fig.tight_layout(pad=3.0)

    def setup_map(ax, title, cmap, vmax):
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#111111")
        ax.set_title(title)
        scat = ax.scatter([], [], s=4, c=[], cmap=cmap, vmin=0, vmax=vmax, alpha=0.9)
        fig.colorbar(scat, ax=ax, fraction=0.03, pad=0.02)
        # Draw farm circles
        for cr, cc in farm_centers:
            circle = mpatches.Circle((cc, cr), farm_radius, fill=False,
                                     edgecolor="gold", linewidth=1.2, linestyle="--", alpha=0.7)
            ax.add_patch(circle)
        return scat

    scat_coin = setup_map(ax_coin, "coin", "plasma",  500)
    scat_nec  = setup_map(ax_nec,  "nec",  "YlGn",    100)

    # Price time series
    ax_price.set_xlabel("tick")
    ax_price.set_ylabel("avg trade price (coin/nec)")
    ax_price.axhline(eq_price, color="red", linewidth=1.5, linestyle="--", label=f"eq={eq_price:.2f}")
    ax_price.legend(fontsize=8)
    price_line, = ax_price.plot([], [], "b-", linewidth=1, alpha=0.8)
    ax_price.set_xlim(0, len(snapshots) * (snapshots[1]["tick"] - snapshots[0]["tick"]) if len(snapshots) > 1 else 1000)
    ax_price.set_ylim(0, eq_price * 4)

    # Nec histogram
    ax_hist.set_xlabel("nec")
    ax_hist.set_ylabel("count")
    bins = np.linspace(0, 100, 50)
    bar_centers = (bins[:-1] + bins[1:]) / 2
    bars = ax_hist.bar(bar_centers, np.zeros(len(bar_centers)), width=bins[1]-bins[0], color="seagreen", alpha=0.7)

    title = fig.suptitle("", fontsize=11)
    price_xs, price_ys = [], []

    def update(fi):
        snap = snapshots[fi]
        tick = snap["tick"]
        agents_data = snap["agents"]
        if not agents_data:
            return

        cols   = [d[1] for d in agents_data]
        rows   = [d[0] for d in agents_data]
        coins  = [d[2] for d in agents_data]
        necs   = [d[3] for d in agents_data]

        scat_coin.set_offsets(np.column_stack([cols, rows]))
        scat_coin.set_array(np.array(coins))
        scat_nec.set_offsets(np.column_stack([cols, rows]))
        scat_nec.set_array(np.array(necs))

        # Price series
        p = snap.get("avg_price", float("nan"))
        if not math.isnan(p):
            price_xs.append(tick)
            price_ys.append(p)
        price_line.set_data(price_xs, price_ys)

        # Nec histogram
        counts, _ = np.histogram(necs, bins=bins)
        for bar, h in zip(bars, counts):
            bar.set_height(h)
        ax_hist.set_ylim(0, max(counts.max() * 1.15, 10))

        n_starving = sum(1 for n in necs if n == 0)
        avg_p = f"{p:.3f}" if not math.isnan(p) else "n/a"
        title.set_text(f"Tick {tick} | n={len(agents_data)} | starving={n_starving} | price={avg_p}")

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000//fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(output, writer=writer)
    print(f"Saved: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
