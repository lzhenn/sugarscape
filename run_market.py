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
    CoinRedistributionEvent,
    GardenEvent,
    GridMovementEvent,
    NecessityLifecycleEvent,
    NecRedistributionEvent,
    StarvationWall,
)
from src.interaction import MarketInteraction
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

    eco_cfg              = config["economy"]
    consumption_rate     = eco_cfg["consumption_rate"]
    min_reserve          = eco_cfg["min_reserve"]
    nec_luxury_ticks     = eco_cfg.get("nec_luxury_ticks", 200)
    nec_luxury           = nec_luxury_ticks * consumption_rate
    nec_tax_cfg          = eco_cfg.get("nec_tax", {})
    nec_tax_start        = nec_tax_cfg.get("start_tick", 0)
    nec_tax_interval     = nec_tax_cfg.get("interval", 100)
    nec_tax_delay        = nec_tax_cfg.get("delay_ticks", 50)
    nec_tax_top          = nec_tax_cfg.get("top_fraction", 0.01)
    nec_tax_rate         = nec_tax_cfg.get("tax_rate", 0.1)
    nec_tax_bottom       = nec_tax_cfg.get("bottom_fraction", 0.1)

    farm_production      = config["farms"].get("production", 1.0)

    farm_cfg     = config["farms"]
    farm_centers = [tuple(c) for c in farm_cfg["centers"]]
    farm_radius  = farm_cfg["radius"]

    garden_cfg      = config.get("garden", {})
    garden_center   = tuple(garden_cfg.get("center", [50, 50]))
    garden_radius   = garden_cfg.get("radius", 30)
    flower_max_age  = garden_cfg.get("flower_max_age", 50)

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

    # Precompute farm and garden cells
    farm_cells: set[tuple[int, int]] = set()
    for cx, cy in farm_centers:
        for row in range(grid_size):
            for col in range(grid_size):
                if (row - cx) ** 2 + (col - cy) ** 2 <= farm_radius ** 2:
                    farm_cells.add((row, col))

    garden_cells: set[tuple[int, int]] = set()
    gcx, gcy = garden_center
    for row in range(grid_size):
        for col in range(grid_size):
            if (row - gcx) ** 2 + (col - gcy) ** 2 <= garden_radius ** 2:
                garden_cells.add((row, col))

    print(f"Garden coverage: {len(garden_cells)/grid_size**2*100:.1f}% of map")

    # Events (tick order): move → [consume + farm produce] → garden → starvation wall
    events = [
        GridMovementEvent(grid_size=grid_size, farm_cells=farm_cells,
                          garden_cells=garden_cells,
                          nec_luxury=nec_luxury, consumption_rate=consumption_rate),
        NecessityLifecycleEvent(
            farm_centers=farm_centers,
            radius=farm_radius,
            grid_size=grid_size,
            consumption_rate=consumption_rate,
            production_rate=farm_production,
        ),
        GardenEvent(garden_cells=garden_cells),
        NecRedistributionEvent(
            start_tick=nec_tax_start,
            interval=nec_tax_interval,
            delay_ticks=nec_tax_delay,
            top_fraction=nec_tax_top,
            tax_rate=nec_tax_rate,
            bottom_fraction=nec_tax_bottom,
        ),
        StarvationWall(),
    ]

    selector    = Grid2DSelector(grid_size=grid_size, radius=radius)
    matcher     = Matcher(factors=[], combiner=WeightedSumCombiner(), selector=selector)
    interaction = MarketInteraction(
        consumption_rate=consumption_rate,
        min_reserve=min_reserve,
        nec_threshold=nec_luxury,
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
          f"{'mean_coin':>9} | {'mean_nec':>8} | {'mean_❀':>8} | {'f-trades':>8}")
    print("-" * 90)

    for tick in range(max_ticks):
        env.process_events(tick, rng)
        pairs   = env.do_matching(rng)
        results = env.do_interactions(pairs, rng)

        # Knowledge propagation: only on successful nec trade, seller → buyer
        for (a, b), result in zip(pairs, results):
            if not result.get("trade"):
                continue
            seller_id = result.get("seller_id")
            sender  = a if a.id == seller_id else b
            receiver = b if a.id == seller_id else a
            if receiver._last_farm_pos is None and sender._last_farm_pos is not None:
                receiver._last_farm_pos = sender._last_farm_pos
            if receiver._last_garden_pos is None and sender._last_garden_pos is not None:
                receiver._last_garden_pos = sender._last_garden_pos

        env.do_lifecycle()

        # Update farm memory and decay flowers; update garden memory
        for a in env.agents:
            if not a.alive:
                continue
            # Flower decay: remove flowers older than flower_max_age
            a._flower_ticks = [t for t in a._flower_ticks if tick - t < flower_max_age]
            # Farm memory: record nearest farm center when on farm
            if a.grid_pos in farm_cells:
                r, c = a.grid_pos
                a._last_farm_pos = min(farm_centers, key=lambda fc: (fc[0]-r)**2 + (fc[1]-c)**2)
            # Garden memory: record garden center when on garden
            if a.grid_pos in garden_cells:
                a._last_garden_pos = garden_center

        # Collect stats
        alive = [a for a in env.agents if a.alive]
        trades        = [r for r in results if r.get("trade")]
        flower_trades = [r for r in results if r.get("flower_trade")]
        n_trades      = len(trades)
        n_starving    = sum(1 for a in alive if a.portfolio.get("nec") == 0)
        avg_price     = (sum(r["price"] for r in trades) / n_trades) if trades else float("nan")
        mean_coin     = np.mean([a.portfolio.get("coin") for a in alive]) if alive else 0
        mean_nec      = np.mean([a.portfolio.get("nec")  for a in alive]) if alive else 0
        mean_flowers  = np.mean([a.flower_count for a in alive]) if alive else 0
        total_nec_now = sum(a.portfolio.get("nec") for a in alive)

        price_history.append(avg_price)
        starving_history.append(n_starving)
        trade_history.append(n_trades)

        if (tick + 1) % 100 == 0:
            print(f"{tick+1:>6} | {n_trades:>6} | {avg_price:>9.3f} | {n_starving:>8} | "
                  f"{mean_coin:>9.1f} | {mean_nec:>8.2f} | {mean_flowers:>7.2f}❀ | {len(flower_trades):>4}f-trades")

        if tick % sample_every == 0:
            snap_agents = [
                (a.grid_pos[0], a.grid_pos[1],
                 a.portfolio.get("coin"), a.portfolio.get("nec"), a.flower_count)
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
        _animate_market(snapshots, grid_size, farm_centers, farm_radius,
                        garden_center, garden_radius, eq_price, fps, output)


def _animate_market(snapshots, grid_size, farm_centers, farm_radius,
                    garden_center, garden_radius, eq_price, fps, output):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.patches as mpatches
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    (ax_coin, ax_nec, ax_flower), (ax_dcoin, ax_dnec, ax_dflower) = axes
    fig.tight_layout(pad=3.0)

    def setup_map(ax, title, cmap, vmax):
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect("equal")
        ax.set_facecolor("#111111")
        ax.set_title(title, fontsize=10)
        scat = ax.scatter([], [], s=4, c=[], cmap=cmap, vmin=0, vmax=vmax, alpha=0.9)
        fig.colorbar(scat, ax=ax, fraction=0.03, pad=0.02)
        # Farm circles (gold dashed)
        for cr, cc in farm_centers:
            ax.add_patch(mpatches.Circle(
                (cc, cr), farm_radius, fill=False,
                edgecolor="gold", linewidth=1.2, linestyle="--", alpha=0.8))
        # Garden circle (red solid)
        gcr, gcc = garden_center
        ax.add_patch(mpatches.Circle(
            (gcc, gcr), garden_radius, fill=False,
            edgecolor="red", linewidth=1.5, linestyle="-", alpha=0.9))
        return scat

    scat_coin   = setup_map(ax_coin,   "coin",    "plasma", 500)
    scat_nec    = setup_map(ax_nec,    "nec",     "YlGn",   50)
    scat_flower = setup_map(ax_flower, "❀ flower", "cool",   20)

    def setup_dist(ax, xlabel, color, xmax, bins=40):
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("count", fontsize=9)
        edges = np.linspace(0, xmax, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bars = ax.bar(centers, np.zeros(bins), width=edges[1]-edges[0],
                      color=color, alpha=0.75)
        return bars, edges

    bars_coin,   edges_coin   = setup_dist(ax_dcoin,   "coin",    "mediumpurple", 800)
    bars_nec,    edges_nec    = setup_dist(ax_dnec,    "nec",     "seagreen",     80)
    bars_flower, edges_flower = setup_dist(ax_dflower, "❀ count", "steelblue",    30, bins=30)

    title = fig.suptitle("", fontsize=12)

    def update(fi):
        snap = snapshots[fi]
        tick = snap["tick"]
        agents_data = snap["agents"]
        if not agents_data:
            return

        cols    = np.array([d[1] for d in agents_data])
        rows    = np.array([d[0] for d in agents_data])
        coins   = np.array([d[2] for d in agents_data])
        necs    = np.array([d[3] for d in agents_data])
        flowers = np.array([d[4] for d in agents_data])
        xy = np.column_stack([cols, rows])

        for scat, vals in ((scat_coin, coins), (scat_nec, necs), (scat_flower, flowers)):
            scat.set_offsets(xy)
            scat.set_array(vals)

        for bars, edges, vals in (
            (bars_coin,   edges_coin,   coins),
            (bars_nec,    edges_nec,    necs),
            (bars_flower, edges_flower, flowers),
        ):
            counts, _ = np.histogram(vals, bins=edges)
            for bar, h in zip(bars, counts):
                bar.set_height(h)
            bars[0].axes.set_ylim(0, max(counts.max() * 1.15, 5))

        p = snap.get("avg_price", float("nan"))
        avg_p = f"{p:.3f}" if not math.isnan(p) else "n/a"
        mean_f = flowers.mean() if len(flowers) else 0
        title.set_text(
            f"Tick {tick} | n={len(agents_data)} | price={avg_p} | mean❀={mean_f:.1f}"
        )

    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=1000//fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(output, writer=writer)
    print(f"Saved: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
