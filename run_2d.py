"""Entry point for 2D geography simulation."""

from __future__ import annotations

import argparse
import os
import random

import yaml

from src.agent import Agent, CoinAgent
from src.environment import Environment, BankruptcyWall, GridMovementEvent, MiningEvent
from src.interaction import RandomExchange, YardSaleExchange
from src.matcher import Grid2DSelector, Matcher, WeightedSumCombiner
from src.stats import StatsCollector


def main():
    parser = argparse.ArgumentParser(description="Sugarscape 2D Geography")
    parser.add_argument("--config", default="configs/phase2_geo2d.yaml")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    rng = random.Random(seed)
    Agent.reset_id_counter()

    sim_cfg = config["simulation"]
    max_ticks = sim_cfg["max_ticks"]
    grid_size = sim_cfg["grid_size"]

    agent_cfg = config["agents"]
    n_agents = agent_cfg["count"]
    initial_wealth = agent_cfg["initial_wealth"]

    matcher_cfg = config.get("matcher", {})
    radius = matcher_cfg.get("radius", 5)

    viz_cfg = config.get("viz", {})
    sample_every = viz_cfg.get("sample_every", 5)
    output = viz_cfg.get("output", "output/geo2d.mp4")
    fps = viz_cfg.get("fps", 20)
    x_max = viz_cfg.get("x_max", 600.0)
    bins = viz_cfg.get("bins", 50)

    # Build agents with random grid positions
    smart_fraction = config.get("smart_fraction", 0.0)
    n_smart = int(n_agents * smart_fraction)
    chase_fraction = config.get("chase_fraction", 0.0)
    n_chase = int(n_agents * chase_fraction)
    agents = []
    for i in range(n_agents):
        a = CoinAgent(initial_wealth=initial_wealth, rng=rng)
        a.grid_pos = (rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1))
        agents.append(a)
    # Randomly assign can_refuse to n_smart agents
    smart_indices = rng.sample(range(n_agents), n_smart)
    for i in smart_indices:
        agents[i].can_refuse = True
    if n_smart:
        print(f"  smart agents (can_refuse): {n_smart} ({smart_fraction*100:.0f}%)")
    # Randomly assign chase_wealth to n_chase agents (non-overlapping with smart)
    remaining = [i for i in range(n_agents) if i not in set(smart_indices)]
    chase_indices = rng.sample(remaining, min(n_chase, len(remaining)))
    chase_ids: set[int] = set()
    for i in chase_indices:
        agents[i].chase_wealth = True
        chase_ids.add(agents[i].id)
    if n_chase:
        print(f"  chase agents (wealth-seeking move): {n_chase} ({chase_fraction*100:.0f}%)")

    # Build environment
    selector = Grid2DSelector(grid_size=grid_size, radius=radius)
    matcher = Matcher(factors=[], combiner=WeightedSumCombiner(), selector=selector)
    interaction_cfg = config["interaction"]
    if interaction_cfg["type"] == "yard_sale":
        interaction = YardSaleExchange(
            fraction=interaction_cfg.get("fraction", 0.2),
            poor_advantage=interaction_cfg.get("poor_advantage", 0.0),
        )
    else:
        interaction = RandomExchange(amount=interaction_cfg.get("amount", 1.0))
    mining_cfg = config.get("mining", None)
    mining_center = None
    mining_radius = None

    events = []
    if config.get("movement", True):
        events.append(GridMovementEvent(grid_size=grid_size))
    if config.get("absorbing_wall", False):
        events.append(BankruptcyWall())
    if mining_cfg:
        mining_center = tuple(mining_cfg["center"])
        mining_radius = mining_cfg["radius"]
        mining_income = mining_cfg.get("income", 1.0)
        events.append(MiningEvent(center=mining_center, radius=mining_radius, income=mining_income))
        print(f"  mining zone: center={mining_center}, R={mining_radius}, income={mining_income}/tick")
    env = Environment(matcher=matcher, interaction=interaction, events=events)
    env.add_agents(agents)

    stats = StatsCollector()
    # Each snapshot: {"tick": int, "agents": [(r,c,wealth),...], "pairs": [(r1,c1,r2,c2),...]}
    position_snapshots: list[dict] = []

    print(f"Running {max_ticks} ticks | {n_agents} agents | grid {grid_size}×{grid_size} | R={radius}")

    for tick in range(max_ticks):
        env.process_events(tick, rng)
        pairs = env.do_matching(rng)
        results = env.do_interactions(pairs, rng)
        # Update partner memory for chase agents
        if n_chase:
            for a, b in pairs:
                if a.chase_wealth and b.grid_pos is not None:
                    a._last_partner_pos = b.grid_pos
                    a._last_partner_wealth = b.wealth()
                if b.chase_wealth and a.grid_pos is not None:
                    b._last_partner_pos = a.grid_pos
                    b._last_partner_wealth = a.wealth()
        env.do_lifecycle()
        stats.record_tick(tick, env.agents, results)

        if tick % sample_every == 0:
            snap = [
                (a.grid_pos[0], a.grid_pos[1], a.wealth())
                for a in env.agents
                if a.alive and a.grid_pos is not None
            ]
            pair_lines = [
                (a.grid_pos[0], a.grid_pos[1], b.grid_pos[0], b.grid_pos[1])
                for a, b in pairs
                if a.grid_pos is not None and b.grid_pos is not None
            ]
            position_snapshots.append({"tick": tick, "agents": snap, "pairs": pair_lines})

        if (tick + 1) % 100 == 0:
            h = stats.history[-1]
            print(f"  tick {tick+1:4d} | agents={h['num_agents']} "
                  f"| pairs={h['num_interactions']} | mean={h['mean_wealth']:.1f} "
                  f"| gini={h['gini']:.3f}")

    # Final summary
    final = stats.history[-1]
    print(f"\nFinal: agents={final['num_agents']} | mean={final['mean_wealth']:.1f} "
          f"| gini={final['gini']:.3f} | snapshots={len(position_snapshots)}")

    # Chase vs normal breakdown
    if n_chase:
        import numpy as np
        from scipy import stats as scipy_stats
        alive = [a for a in env.agents if a.alive]
        chase_w = np.array([a.wealth() for a in alive if a.id in chase_ids])
        normal_w = np.array([a.wealth() for a in alive if a.id not in chase_ids])
        print(f"\n=== Chase vs Normal (final tick) ===")
        for label, arr in [("Chase", chase_w), ("Normal", normal_w)]:
            print(f"  {label}: n={len(arr)}, mean={arr.mean():.1f}, "
                  f"median={np.median(arr):.1f}, max={arr.max():.1f}, cv={arr.std()/arr.mean():.2f}")
        stat, p = scipy_stats.mannwhitneyu(chase_w, normal_w, alternative="two-sided")
        print(f"  Mann-Whitney U: U={stat:.0f}, p={p:.4f} ({'significant' if p < 0.05 else 'not significant'} at α=0.05)")

    if not args.no_viz:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        from viz.animate_2d import animate_2d
        animate_2d(
            snapshots=position_snapshots,
            grid_size=grid_size,
            x_max=x_max,
            bins=bins,
            fps=fps,
            output=output,
            mining_center=mining_center,
            mining_radius=mining_radius,
        )


if __name__ == "__main__":
    main()
