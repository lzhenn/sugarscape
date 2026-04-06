"""Entry point for 2D geography simulation."""

from __future__ import annotations

import argparse
import os
import random

import yaml

from src.agent import Agent, CoinAgent
from src.environment import Environment, GridMovementEvent
from src.interaction import RandomExchange
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
    agents = []
    for _ in range(n_agents):
        a = CoinAgent(initial_wealth=initial_wealth, rng=rng)
        a.grid_pos = (rng.randint(0, grid_size - 1), rng.randint(0, grid_size - 1))
        agents.append(a)

    # Build environment
    selector = Grid2DSelector(grid_size=grid_size, radius=radius)
    matcher = Matcher(factors=[], combiner=WeightedSumCombiner(), selector=selector)
    interaction = RandomExchange(amount=config["interaction"]["amount"])
    movement = GridMovementEvent(grid_size=grid_size)
    env = Environment(matcher=matcher, interaction=interaction, events=[movement])
    env.add_agents(agents)

    stats = StatsCollector()
    # Each snapshot: {"tick": int, "agents": [(r,c,wealth),...], "pairs": [(r1,c1,r2,c2),...]}
    position_snapshots: list[dict] = []

    print(f"Running {max_ticks} ticks | {n_agents} agents | grid {grid_size}×{grid_size} | R={radius}")

    for tick in range(max_ticks):
        env.process_events(tick, rng)
        pairs = env.do_matching(rng)
        results = env.do_interactions(pairs, rng)
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
        )


if __name__ == "__main__":
    main()
