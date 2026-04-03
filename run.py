"""Entry point for Sugarscape ABM simulation."""

import argparse

import yaml

from src.simulation import Simulation
from viz.animate import animate_wealth_distribution


def main():
    parser = argparse.ArgumentParser(description="Sugarscape ABM")
    parser.add_argument(
        "--config", type=str, default="configs/phase1.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization",
    )
    args = parser.parse_args()

    sim = Simulation.from_yaml(args.config)
    print(
        f"Running: {sim.max_ticks} ticks, "
        f"{len(sim.env.agents)} agents, seed={sim.config.get('seed')}"
    )

    stats = sim.run()

    final = stats.history[-1]
    print(
        f"Done. Tick {final['tick']}: "
        f"mean={final['mean_wealth']:.1f}, "
        f"gini={final['gini']:.3f}, "
        f"min={final['min_wealth']:.0f}, "
        f"max={final['max_wealth']:.0f}"
    )

    if not args.no_viz:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        animate_wealth_distribution(stats, config)


if __name__ == "__main__":
    main()
