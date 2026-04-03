"""Simulation engine: builds and runs the model from config."""

from __future__ import annotations

import random

import yaml

from .agent import Agent, CoinAgent
from .environment import Environment
from .interaction import RandomExchange
from .matcher import Matcher, RandomSelector, WeightedSumCombiner
from .stats import StatsCollector


class Simulation:
    """Top-level controller: owns the RNG, Environment, and StatsCollector."""

    def __init__(self, config: dict):
        self.config = config
        self.rng = random.Random(config.get("seed", 42))
        self.tick = 0
        self.max_ticks: int = config["simulation"]["max_ticks"]
        self.stats = StatsCollector()

        Agent.reset_id_counter()
        self.env = self._build_environment(config)
        self._populate_agents(config)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    def _build_environment(self, config: dict) -> Environment:
        selector = RandomSelector()
        combiner = WeightedSumCombiner()
        matcher = Matcher(factors=[], combiner=combiner, selector=selector)

        interaction_cfg = config.get("interaction", {})
        interaction = RandomExchange(amount=interaction_cfg.get("amount", 1.0))

        return Environment(matcher=matcher, interaction=interaction)

    def _populate_agents(self, config: dict) -> None:
        agent_cfg = config["agents"]
        n = agent_cfg["count"]
        initial_wealth = agent_cfg["initial_wealth"]
        agents = [CoinAgent(initial_wealth=initial_wealth, rng=self.rng) for _ in range(n)]
        self.env.add_agents(agents)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Execute one tick: Events → Match → Interact → Lifecycle → Stats."""
        self.env.process_events(self.tick, self.rng)
        pairs = self.env.do_matching(self.rng)
        results = self.env.do_interactions(pairs, self.rng)
        self.env.do_lifecycle()
        self.stats.record_tick(self.tick, self.env.agents, results)
        self.tick += 1

    def run(self) -> StatsCollector:
        for _ in range(self.max_ticks):
            self.step()
        return self.stats

    @classmethod
    def from_yaml(cls, path: str) -> Simulation:
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)
