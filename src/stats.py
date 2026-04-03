"""Statistics collection and recording."""

from __future__ import annotations

from .agent import Agent


class StatsCollector:
    """Records per-tick snapshots of simulation state."""

    def __init__(self):
        self.history: list[dict] = []

    def record_tick(
        self, tick: int, agents: list[Agent], interaction_results: list[dict]
    ) -> None:
        wealths = [a.wealth() for a in agents if a.alive]
        n = len(wealths)
        entry = {
            "tick": tick,
            "num_agents": n,
            "wealths": list(wealths),
            "mean_wealth": sum(wealths) / n if n else 0.0,
            "min_wealth": min(wealths) if n else 0.0,
            "max_wealth": max(wealths) if n else 0.0,
            "gini": self._gini(wealths),
            "num_interactions": len(interaction_results),
        }
        self.history.append(entry)

    @staticmethod
    def _gini(values: list[float]) -> float:
        """Gini coefficient. 0 = perfect equality, 1 = maximum inequality."""
        if not values or sum(values) == 0:
            return 0.0
        sorted_v = sorted(values)
        n = len(sorted_v)
        weighted_sum = sum((i + 1) * v for i, v in enumerate(sorted_v))
        total = sum(sorted_v)
        return (2.0 * weighted_sum) / (n * total) - (n + 1.0) / n

    def get_wealth_snapshots(
        self, every_n: int = 1
    ) -> list[tuple[int, list[float]]]:
        """Return (tick, wealths) tuples, sampled every N ticks."""
        return [
            (h["tick"], h["wealths"])
            for h in self.history
            if h["tick"] % every_n == 0
        ]
