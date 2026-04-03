"""Matcher module: Factor → Combiner → Selector pipeline for agent pairing."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from .agent import Agent


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class Factor(ABC):
    """Scores a potential pairing. May be asymmetric: score(a,b) != score(b,a)."""

    @abstractmethod
    def score(self, agent_a: Agent, agent_b: Agent) -> float: ...


class Combiner(ABC):
    """Combines multiple factor scores into a single score."""

    @abstractmethod
    def combine(self, scores: list[float]) -> float: ...


class Selector(ABC):
    """Produces matched pairs from the agent pool using factor scores."""

    @abstractmethod
    def select(
        self,
        agents: list[Agent],
        factors: list[Factor],
        combiner: Combiner,
        rng: random.Random,
    ) -> list[tuple[Agent, Agent]]: ...


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

class WeightedSumCombiner(Combiner):
    def __init__(self, weights: list[float] | None = None):
        self.weights = weights or []

    def combine(self, scores: list[float]) -> float:
        return sum(w * s for w, s in zip(self.weights, scores))


class RandomSelector(Selector):
    """Phase 1: shuffle and pair adjacent agents. Ignores factors/combiner."""

    def select(
        self,
        agents: list[Agent],
        factors: list[Factor],
        combiner: Combiner,
        rng: random.Random,
    ) -> list[tuple[Agent, Agent]]:
        pool = [a for a in agents if a.alive]
        rng.shuffle(pool)
        return [(pool[i], pool[i + 1]) for i in range(0, len(pool) - 1, 2)]


# ---------------------------------------------------------------------------
# Matcher: composition container
# ---------------------------------------------------------------------------

class Matcher:
    """Composes Factor(s) + Combiner + Selector into a single match() call."""

    def __init__(
        self,
        factors: list[Factor] | None = None,
        combiner: Combiner | None = None,
        selector: Selector | None = None,
    ):
        self.factors: list[Factor] = factors or []
        self.combiner: Combiner = combiner or WeightedSumCombiner()
        self.selector: Selector = selector or RandomSelector()

    def match(self, agents: list[Agent], rng: random.Random) -> list[tuple[Agent, Agent]]:
        return self.selector.select(agents, self.factors, self.combiner, rng)
