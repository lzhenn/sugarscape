"""Matcher module: Factor → Combiner → Selector pipeline for agent pairing."""

from __future__ import annotations

import math
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


class GeographyFactor(Factor):
    """Score based on 1D ring distance with exponential decay.

    Agents live on a ring of circumference `ring_size`. Distance is the
    shorter arc. Score = exp(-distance / sigma).
    """

    def __init__(self, ring_size: float = 1.0, sigma: float = 0.1):
        self.ring_size = ring_size
        self.sigma = sigma

    def score(self, agent_a: Agent, agent_b: Agent) -> float:
        if agent_a.position is None or agent_b.position is None:
            return 1.0
        d = abs(agent_a.position - agent_b.position)
        d = min(d, self.ring_size - d)  # shorter arc on the ring
        return math.exp(-d / self.sigma)


class ProbabilisticSelector(Selector):
    """Select pairs probabilistically based on factor scores.

    For efficiency with GeographyFactor, uses a local window (±3σ) when
    agents have positions. Falls back to full scan otherwise.
    """

    def __init__(self, window: float | None = None):
        self.window = window  # If set, only consider candidates within this distance

    def select(
        self,
        agents: list[Agent],
        factors: list[Factor],
        combiner: Combiner,
        rng: random.Random,
    ) -> list[tuple[Agent, Agent]]:
        pool = [a for a in agents if a.alive]
        rng.shuffle(pool)

        # If agents have positions, sort by position for efficient windowing
        has_geo = pool and pool[0].position is not None
        if has_geo:
            pool.sort(key=lambda a: a.position)
            pos_idx = {a.id: i for i, a in enumerate(pool)}

        matched: set[int] = set()
        pairs: list[tuple[Agent, Agent]] = []
        order = list(range(len(pool)))
        rng.shuffle(order)

        for i in order:
            a = pool[i]
            if a.id in matched:
                continue

            # Find candidates
            if has_geo and self.window is not None:
                # Only look within window on the ring
                candidates = []
                n = len(pool)
                for offset in range(1, n):
                    for direction in [1, -1]:
                        j = (i + direction * offset) % n
                        b = pool[j]
                        if b.id in matched:
                            continue
                        # Ring distance
                        d = abs(a.position - b.position)
                        ring = 1.0  # assume ring_size=1
                        d = min(d, ring - d)
                        if d > self.window:
                            continue
                        candidates.append(b)
                    if len(candidates) >= 20:  # cap candidates for speed
                        break
            else:
                candidates = [b for b in pool if b.id != a.id and b.id not in matched]

            if not candidates:
                continue

            if factors:
                weights = []
                for b in candidates:
                    scores = [f.score(a, b) for f in factors]
                    weights.append(max(0.0, combiner.combine(scores)))
            else:
                weights = [1.0] * len(candidates)

            total = sum(weights)
            if total <= 0:
                partner = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                r = rng.random() * total
                cumsum = 0.0
                partner = candidates[-1]
                for b, w in zip(candidates, weights):
                    cumsum += w
                    if cumsum >= r:
                        partner = b
                        break

            matched.add(a.id)
            matched.add(partner.id)
            pairs.append((a, partner))

        return pairs


class Grid2DSelector(Selector):
    """Match agents within radius R on a 2D torus grid.

    Agents must have a grid_pos attribute set. For each unmatched agent,
    collects all unmatched agents in the (2R+1)×(2R+1) square window
    (torus-wrapped), then picks one uniformly at random.
    """

    def __init__(self, grid_size: int = 200, radius: int = 5):
        self.grid_size = grid_size
        self.radius = radius

    def select(
        self,
        agents: list[Agent],
        factors: list[Factor],
        combiner: Combiner,
        rng: random.Random,
    ) -> list[tuple[Agent, Agent]]:
        pool = [a for a in agents if a.alive and a.grid_pos is not None]
        rng.shuffle(pool)

        # Build cell lookup: (row, col) → list of agents
        cell_map: dict[tuple[int, int], list[Agent]] = {}
        for a in pool:
            key = a.grid_pos
            if key not in cell_map:
                cell_map[key] = []
            cell_map[key].append(a)

        matched: set[int] = set()
        pairs: list[tuple[Agent, Agent]] = []
        G = self.grid_size
        R = self.radius

        for a in pool:
            if a.id in matched:
                continue
            r, c = a.grid_pos
            candidates = []
            for dr in range(-R, R + 1):
                for dc in range(-R, R + 1):
                    nr, nc = (r + dr) % G, (c + dc) % G
                    for b in cell_map.get((nr, nc), []):
                        if b.id != a.id and b.id not in matched:
                            candidates.append(b)

            if not candidates:
                continue

            partner = rng.choice(candidates)
            matched.add(a.id)
            matched.add(partner.id)
            pairs.append((a, partner))

        return pairs


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
