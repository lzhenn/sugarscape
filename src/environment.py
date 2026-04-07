"""Environment module: world container holding agents, matcher, interaction, and events."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from .agent import Agent
from .interaction import Interaction
from .matcher import Matcher


class Event(ABC):
    """External force that acts on the environment (government, disaster, etc.)."""

    @abstractmethod
    def should_trigger(self, tick: int, env: Environment) -> bool: ...

    @abstractmethod
    def execute(self, env: Environment, rng: random.Random) -> None: ...


class BankruptcyWall(Event):
    """Kill any agent whose wealth drops to zero or below."""

    def should_trigger(self, tick: int, env: Environment) -> bool:
        return True

    def execute(self, env: Environment, rng: random.Random) -> None:
        for a in env.agents:
            if a.alive and a.wealth() <= 0:
                a.alive = False


class GridMovementEvent(Event):
    """Move each agent one step on a 2D torus grid each tick.

    Each alive agent with a grid_pos picks one of 9 directions uniformly
    at random (8 compass directions + stay) and moves accordingly.
    """

    _DIRECTIONS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]

    def __init__(self, grid_size: int = 200):
        self.grid_size = grid_size

    def should_trigger(self, tick: int, env: Environment) -> bool:
        return True

    def execute(self, env: Environment, rng: random.Random) -> None:
        G = self.grid_size
        for a in env.agents:
            if not a.alive or a.grid_pos is None:
                continue
            r, c = a.grid_pos
            if a.chase_wealth and a._last_partner_pos is not None and a._last_partner_wealth is not None:
                pr, pc = a._last_partner_pos
                # Sign of direction vector toward last partner (no torus correction, cheapest)
                dr = (pr > r) - (pr < r)
                dc = (pc > c) - (pc < c)
                if a._last_partner_wealth == a.wealth() or (dr == 0 and dc == 0):
                    dr, dc = rng.choice(self._DIRECTIONS)  # tied wealth or same cell → random
                elif a._last_partner_wealth < a.wealth():
                    dr, dc = -dr, -dc  # partner poorer → move away
                # else partner richer → move toward (dr, dc already correct)
            else:
                dr, dc = rng.choice(self._DIRECTIONS)
            a.grid_pos = ((r + dr) % G, (c + dc) % G)


class NecessityLifecycleEvent(Event):
    """Atomically handle necessity consumption + farm production each tick.

    Execution order within this single event:
      1. Each agent consumes exactly min(nec, rate) — total consumed tracked.
      2. The total consumed is redistributed equally to agents on farm patches.

    This guarantees exact global conservation: Σ nec is invariant every tick.
    Agents whose nec reaches 0 are 'starving' and become forced buyers next phase.
    """

    def __init__(
        self,
        farm_centers: list[tuple[int, int]],
        radius: int,
        grid_size: int,
        consumption_rate: float = 1.0,
    ):
        self.consumption_rate = consumption_rate
        radius_sq = radius * radius
        self._farm_cells: set[tuple[int, int]] = {
            (r, c)
            for cr, cc in farm_centers
            for r in range(grid_size)
            for c in range(grid_size)
            if (r - cr) ** 2 + (c - cc) ** 2 <= radius_sq
        }

    def should_trigger(self, tick: int, env: "Environment") -> bool:
        return True

    def execute(self, env: "Environment", rng: random.Random) -> None:
        alive = [a for a in env.agents if a.alive]
        if not alive:
            return

        # Step 1: consume — track exactly what each agent loses
        total_consumed = 0.0
        for a in alive:
            current = a.portfolio.get("nec")
            consumed = min(current, self.consumption_rate)
            a.portfolio.set("nec", current - consumed)
            total_consumed += consumed

        # Step 2: redistribute exactly total_consumed to farm agents
        if total_consumed == 0.0:
            return
        on_farm = [
            a for a in alive
            if a.grid_pos is not None and a.grid_pos in self._farm_cells
        ]
        if not on_farm:
            return
        share = total_consumed / len(on_farm)
        for a in on_farm:
            a.portfolio.add("nec", share)


class StarvationWall(Event):
    """Kill agents whose necessity hits 0; redistribute their coin to survivors.

    nec of dying agents is already 0, so total nec is conserved automatically.
    Coin is redistributed equally to keep total coin conserved.
    """

    def should_trigger(self, tick: int, env: "Environment") -> bool:
        return True

    def execute(self, env: "Environment", rng: random.Random) -> None:
        dying    = [a for a in env.agents if a.alive and a.portfolio.get("nec") <= 0]
        survivors = [a for a in env.agents if a.alive and a.portfolio.get("nec") > 0]
        if not dying:
            return
        # Collect and redistribute coin
        orphan_coin = sum(a.portfolio.get("coin") for a in dying)
        for a in dying:
            a.alive = False
        if survivors and orphan_coin > 0:
            share = orphan_coin / len(survivors)
            for a in survivors:
                a.portfolio.add("coin", share)


class MiningEvent(Event):
    """Grant +income wealth per tick to agents standing inside a circular mining zone.

    Distance is plain Euclidean (not torus-wrapped) from the center cell.
    Multiple agents on the same cell each receive the full income (inflation).
    """

    def __init__(
        self,
        center: tuple[int, int],
        radius: int,
        income: float = 1.0,
    ):
        self.center = center
        self.radius = radius
        self.income = income
        self._radius_sq = radius * radius

    def should_trigger(self, tick: int, env: "Environment") -> bool:
        return True

    def execute(self, env: "Environment", rng: random.Random) -> None:
        cr, cc = self.center
        for a in env.agents:
            if a.alive and a.grid_pos is not None:
                r, c = a.grid_pos
                if (r - cr) ** 2 + (c - cc) ** 2 <= self._radius_sq:
                    a.portfolio.add("coin", self.income)


class Environment:
    """Holds all agents and orchestrates matching, interaction, and lifecycle."""

    def __init__(
        self,
        matcher: Matcher,
        interaction: Interaction,
        events: list[Event] | None = None,
    ):
        self.agents: list[Agent] = []
        self.matcher = matcher
        self.interaction = interaction
        self.events: list[Event] = events or []

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def add_agents(self, agents: list[Agent]) -> None:
        self.agents.extend(agents)

    def remove_dead(self) -> list[Agent]:
        dead = [a for a in self.agents if not a.alive]
        self.agents = [a for a in self.agents if a.alive]
        return dead

    def process_events(self, tick: int, rng: random.Random) -> None:
        for event in self.events:
            if event.should_trigger(tick, self):
                event.execute(self, rng)

    def do_matching(self, rng: random.Random) -> list[tuple[Agent, Agent]]:
        return self.matcher.match(self.agents, rng)

    def do_interactions(
        self, pairs: list[tuple[Agent, Agent]], rng: random.Random
    ) -> list[dict]:
        results = []
        for a, b in pairs:
            result = self.interaction.interact(a, b, rng)
            results.append(result)
        return results

    def do_lifecycle(self) -> None:
        for agent in self.agents:
            agent.step()
        self.remove_dead()
