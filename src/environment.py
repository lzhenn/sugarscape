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

    def __init__(self, grid_size: int = 200, farm_cells: set | None = None,
                 garden_cells: set | None = None,
                 nec_luxury: float = 20.0, consumption_rate: float = 0.1):
        self.grid_size = grid_size
        self._farm_cells: set = farm_cells or set()
        self._garden_cells: set = garden_cells or set()
        self._nec_luxury = nec_luxury
        self._consumption_rate = consumption_rate

    def should_trigger(self, tick: int, env: Environment) -> bool:
        return True

    def execute(self, env: Environment, rng: random.Random) -> None:
        G = self.grid_size
        occupied = {a.grid_pos for a in env.agents if a.alive and a.grid_pos is not None}
        order = [a for a in env.agents if a.alive and a.grid_pos is not None]
        rng.shuffle(order)
        for a in order:
            r, c = a.grid_pos
            nec = a.portfolio.get("nec")
            # Priority 1: knows farm + off farm + nec barely enough to return → head back
            if a._last_farm_pos is not None and (r, c) not in self._farm_cells:
                fr, fc = a._last_farm_pos
                steps_to_farm = max(abs(fr - r), abs(fc - c))  # Chebyshev distance
                nec_needed = steps_to_farm * self._consumption_rate
                if nec <= nec_needed:
                    dr = (fr > r) - (fr < r)
                    dc = (fc > c) - (fc < c)
                    if dr == 0 and dc == 0:
                        dr, dc = rng.choice(self._DIRECTIONS)
                else:
                    dr = None  # nec sufficient → fall through to lower priorities
            else:
                dr = None

            if dr is None:
                # Priority 2: nec above luxury threshold + knows garden → seek flowers
                if (nec > self._nec_luxury
                        and a._last_garden_pos is not None
                        and (r, c) not in self._garden_cells):
                    gr, gc = a._last_garden_pos
                    dr = (gr > r) - (gr < r)
                    dc = (gc > c) - (gc < c)
                    if dr == 0 and dc == 0:
                        dr, dc = rng.choice(self._DIRECTIONS)
                else:
                    dr, dc = rng.choice(self._DIRECTIONS)
            new_pos = ((r + dr) % G, (c + dc) % G)
            if new_pos not in occupied or new_pos == (r, c):
                occupied.discard((r, c))
                occupied.add(new_pos)
                a.grid_pos = new_pos
            # else: cell occupied → stay


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
        production_rate: float = 1.0,
    ):
        self.consumption_rate = consumption_rate
        self.production_rate = production_rate
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

        # Step 1: all agents consume nec
        for a in alive:
            current = a.portfolio.get("nec")
            a.portfolio.set("nec", max(0.0, current - self.consumption_rate))

        # Step 2: farm agents gain production_rate nec (not conserved)
        for a in alive:
            if a.grid_pos is not None and a.grid_pos in self._farm_cells:
                a.portfolio.add("nec", self.production_rate)


class GardenEvent(Event):
    """Each tick, agents standing on garden cells receive one flower (timestamped)."""

    def __init__(self, garden_cells: set):
        self._garden_cells = garden_cells

    def should_trigger(self, tick: int, env: "Environment") -> bool:
        return True

    def execute(self, env: "Environment", rng: random.Random) -> None:
        tick = env.current_tick
        for a in env.agents:
            if a.alive and a.grid_pos in self._garden_cells:
                a._flower_ticks.append(tick)


class NecRedistributionEvent(Event):
    """Tax top nec holders every interval ticks; redistribute after a delay.

    Collected nec is held in a buffer and distributed delay_ticks later
    to the bottom nec holders at that future tick.
    """

    def __init__(self, start_tick: int = 0, interval: int = 100,
                 delay_ticks: int = 50,
                 top_fraction: float = 0.01, tax_rate: float = 0.1,
                 bottom_fraction: float = 0.1):
        self.start_tick = start_tick
        self.interval = interval
        self.delay_ticks = delay_ticks
        self.top_fraction = top_fraction
        self.tax_rate = tax_rate
        self.bottom_fraction = bottom_fraction
        # {release_tick: nec_amount}
        self._buffer: dict[int, float] = {}

    def should_trigger(self, tick: int, env: "Environment") -> bool:
        return tick >= self.start_tick

    def execute(self, env: "Environment", rng: random.Random) -> None:
        tick = env.current_tick
        alive = [a for a in env.agents if a.alive]
        if not alive:
            return

        # Step 1: collect tax every interval ticks, store for later release
        if (tick - self.start_tick) % self.interval == 0:
            n_top = max(1, int(len(alive) * self.top_fraction))
            top = sorted(alive, key=lambda a: a.portfolio.get("nec"), reverse=True)[:n_top]
            total_tax = 0.0
            for a in top:
                tax = a.portfolio.get("nec") * self.tax_rate
                a.portfolio.remove("nec", tax)
                total_tax += tax
            if total_tax > 0:
                release_tick = tick + self.delay_ticks
                self._buffer[release_tick] = self._buffer.get(release_tick, 0.0) + total_tax

        # Step 2: release any buffered nec due this tick
        if tick in self._buffer:
            amount = self._buffer.pop(tick)
            n_bottom = max(1, int(len(alive) * self.bottom_fraction))
            bottom = sorted(alive, key=lambda a: a.portfolio.get("nec"))[:n_bottom]
            share = amount / len(bottom)
            for a in bottom:
                a.portfolio.add("nec", share)


class CoinRedistributionEvent(Event):
    """Every interval ticks after start_tick: tax top coin holders, give coin to bottom nec holders.

    Tax: top_fraction of agents by coin pay tax_rate of their coin.
    Recipients: bottom_fraction of agents by nec receive equal coin shares.
    Conserves total coin exactly.
    """

    def __init__(self, start_tick: int = 200, interval: int = 100,
                 top_fraction: float = 0.01, tax_rate: float = 0.1,
                 bottom_fraction: float = 0.5):
        self.start_tick = start_tick
        self.interval = interval
        self.top_fraction = top_fraction
        self.tax_rate = tax_rate
        self.bottom_fraction = bottom_fraction

    def should_trigger(self, tick: int, env: "Environment") -> bool:
        return tick >= self.start_tick and (tick - self.start_tick) % self.interval == 0

    def execute(self, env: "Environment", rng: random.Random) -> None:
        alive = [a for a in env.agents if a.alive]
        if not alive:
            return
        n = len(alive)
        n_top    = max(1, int(n * self.top_fraction))
        n_bottom = max(1, int(n * self.bottom_fraction))

        top    = sorted(alive, key=lambda a: a.portfolio.get("coin"), reverse=True)[:n_top]
        bottom = sorted(alive, key=lambda a: a.portfolio.get("nec"))[:n_bottom]

        total_tax = 0.0
        for a in top:
            tax = a.portfolio.get("coin") * self.tax_rate
            a.portfolio.remove("coin", tax)
            total_tax += tax

        if total_tax > 0 and bottom:
            share = total_tax / len(bottom)
            for a in bottom:
                a.portfolio.add("coin", share)


class StarvationWall(Event):
    """Kill agents whose necessity hits 0; redistribute their coin to survivors.

    nec of dying agents is already 0, so total nec is conserved automatically.
    Coin is redistributed equally to keep total coin conserved.
    """

    def should_trigger(self, tick: int, env: "Environment") -> bool:
        return True

    def execute(self, env: "Environment", rng: random.Random) -> None:
        for a in env.agents:
            if a.alive and a.portfolio.get("nec") <= 0:
                # All assets buried with the agent
                a.portfolio.set("coin", 0.0)
                a.portfolio.set("nec", 0.0)
                a._flower_ticks.clear()
                a.alive = False


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
        self.current_tick: int = 0

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def add_agents(self, agents: list[Agent]) -> None:
        self.agents.extend(agents)

    def remove_dead(self) -> list[Agent]:
        dead = [a for a in self.agents if not a.alive]
        self.agents = [a for a in self.agents if a.alive]
        return dead

    def process_events(self, tick: int, rng: random.Random) -> None:
        self.current_tick = tick
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
