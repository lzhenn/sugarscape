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
            if a.alive and a.grid_pos is not None:
                r, c = a.grid_pos
                dr, dc = rng.choice(self._DIRECTIONS)
                a.grid_pos = ((r + dr) % G, (c + dc) % G)


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
