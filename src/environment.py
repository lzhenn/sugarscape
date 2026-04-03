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
