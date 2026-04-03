"""Interaction module: defines how paired agents interact."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from .agent import Agent


class Interaction(ABC):
    """Base class for all interaction rules."""

    @abstractmethod
    def interact(self, agent_a: Agent, agent_b: Agent, rng: random.Random) -> dict:
        """Execute interaction between a matched pair.

        Returns a dict of outcome metadata for stats collection.
        """


class RandomExchange(Interaction):
    """Phase 1: fair coin flip, winner takes a fixed amount from loser."""

    def __init__(self, amount: float = 1.0):
        self.amount = amount

    def interact(self, agent_a: Agent, agent_b: Agent, rng: random.Random) -> dict:
        if rng.random() < 0.5:
            winner, loser = agent_a, agent_b
        else:
            winner, loser = agent_b, agent_a

        actual = min(self.amount, loser.portfolio.get("coin"))
        if actual > 0:
            loser.portfolio.remove("coin", actual)
            winner.portfolio.add("coin", actual)

        return {"winner_id": winner.id, "loser_id": loser.id, "amount": actual}
