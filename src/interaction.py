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


class YardSaleExchange(Interaction):
    """Yard-sale (multiplicative) exchange: stake = fraction × min(w_A, w_B).

    The poorer agent can never lose more than they have, so wealth stays
    non-negative without an explicit absorbing wall. With a wall, agents
    that hit exactly 0 are eliminated.
    """

    def __init__(self, fraction: float = 0.2, poor_advantage: float = 0.0):
        self.fraction = fraction
        self.poor_advantage = poor_advantage  # extra win prob for the poorer agent

    def interact(self, agent_a: Agent, agent_b: Agent, rng: random.Random) -> dict:
        wa = agent_a.portfolio.get("coin")
        wb = agent_b.portfolio.get("coin")
        stake = self.fraction * min(wa, wb)

        # Determine win probability for agent_a
        if self.poor_advantage > 0 and wa != wb:
            p_a = (0.5 - self.poor_advantage) if wa > wb else (0.5 + self.poor_advantage)
        else:
            p_a = 0.5

        if rng.random() < p_a:
            winner, loser = agent_a, agent_b
        else:
            winner, loser = agent_b, agent_a

        if stake > 0:
            loser.portfolio.remove("coin", stake)
            winner.portfolio.add("coin", stake)

        return {"winner_id": winner.id, "loser_id": loser.id, "amount": stake}


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
