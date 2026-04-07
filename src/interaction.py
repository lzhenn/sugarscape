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


class NecessityTradeInteraction(Interaction):
    """Bilateral necessity ↔ coin trade via shadow price midpoint.

    Shadow price for agent i  =  coin_i / nec_i  (coin per unit of necessity).
    The agent with the higher shadow price is the buyer (coin-rich, nec-poor).
    Trade price P = midpoint of both shadow prices.
    Trade quantity Q = min(seller_surplus, buyer_can_afford).

    Starving buyer (nec == 0): accepts any price; price = seller's shadow price.
    Seller protects a minimum reserve of `min_reserve` units before selling.
    If seller has no surplus, the interaction is skipped.
    """

    def __init__(self, consumption_rate: float = 1.0, min_reserve: float = 1.0):
        self.consumption_rate = consumption_rate
        self.min_reserve = min_reserve

    def interact(self, agent_a: Agent, agent_b: Agent, rng: random.Random) -> dict:
        coin_a = agent_a.portfolio.get("coin")
        nec_a  = agent_a.portfolio.get("nec")
        coin_b = agent_b.portfolio.get("coin")
        nec_b  = agent_b.portfolio.get("nec")

        # Shadow price = coin / nec; 0-nec agents are forced buyers (inf shadow price)
        shadow_a = coin_a / nec_a if nec_a > 0 else float("inf")
        shadow_b = coin_b / nec_b if nec_b > 0 else float("inf")

        if shadow_a == shadow_b:
            return {"trade": False, "price": 0.0, "quantity": 0.0}

        if shadow_a > shadow_b:
            buyer, seller = agent_a, agent_b
            shadow_buyer, shadow_seller = shadow_a, shadow_b
            coin_buyer = coin_a
            nec_seller = nec_b
        else:
            buyer, seller = agent_b, agent_a
            shadow_buyer, shadow_seller = shadow_b, shadow_a
            coin_buyer = coin_b
            nec_seller = nec_a

        # Seller must keep at least min_reserve units
        seller_surplus = nec_seller - self.min_reserve
        if seller_surplus <= 0:
            return {"trade": False, "price": 0.0, "quantity": 0.0}

        # Price: midpoint; starving buyer accepts seller's full shadow price
        if shadow_buyer == float("inf"):
            price = shadow_seller
        else:
            price = (shadow_buyer + shadow_seller) / 2.0

        if price <= 0:
            return {"trade": False, "price": 0.0, "quantity": 0.0}

        # Quantity: limited by seller surplus and buyer's coin
        buyer_can_afford = coin_buyer / price
        quantity = min(seller_surplus, buyer_can_afford)
        if quantity <= 0:
            return {"trade": False, "price": 0.0, "quantity": 0.0}

        # Cap coin_paid to what buyer actually has (floating-point safety)
        coin_paid = min(quantity * price, coin_buyer)
        quantity  = coin_paid / price  # recalculate to keep trade balanced

        seller.portfolio.remove("nec", quantity)
        buyer.portfolio.add("nec", quantity)
        buyer.portfolio.remove("coin", coin_paid)
        seller.portfolio.add("coin", coin_paid)

        return {
            "trade": True,
            "price": price,
            "quantity": quantity,
            "coin": coin_paid,
            "buyer_id": buyer.id,
            "seller_id": seller.id,
        }


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
