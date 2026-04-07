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


class MarketInteraction(Interaction):
    """Combined nec↔coin and flower↔coin bilateral trade.

    Nec trade: shadow price midpoint (same as NecessityTradeInteraction).
    Flower trade: only when buyer has nec > nec_threshold (luxury condition).
      Shadow price for flowers = coin / flower_count.
      Trade quantity = 1 flower per interaction.
    """

    def __init__(self, consumption_rate: float = 1.0, min_reserve: float = 1.0,
                 nec_threshold: float = 0.0):
        self.consumption_rate = consumption_rate
        self.min_reserve = min_reserve
        self.nec_threshold = nec_threshold

    def interact(self, agent_a: Agent, agent_b: Agent, rng: random.Random) -> dict:
        result = self._nec_trade(agent_a, agent_b)
        flower_result = self._flower_trade(agent_a, agent_b)
        result["flower_trade"] = flower_result.get("flower_trade", False)
        result["flower_price"] = flower_result.get("flower_price", 0.0)
        return result

    def _nec_trade(self, agent_a: Agent, agent_b: Agent) -> dict:
        coin_a, nec_a = agent_a.portfolio.get("coin"), agent_a.portfolio.get("nec")
        coin_b, nec_b = agent_b.portfolio.get("coin"), agent_b.portfolio.get("nec")
        shadow_a = coin_a / nec_a if nec_a > 0 else float("inf")
        shadow_b = coin_b / nec_b if nec_b > 0 else float("inf")
        if shadow_a == shadow_b:
            return {"trade": False, "price": 0.0, "quantity": 0.0}
        if shadow_a > shadow_b:
            buyer, seller, coin_buyer, nec_seller = agent_a, agent_b, coin_a, nec_b
            shadow_buyer, shadow_seller = shadow_a, shadow_b
        else:
            buyer, seller, coin_buyer, nec_seller = agent_b, agent_a, coin_b, nec_a
            shadow_buyer, shadow_seller = shadow_b, shadow_a
        seller_surplus = nec_seller - self.min_reserve
        if seller_surplus <= 0:
            return {"trade": False, "price": 0.0, "quantity": 0.0}
        price = shadow_seller if shadow_buyer == float("inf") else (shadow_buyer + shadow_seller) / 2.0
        if price <= 0:
            return {"trade": False, "price": 0.0, "quantity": 0.0}
        quantity = min(seller_surplus, coin_buyer / price)
        if quantity <= 0:
            return {"trade": False, "price": 0.0, "quantity": 0.0}
        coin_paid = min(quantity * price, coin_buyer)
        quantity = coin_paid / price
        seller.portfolio.remove("nec", quantity)
        buyer.portfolio.add("nec", quantity)
        buyer.portfolio.remove("coin", coin_paid)
        seller.portfolio.add("coin", coin_paid)
        return {"trade": True, "price": price, "quantity": quantity,
                "coin": coin_paid, "buyer_id": buyer.id, "seller_id": seller.id}

    def _flower_trade(self, agent_a: Agent, agent_b: Agent) -> dict:
        for buyer, seller in ((agent_a, agent_b), (agent_b, agent_a)):
            if buyer.portfolio.get("nec") <= self.nec_threshold:
                continue
            if seller.flower_count == 0:
                continue
            coin_buyer = buyer.portfolio.get("coin")
            if coin_buyer <= 0:
                continue
            fc_buyer = buyer.flower_count
            fc_seller = seller.flower_count
            # Shadow price = coin / flower_count (coin per flower owned)
            shadow_buyer = coin_buyer / fc_buyer if fc_buyer > 0 else float("inf")
            coin_seller = seller.portfolio.get("coin")
            shadow_seller = coin_seller / fc_seller  # seller always has fc_seller > 0
            if shadow_buyer <= shadow_seller:
                continue  # buyer not willing to pay more than seller's valuation
            price = shadow_seller if shadow_buyer == float("inf") else (shadow_buyer + shadow_seller) / 2.0
            if price <= 0:
                continue
            coin_paid = min(price, coin_buyer)
            # Transfer 1 flower (oldest) from seller to buyer
            flower_tick = seller._flower_ticks.pop(0)
            buyer._flower_ticks.append(flower_tick)
            buyer.portfolio.remove("coin", coin_paid)
            seller.portfolio.add("coin", coin_paid)
            return {"flower_trade": True, "flower_price": coin_paid}
        return {"flower_trade": False, "flower_price": 0.0}


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
