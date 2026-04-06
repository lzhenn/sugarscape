"""Agent module: Portfolio, Agent ABC, and concrete implementations."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class Portfolio:
    """Multi-asset holding container. Maps asset name (str) to quantity (float).

    Phase 1 uses a single asset 'coin'. Future phases can add arbitrary assets
    without changing this interface.
    """

    def __init__(self, holdings: dict[str, float] | None = None):
        self._holdings: dict[str, float] = dict(holdings or {})

    def get(self, asset: str) -> float:
        return self._holdings.get(asset, 0.0)

    def set(self, asset: str, qty: float) -> None:
        self._holdings[asset] = qty

    def add(self, asset: str, qty: float) -> None:
        self._holdings[asset] = self._holdings.get(asset, 0.0) + qty

    def remove(self, asset: str, qty: float) -> bool:
        """Remove qty of asset. Returns False if insufficient (no change)."""
        current = self._holdings.get(asset, 0.0)
        if current < qty:
            return False
        self._holdings[asset] = current - qty
        return True

    def total(self) -> float:
        return sum(self._holdings.values())

    def copy(self) -> Portfolio:
        return Portfolio(dict(self._holdings))

    def __repr__(self) -> str:
        items = ", ".join(f"{k}: {v:.1f}" for k, v in self._holdings.items())
        return f"Portfolio({{{items}}})"


class Agent(ABC):
    """Base class for all agents in the simulation."""

    _next_id: int = 0

    def __init__(self, portfolio: Portfolio, rng: random.Random,
                 position: float | None = None,
                 grid_pos: tuple[int, int] | None = None):
        self.id: int = Agent._next_id
        Agent._next_id += 1
        self.portfolio: Portfolio = portfolio
        self.rng: random.Random = rng
        self.alive: bool = True
        self.age: int = 0
        self.position: float | None = position
        self.grid_pos: tuple[int, int] | None = grid_pos
        self.can_refuse: bool = False  # if True, refuses to trade with wealthier agents

    @abstractmethod
    def step(self) -> None:
        """Per-tick lifecycle logic (aging, metabolism, death check)."""

    def wealth(self) -> float:
        return self.portfolio.total()

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._next_id = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.id}, {self.portfolio})"


class CoinAgent(Agent):
    """Phase 1 agent: holds only 'coin', no metabolism or death."""

    def __init__(self, initial_wealth: float, rng: random.Random):
        portfolio = Portfolio({"coin": initial_wealth})
        super().__init__(portfolio, rng)

    def step(self) -> None:
        self.age += 1
