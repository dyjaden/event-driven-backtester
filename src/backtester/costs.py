"""Transaction cost models. Pure functions of an order, no engine coupling."""
from __future__ import annotations
from abc import ABC, abstractmethod


# ------------------------------------------------------------- commission
class CommissionModel(ABC):
    @abstractmethod
    def commission(self, quantity: int, fill_price: float) -> float:
        """Always a POSITIVE cost, whatever the sign of quantity."""


class ZeroCommission(CommissionModel):
    def commission(self, quantity: int, fill_price: float) -> float:
        return 0.0


class PerShareCommission(CommissionModel):
    """Interactive-Brokers-shaped: per share, with a floor and a cap.

    The cap matters for cheap stocks. Without it, 10,000 shares of a $0.50
    stock would cost $50 of commission on a $5,000 trade.
    """

    def __init__(self, per_share: float = 0.005, minimum: float = 1.0,
                 max_pct_of_value: float = 0.01) -> None:
        if per_share < 0 or minimum < 0:
            raise ValueError("commission parameters cannot be negative")
        if not 0.0 < max_pct_of_value <= 1.0:
            raise ValueError("max_pct_of_value must be in (0, 1]")
        self.per_share = float(per_share)
        self.minimum = float(minimum)
        self.max_pct_of_value = float(max_pct_of_value)

    def commission(self, quantity: int, fill_price: float) -> float:
        shares = abs(int(quantity))
        if shares == 0:
            return 0.0
        raw = max(shares * self.per_share, self.minimum)
        cap = shares * abs(fill_price) * self.max_pct_of_value
        return min(raw, cap)


# --------------------------------------------------------------- slippage
class SlippageModel(ABC):
    @abstractmethod
    def fill_price(self, quantity: int, mid: float) -> float:
        """The price you ACTUALLY get. Never better than mid."""


class ZeroSlippage(SlippageModel):
    def fill_price(self, quantity: int, mid: float) -> float:
        return mid


class HalfSpreadSlippage(SlippageModel):
    """You cross the spread, so you pay half of it, in the direction that hurts.

    `spread_bps` is the FULL quoted spread in basis points. A buyer lifts the
    offer and pays mid + spread/2. A seller hits the bid and receives
    mid - spread/2.
    """

    def __init__(self, spread_bps: float = 1.0) -> None:
        if spread_bps < 0:
            raise ValueError("spread_bps cannot be negative")
        self.spread_bps = float(spread_bps)

    def fill_price(self, quantity: int, mid: float) -> float:
        half = mid * (self.spread_bps / 10_000.0) / 2.0
        return mid + half if quantity > 0 else mid - half