"""Event types. Everything in the engine communicates through these."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MarketEvent:
    """A new bar is available. Emitted by the DataHandler."""
    timestamp: datetime


@dataclass(frozen=True)
class SignalEvent:
    """The Strategy's opinion. Deliberately carries NO position size."""
    timestamp: datetime
    symbol: str
    direction: int              # +1 long, -1 short, 0 flat
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in (-1, 0, 1):
            raise ValueError(f"direction must be -1, 0 or 1, got {self.direction}")


@dataclass(frozen=True)
class OrderEvent:
    """A concrete instruction. Sizing has been decided by the Portfolio."""
    timestamp: datetime
    symbol: str
    quantity: int              # signed: negative = sell
    order_type: str = "MKT"

    def __post_init__(self) -> None:
        if self.quantity == 0:
            raise ValueError("refusing to create a zero-quantity order")


@dataclass(frozen=True)
class FillEvent:
    """What actually happened, after costs."""
    timestamp: datetime
    symbol: str
    quantity: int
    fill_price: float          # already includes slippage
    commission: float = 0.0
    impact_cost: float = 0.0

    @property
    def gross_value(self) -> float:
        return self.quantity * self.fill_price

    @property
    def total_cost(self) -> float:
        return self.commission + self.impact_cost