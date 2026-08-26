"""Cash, positions, sizing and the equity curve. The Strategy sees none of this."""
from __future__ import annotations
import queue

import pandas as pd

from .data import HistoricBarHandler
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent

from abc import ABC, abstractmethod


class Sizer(ABC):
    """Decides HOW MANY shares to hold. Never decides direction.

    Direction is the Strategy's job and arrives in the SignalEvent. This
    object answers only the size question, which is the part you control.
    """

    @abstractmethod
    def target_quantity(self, *, direction: int, price: float,
                        equity: float, initial_capital: float) -> int:
        """Signed target position. Negative means short."""
        ...


class TargetWeightSizer(Sizer):
    """Hold `weight` of CURRENT equity.

    Compounds: as the account grows the position grows with it. This is what
    Day 4 did, and it is the default.
    """

    def __init__(self, weight: float = 1.0) -> None:
        if not 0.0 < weight <= 1.0:
            raise ValueError("weight must be in (0, 1]")
        self.weight = float(weight)

    def target_quantity(self, *, direction, price, equity, initial_capital):
        return int(equity * self.weight * direction / price)


class FixedFractionalSizer(Sizer):
    """Hold `fraction` of INITIAL capital.

    Does NOT compound: constant dollar exposure for the whole run. Losses do
    not shrink the position and gains do not grow it.
    """

    def __init__(self, fraction: float = 1.0) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = float(fraction)

    def target_quantity(self, *, direction, price, equity, initial_capital):
        return int(initial_capital * self.fraction * direction / price)


class FixedQuantitySizer(Sizer):
    """Always exactly N shares, whatever the account is worth.

    Mostly a test instrument: it makes expected values trivial to compute
    by hand.
    """

    def __init__(self, quantity: int = 100) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self.quantity = int(quantity)

    def target_quantity(self, *, direction, price, equity, initial_capital):
        return self.quantity * direction

class Portfolio:
    """Owns the money.

    Strategy decides direction, Portfolio decides size. Keeping them apart is
    what lets risk rules change without touching strategy logic.
    """

    def __init__(self, initial_capital: float = 100_000.0,
                 sizer: Sizer | None = None,
                 min_trade_fraction: float = 0.0) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0.0 <= min_trade_fraction < 1.0:
            raise ValueError("min_trade_fraction must be in [0, 1)")

        self.initial_capital = float(initial_capital)
        self.sizer = sizer if sizer is not None else TargetWeightSizer(1.0)
        # The rebalance band (Day 9). Integer truncation makes a held name's
        # target drift by a share or two as equity moves, and on a 50-name
        # book every formation date would emit ~50 dust orders, each paying
        # the commission minimum. Orders smaller than this fraction of
        # current equity are skipped. 0.0 (the default) trades everything,
        # which keeps every pre-Day-9 result byte-identical.
        self.min_trade_fraction = float(min_trade_fraction)

        self.cash = float(initial_capital)
        self.positions: dict[str, int] = {}
        self.equity_history: list[tuple] = []
        self.fills: list[FillEvent] = []

    # ------------------------------------------------------------- state
    def holdings_value(self, data: HistoricBarHandler) -> float:
        """Market value of everything held. Single-symbol until Day 9."""
        if not self.positions:
            return 0.0
        return sum(qty * data.current_price(sym)
             for sym, qty in self.positions.items() if qty)

    def equity(self, data: HistoricBarHandler) -> float:
        return self.cash + self.holdings_value(data)

    # ------------------------------------------------------------- hooks
    def mark_to_market(self, event: MarketEvent, data: HistoricBarHandler) -> None:
        """Record account value once per bar. This builds the equity curve."""
        self.equity_history.append((event.timestamp, self.equity(data)))

    def on_signal(self, event: SignalEvent, events: queue.Queue,
                  data: HistoricBarHandler) -> None:
        """Turn a direction into a concrete share count."""
        price = data.current_price(event.symbol)
        if price <= 0:
            return

        held = self.positions.get(event.symbol, 0)
        target_qty = self.sizer.target_quantity(
            direction=event.direction,
            price=price,
            equity=self.equity(data),
            initial_capital=self.initial_capital,
        )
        delta = target_qty - held

        if delta == 0:
            return
        if (self.min_trade_fraction
                and abs(delta) * price
                < self.min_trade_fraction * self.equity(data)):
            return                      # dust: inside the rebalance band
        events.put(OrderEvent(timestamp=event.timestamp,
                              symbol=event.symbol,
                              quantity=delta))

    def on_fill(self, fill: FillEvent) -> None:
        """Money and shares move here, and nowhere else."""
        self.positions[fill.symbol] = (
            self.positions.get(fill.symbol, 0) + fill.quantity
        )
        self.cash -= fill.gross_value + fill.total_cost
        self.fills.append(fill)

    # ------------------------------------------------------------- output
    def equity_curve(self) -> pd.Series:
        if not self.equity_history:
            return pd.Series(dtype=float, name="equity")
        idx, vals = zip(*self.equity_history)
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")