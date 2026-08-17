"""Cash, positions, sizing and the equity curve. The Strategy sees none of this."""
from __future__ import annotations
import queue

import pandas as pd

from .data import HistoricBarHandler
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent


class Portfolio:
    """Owns the money.

    Strategy decides direction, Portfolio decides size. Keeping them apart is
    what lets risk rules change without touching strategy logic.
    """

    def __init__(self, initial_capital: float = 100_000.0,
                 target_weight: float = 1.0) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0.0 < target_weight <= 1.0:
            raise ValueError("target_weight must be in (0, 1]")

        self.initial_capital = float(initial_capital)
        self.target_weight = float(target_weight)

        self.cash = float(initial_capital)
        self.positions: dict[str, int] = {}
        self.equity_history: list[tuple] = []
        self.fills: list[FillEvent] = []

    # ------------------------------------------------------------- state
    def holdings_value(self, data: HistoricBarHandler) -> float:
        """Market value of everything held. Single-symbol until Day 9."""
        if not self.positions:
            return 0.0
        price = data.current_price()
        return sum(qty * price for qty in self.positions.values())

    def equity(self, data: HistoricBarHandler) -> float:
        return self.cash + self.holdings_value(data)

    # ------------------------------------------------------------- hooks
    def mark_to_market(self, event: MarketEvent, data: HistoricBarHandler) -> None:
        """Record account value once per bar. This builds the equity curve."""
        self.equity_history.append((event.timestamp, self.equity(data)))

    def on_signal(self, event: SignalEvent, events: queue.Queue,
                  data: HistoricBarHandler) -> None:
        """Turn a direction into a concrete share count."""
        price = data.current_price()
        if price <= 0:
            return

        held = self.positions.get(event.symbol, 0)
        target_value = self.equity(data) * self.target_weight * event.direction
        target_qty = int(target_value / price)      # truncates toward zero
        delta = target_qty - held

        if delta == 0:
            return
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