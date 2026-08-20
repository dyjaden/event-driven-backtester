"""Strategies decide DIRECTION only. Sizing belongs to the Portfolio."""
from __future__ import annotations
import queue
from abc import ABC, abstractmethod

from .data import HistoricBarHandler
from .events import MarketEvent, SignalEvent


class Strategy(ABC):
    """Consumes MarketEvents, emits SignalEvents. Never sees cash or positions."""

    @abstractmethod
    def on_market(self, event: MarketEvent, events: queue.Queue) -> None:
        ...


class BuyAndHoldStrategy(Strategy):
    """Goes long once and never changes its mind.

    `warmup` suppresses the signal for the first N bars. It defaults to 0, so
    every existing test is unaffected. Set it when the execution model needs
    trailing statistics that do not exist on bar 0.
    """

    def __init__(self, data: HistoricBarHandler, warmup: int = 0) -> None:
        if warmup < 0:
            raise ValueError("warmup cannot be negative")
        self.data = data
        self.warmup = int(warmup)
        self.invested = False
        self._bars_seen = 0

    def on_market(self, event: MarketEvent, events: queue.Queue) -> None:
        self._bars_seen += 1
        if self.invested or self._bars_seen <= self.warmup:
            return
        events.put(SignalEvent(timestamp=event.timestamp,
                               symbol=self.data.symbol, direction=1))
        self.invested = True

class AlternatingStrategy(Strategy):
    """Flips between fully long and flat on every single bar.

    Nobody would trade this. It exists because buy-and-hold trades ONCE, which
    hides the cost model completely. To see costs you need turnover, and this
    generates the maximum possible amount of it.
    """

    def __init__(self, data: HistoricBarHandler) -> None:
        self.data = data
        self.long = False

    def on_market(self, event: MarketEvent, events: queue.Queue) -> None:
        self.long = not self.long
        events.put(SignalEvent(
            timestamp=event.timestamp,
            symbol=self.data.symbol,
            direction=1 if self.long else 0,
        ))