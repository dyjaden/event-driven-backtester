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
    """Goes long on the first bar and never changes its mind.

    Deliberately trivial. Its job is to be a known-answer baseline: with zero
    costs, a buy-and-hold backtest MUST reproduce the asset's own return. If it
    does not, the engine is wrong, not the strategy.
    """

    def __init__(self, data: HistoricBarHandler) -> None:
        self.data = data
        self.invested = False

    def on_market(self, event: MarketEvent, events: queue.Queue) -> None:
        if self.invested:
            return
        events.put(
            SignalEvent(
                timestamp=event.timestamp,
                symbol=self.data.symbol,
                direction=1,
            )
        )
        self.invested = True