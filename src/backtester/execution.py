"""Orders to fills. Today's version: instant, at the close, free."""
from __future__ import annotations
import queue

from .data import HistoricBarHandler
from .events import FillEvent, OrderEvent


class NaiveExecutionHandler:
    """Fills every order immediately at the current close, with zero cost.

    This is NOT a placeholder to be deleted on Day 6. It is the zero-cost
    baseline arm of the experiment. Day 6 adds a realistic handler, and the
    DIFFERENCE between the two is the headline result of the whole project.
    """

    def on_order(self, event: OrderEvent, events: queue.Queue,
                 data: HistoricBarHandler) -> None:
        events.put(FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            quantity=event.quantity,
            fill_price=data.current_price(),
            commission=0.0,
            impact_cost=0.0,
        ))