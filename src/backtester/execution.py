"""Orders to fills, with costs."""
from __future__ import annotations
import queue

from .costs import (
    CommissionModel,
    SlippageModel,
    ZeroCommission,
    ZeroSlippage,
)
from .data import HistoricBarHandler
from .events import FillEvent, OrderEvent


class SimulatedExecutionHandler:
    """Fills at this bar's close, adjusted for spread, plus commission.

    Composes two swappable models so the SAME handler runs with costs on or
    off. That comparison is the headline result of the project.
    """

    def __init__(self, commission: CommissionModel | None = None,
                 slippage: SlippageModel | None = None) -> None:
        self.commission_model = commission or ZeroCommission()
        self.slippage_model = slippage or ZeroSlippage()

    def on_order(self, event: OrderEvent, events: queue.Queue,
                 data: HistoricBarHandler) -> None:
        mid = data.current_price()
        fill_price = self.slippage_model.fill_price(event.quantity, mid)
        commission = self.commission_model.commission(event.quantity, fill_price)
        events.put(FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            quantity=event.quantity,
            fill_price=fill_price,
            commission=commission,
            impact_cost=0.0,          # Day 7
        ))


class NaiveExecutionHandler(SimulatedExecutionHandler):
    """The zero-cost baseline arm, kept under its own name deliberately.

    Not scaffolding. 'Costs off versus costs on' needs both arms to exist
    side by side, and the baseline deserves a name you can point at.
    """

    def __init__(self) -> None:
        super().__init__(ZeroCommission(), ZeroSlippage())