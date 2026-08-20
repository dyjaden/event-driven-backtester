"""Orders to fills, with costs."""
from __future__ import annotations
import queue

from .costs import (
    CommissionModel,
    ImpactModel,
    SlippageModel,
    ZeroCommission,
    ZeroImpact,
    ZeroSlippage,
)
from .data import HistoricBarHandler
from .events import FillEvent, OrderEvent

class SimulatedExecutionHandler:
    """Fills at this bar's close, adjusted for spread and impact, plus commission.

    Three swappable cost models plus one hard constraint. The same handler
    runs the zero-cost baseline and the fully-loaded arm, which is what makes
    the comparisons in `_demo()` meaningful.
    """

    def __init__(self, commission: CommissionModel | None = None,
                 slippage: SlippageModel | None = None,
                 impact: ImpactModel | None = None,
                 max_participation: float | None = None,
                 window: int = 21) -> None:
        self.commission_model = commission or ZeroCommission()
        self.slippage_model = slippage or ZeroSlippage()
        self.impact_model = impact or ZeroImpact()
        self.max_participation = max_participation
        self.window = window

    def on_order(self, event: OrderEvent, events: queue.Queue,
                 data: HistoricBarHandler) -> None:
        mid = data.current_price()
        adv = data.adv(self.window)
        quantity = event.quantity

        # --- the hard constraint, applied BEFORE any cost is computed
        if self.max_participation is not None and adv > 0:
            limit = int(self.max_participation * adv)
            if abs(quantity) > limit:
                quantity = limit if quantity > 0 else -limit

        if quantity == 0:
            return                      # nothing tradable this bar, no fill

        daily_vol = data.trailing_volatility(self.window)
        move = self.impact_model.price_move(quantity, mid, adv, daily_vol)

        # Slippage rides in the PRICE. Impact rides in impact_cost. See the
        # callout below: Portfolio.on_fill already debits FillEvent.total_cost,
        # so putting impact in both places charges it twice.
        fill_price = self.slippage_model.fill_price(quantity, mid)
        commission = self.commission_model.commission(quantity, fill_price)

        events.put(FillEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            quantity=quantity,
            fill_price=fill_price,          # slippage only, NOT impact
            commission=commission,
            impact_cost=abs(quantity) * move,
        ))


class NaiveExecutionHandler(SimulatedExecutionHandler):
    """The zero-cost, infinite-liquidity baseline arm."""

    def __init__(self) -> None:
        super().__init__(ZeroCommission(), ZeroSlippage(), ZeroImpact(), None)