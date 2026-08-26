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


class CrossSectionalMomentumStrategy(Strategy):
    """12-1 cross-sectional momentum on a panel: long the top decile, flat
    everything else. Jegadeesh & Titman (1993) formation, long-only.

    At each formation date t (every `rebalance` bars, once `lookback + 1`
    bars of history exist) the signal for every eligible name is

        close[t - skip] / close[t - lookback] - 1

    with skip = 21: the most recent month is EXCLUDED, because the shortest
    horizon reverses and a 12-0 implementation measures momentum minus
    reversal. Eligible means: a price today, prices at both endpoints of the
    window, and -- when a membership mask is supplied -- point-in-time index
    membership on the formation date.

    The strategy never sees cash or positions (that rule survives Day 9).
    It tracks ITS OWN last selection, so a chosen name that has died since
    the last formation still receives its direction-0 exit signal even
    though it no longer prints a bar; the portfolio then sells it at the
    handler's last known price, which is the corpse policy end to end.

    Emits direction=+1 for the chosen `top_n` and direction=0 for every
    other name with a bar today (plus dead ex-holdings). The flood of zeros
    is intentional and cheap: Portfolio.on_signal computes delta == 0 for
    names neither held nor wanted and never creates an order.

    Sizing note, from the guide: construct the Portfolio with
    TargetWeightSizer(1 / top_n). The default weight of 1.0 would put the
    whole account into whichever signal dequeues first.
    """

    def __init__(self, data, membership=None, top_n: int = 50,
                 lookback: int = 252, skip: int = 21,
                 rebalance: int = 21) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if not 0 <= skip < lookback:
            raise ValueError("need 0 <= skip < lookback")
        if rebalance < 1:
            raise ValueError("rebalance must be at least 1 bar")
        self.data = data
        self.membership = (membership.rename(columns=str)
                           if membership is not None else None)
        self.top_n = int(top_n)
        self.lookback = int(lookback)
        self.skip = int(skip)
        self.rebalance = int(rebalance)
        self._bars_since_form: int | None = None
        self._holding: set = set()

    def on_market(self, event: MarketEvent, events: queue.Queue) -> None:
        if self._bars_since_form is not None:
            self._bars_since_form += 1
            if self._bars_since_form < self.rebalance:
                return

        window = self.data.latest_closes(self.lookback + 1)
        if len(window) < self.lookback + 1:
            return                                  # not enough history yet

        today = window.iloc[-1]
        past = window.iloc[0]                       # t - lookback
        ref = window.iloc[-1 - self.skip]           # t - skip
        eligible = today.notna() & past.notna() & ref.notna()
        if self.membership is not None:
            member = (self.membership
                      .reindex(index=[event.timestamp],
                               columns=window.columns)
                      .iloc[0].fillna(False).astype(bool))
            eligible &= member

        scores = (ref / past - 1.0)[eligible]
        chosen = (set() if scores.empty else
                  set(scores.sort_values(ascending=False, kind="mergesort")
                      .head(self.top_n).index))

        for sym in sorted(chosen):
            events.put(SignalEvent(timestamp=event.timestamp,
                                   symbol=sym, direction=1))
        with_bar = set(today.index[today.notna()])
        for sym in sorted((with_bar | self._holding) - chosen):
            events.put(SignalEvent(timestamp=event.timestamp,
                                   symbol=sym, direction=0))

        self._holding = chosen
        self._bars_since_form = 0
