import numpy as np
import pandas as pd
import pytest

from backtester.data import HistoricBarHandler
from backtester.engine import Backtest
from backtester.events import MarketEvent
from backtester.strategy import BuyAndHoldStrategy


def make_frame(n=10):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": np.arange(n, dtype=float) + 100,
         "high": np.arange(n, dtype=float) + 101,
         "low": np.arange(n, dtype=float) + 99,
         "close": np.arange(n, dtype=float) + 100.5,
         "volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


def test_streams_every_bar_exactly_once():
    frame = make_frame(10)
    data = HistoricBarHandler(frame, "TEST")
    bt = Backtest(data, BuyAndHoldStrategy(data))
    counts = bt.run()
    assert counts["MarketEvent"] == 10


def test_buy_and_hold_signals_exactly_once():
    frame = make_frame(10)
    data = HistoricBarHandler(frame, "TEST")
    bt = Backtest(data, BuyAndHoldStrategy(data))
    counts = bt.run()
    assert counts["SignalEvent"] == 1


def test_queue_is_empty_when_the_run_ends():
    """If anything is left in the queue, an event was created and never handled."""
    frame = make_frame(10)
    data = HistoricBarHandler(frame, "TEST")
    bt = Backtest(data, BuyAndHoldStrategy(data))
    bt.run()
    assert bt.events.empty()


def test_strategy_never_sees_the_future():
    """THE test. Records the bars visible at each signal and asserts none
    postdate the signal's own timestamp."""
    frame = make_frame(10)
    data = HistoricBarHandler(frame, "TEST")

    seen = []

    class Spy(BuyAndHoldStrategy):
        def on_market(self, event, events):
            window = self.data.latest_bars(5)
            if len(window):
                seen.append((event.timestamp, window.index.max()))
            super().on_market(event, events)

    Backtest(data, Spy(data)).run()
    assert seen, "strategy never ran"
    for now, latest_visible in seen:
        assert latest_visible <= now


def test_unknown_event_type_raises():
    frame = make_frame(3)
    data = HistoricBarHandler(frame, "TEST")
    bt = Backtest(data, BuyAndHoldStrategy(data))

    class Bogus:
        timestamp = frame.index[0]

    with pytest.raises(TypeError):
        bt._handle(Bogus())