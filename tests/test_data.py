import queue
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from backtester.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from backtester.data import HistoricBarHandler


def make_frame(n=10):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    px = np.arange(100, 100 + n, dtype=float)
    return pd.DataFrame(
        {"open": px, "high": px + 1, "low": px - 1,
         "close": px, "volume": np.full(n, 1_000_000)},
        index=idx,
    )


def test_signal_rejects_bad_direction():
    with pytest.raises(ValueError):
        SignalEvent(datetime(2020, 1, 1), "SPY", 5)


def test_order_rejects_zero_quantity():
    with pytest.raises(ValueError):
        OrderEvent(datetime(2020, 1, 1), "SPY", 0)


def test_fill_cost_accounting():
    f = FillEvent(datetime(2020, 1, 1), "SPY", 10, 100.0,
                  commission=1.0, impact_cost=0.5)
    assert f.gross_value == 1000.0
    assert f.total_cost == 1.5


def test_no_lookahead():
    """The invariant the whole project rests on."""
    dh = HistoricBarHandler(make_frame(10), "SPY")
    events = queue.Queue()
    prices = []

    while dh.continue_backtest:
        dh.update_bars(events)
        if not dh.continue_backtest:
            break
        ev = events.get()
        assert isinstance(ev, MarketEvent)
        bars = dh.latest_bars(3)
        assert bars.index.max() <= ev.timestamp, "LOOK-AHEAD LEAK"
        assert len(bars) <= 3
        prices.append(dh.current_price())

    assert len(prices) == 10
    assert prices[0] == 100.0
    assert prices[-1] == 109.0


def test_window_truncates_at_start():
    dh = HistoricBarHandler(make_frame(10), "SPY")
    dh.update_bars(queue.Queue())
    assert len(dh.latest_bars(5)) == 1