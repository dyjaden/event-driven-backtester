import numpy as np
import pandas as pd
import pytest

from backtester.data import HistoricBarHandler
from backtester.engine import Backtest
from backtester.events import FillEvent
from backtester.execution import NaiveExecutionHandler
from backtester.portfolio import Portfolio
from backtester.strategy import BuyAndHoldStrategy


def make_frame(n=10, start=100.0, step=1.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.arange(n, dtype=float) * step + start
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


def run(frame, capital=100_000.0):
    data = HistoricBarHandler(frame, "TEST")
    pf = Portfolio(initial_capital=capital)
    Backtest(data, BuyAndHoldStrategy(data), pf, NaiveExecutionHandler()).run()
    return pf, data


def test_zero_cost_fill_does_not_change_equity():
    """THE conservation test. Swapping cash for shares at the same price
    cannot create or destroy value."""
    frame = make_frame(5)
    data = HistoricBarHandler(frame, "TEST")
    data.update_bars(__import__("queue").Queue())      # advance to bar 0
    pf = Portfolio(initial_capital=100_000.0)

    before = pf.equity(data)
    price = data.current_price()
    pf.on_fill(FillEvent(timestamp=frame.index[0], symbol="TEST",
                         quantity=10, fill_price=price))
    after = pf.equity(data)

    assert after == pytest.approx(before, abs=1e-9)


def test_buy_and_hold_final_equity_is_exactly_predictable():
    """With zero costs the answer is arithmetic, so assert it exactly."""
    frame = make_frame(10)
    pf, _ = run(frame)

    p0, pn = frame["close"].iloc[0], frame["close"].iloc[-1]
    qty = int(100_000.0 / p0)
    remainder = 100_000.0 - qty * p0
    expected = remainder + qty * pn

    assert pf.equity_curve().iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_cash_never_negative_at_full_weight():
    frame = make_frame(20)
    pf, _ = run(frame)
    assert pf.cash >= 0.0


def test_equity_curve_has_one_point_per_bar():
    frame = make_frame(37)
    pf, _ = run(frame)
    assert len(pf.equity_curve()) == 37


def test_positions_equal_sum_of_fills():
    frame = make_frame(15)
    pf, _ = run(frame)
    assert pf.positions["TEST"] == sum(f.quantity for f in pf.fills)


def test_rejects_bad_construction():
    with pytest.raises(ValueError):
        Portfolio(initial_capital=0)
    with pytest.raises(ValueError):
        Portfolio(target_weight=1.5)