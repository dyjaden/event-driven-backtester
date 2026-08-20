"""Cost model unit tests, plus the execution-handler tests that prove the
models are reached and that no cost is charged twice.

Day 6 built the commission and slippage sections.
Day 7 added the participation cap and the impact accounting section.
"""
import queue

import numpy as np
import pandas as pd
import pytest

from backtester.costs import (
    HalfSpreadSlippage,
    PerShareCommission,
    SquareRootImpact,
    ZeroCommission,
    ZeroImpact,
    ZeroSlippage,
)
from backtester.data import HistoricBarHandler
from backtester.events import FillEvent, OrderEvent
from backtester.execution import SimulatedExecutionHandler
from backtester.portfolio import Portfolio


def make_frame(n=10, start=100.0, step=1.0):
    """Local copy. The original lives in test_portfolio.py, which is not
    importable from here."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.arange(n, dtype=float) * step + start
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


# =========================================================== Day 6: commission
def test_commission_is_positive_on_a_sell():
    """quantity is negative on a sell; the cost is not."""
    c = PerShareCommission()
    assert c.commission(-500, 100.0) > 0


def test_commission_respects_the_floor():
    c = PerShareCommission(per_share=0.005, minimum=1.0)
    assert c.commission(10, 100.0) == pytest.approx(1.0)      # 10*0.005 = 0.05
    assert c.commission(1000, 100.0) == pytest.approx(5.0)    # above the floor


def test_commission_capped_as_share_of_trade_value():
    """A penny stock must not cost more in commission than a cap allows."""
    c = PerShareCommission(per_share=0.005, minimum=1.0, max_pct_of_value=0.01)
    # 10,000 shares at $0.50 = $5,000 notional. Raw would be $50; cap is $50.
    assert c.commission(10_000, 0.50) == pytest.approx(50.0)
    # At $0.10 the notional is $1,000, so the cap of $10 binds.
    assert c.commission(10_000, 0.10) == pytest.approx(10.0)


def test_zero_commission_is_zero():
    assert ZeroCommission().commission(1000, 100.0) == 0.0


# ============================================================= Day 6: slippage
def test_buy_fills_above_mid_and_sell_below():
    s = HalfSpreadSlippage(spread_bps=10.0)
    assert s.fill_price(100, 100.0) > 100.0
    assert s.fill_price(-100, 100.0) < 100.0


def test_half_spread_is_exactly_half_the_full_spread():
    s = HalfSpreadSlippage(spread_bps=10.0)      # 10bp = 0.1% full spread
    # half of 0.1% of 100.0 is 0.05
    assert s.fill_price(1, 100.0) == pytest.approx(100.05)
    assert s.fill_price(-1, 100.0) == pytest.approx(99.95)


def test_zero_slippage_returns_mid():
    assert ZeroSlippage().fill_price(100, 123.45) == 123.45


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        HalfSpreadSlippage(spread_bps=-1)
    with pytest.raises(ValueError):
        PerShareCommission(per_share=-0.01)
    with pytest.raises(ValueError):
        PerShareCommission(max_pct_of_value=0)


# ================================================ Day 7: the execution handler
def _primed(max_participation=None, commission=None, slippage=None,
            impact=None, bars=6):
    """A data handler advanced `bars` bars, a drained queue, and a handler.

    It has to be more than two bars: trailing_volatility needs at least three
    rows before it returns a number instead of NaN, and a NaN vol makes
    SquareRootImpact correctly return zero. Prime on one bar and the impact
    tests below silently pass for the wrong reason.
    """
    frame = make_frame(60)
    data = HistoricBarHandler(frame, "TEST")
    events = queue.Queue()
    for _ in range(bars):
        data.update_bars(events)
    while not events.empty():
        events.get()

    handler = SimulatedExecutionHandler(
        commission or ZeroCommission(),
        slippage or ZeroSlippage(),
        impact or ZeroImpact(),
        max_participation=max_participation,
    )
    return frame, data, events, handler


def _order(data, quantity):
    return OrderEvent(data.current_time, "TEST", quantity)


def test_participation_cap_truncates_a_buy():
    frame, data, events, handler = _primed(max_participation=0.10)
    huge = 10 * int(frame["volume"].iloc[0])          # 10,000,000 shares
    handler.on_order(_order(data, huge), events, data)

    fill = events.get()
    assert fill.quantity < huge
    assert fill.quantity <= int(0.10 * data.adv())    # 100,000 shares


def test_participation_cap_truncates_a_sell_too():
    """The sign is easy to lose. -10,000,000 becomes -100,000, not +100,000."""
    frame, data, events, handler = _primed(max_participation=0.10)
    huge = -10 * int(frame["volume"].iloc[0])
    handler.on_order(_order(data, huge), events, data)

    fill = events.get()
    assert fill.quantity < 0
    assert fill.quantity > huge                       # less negative
    assert abs(fill.quantity) <= int(0.10 * data.adv())


def test_no_cap_means_no_truncation():
    frame, data, events, handler = _primed(max_participation=None)
    huge = 10 * int(frame["volume"].iloc[0])
    handler.on_order(_order(data, huge), events, data)

    assert events.get().quantity == huge


def test_cap_does_not_touch_an_order_that_already_fits():
    frame, data, events, handler = _primed(max_participation=0.10)
    small = 1_000
    handler.on_order(_order(data, small), events, data)

    assert events.get().quantity == small


def test_impact_does_not_move_the_fill_price():
    """Slippage rides in the price. Impact rides in impact_cost.

    Both channels would double-charge, because Portfolio.on_fill debits
    gross_value + total_cost and total_cost already includes impact_cost.
    """
    frame, data, events, handler = _primed(impact=SquareRootImpact(1.0))
    handler.on_order(_order(data, 500_000), events, data)

    fill = events.get()
    assert fill.fill_price == pytest.approx(data.current_price())
    assert fill.impact_cost > 0


def test_impact_is_charged_exactly_once():
    """The arithmetic behind the test above, asserted on cash directly."""
    pf = Portfolio(initial_capital=100_000.0)
    pf.on_fill(FillEvent(timestamp=pd.Timestamp("2024-01-02"), symbol="TEST",
                         quantity=100, fill_price=50.0,
                         commission=1.0, impact_cost=25.0))

    assert pf.cash == pytest.approx(100_000.0 - 100 * 50.0 - 1.0 - 25.0)


def test_capping_reduces_the_impact_paid():
    """What the cap is FOR. Same order, same models, less damage."""
    frame, data_free, ev_free, h_free = _primed(impact=SquareRootImpact(1.0))
    _, data_cap, ev_cap, h_cap = _primed(max_participation=0.10,
                                         impact=SquareRootImpact(1.0))
    huge = 10 * int(frame["volume"].iloc[0])

    h_free.on_order(_order(data_free, huge), ev_free, data_free)
    h_cap.on_order(_order(data_cap, huge), ev_cap, data_cap)

    assert ev_cap.get().impact_cost < ev_free.get().impact_cost


def test_zero_impact_handler_reports_no_impact_cost():
    """The baseline arm must produce a clean zero, not a small number."""
    frame, data, events, handler = _primed(impact=ZeroImpact())
    handler.on_order(_order(data, 500_000), events, data)

    assert events.get().impact_cost == 0.0