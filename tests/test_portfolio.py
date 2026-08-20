import numpy as np
import pandas as pd
import pytest

from backtester.data import HistoricBarHandler
from backtester.engine import Backtest
from backtester.events import FillEvent
from backtester.execution import NaiveExecutionHandler
from backtester.portfolio import (
    FixedFractionalSizer,
    FixedQuantitySizer,
    Portfolio,
    TargetWeightSizer,
)
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
        TargetWeightSizer(weight=1.5)
    with pytest.raises(ValueError):
        FixedQuantitySizer(quantity=0)

def test_default_sizer_is_target_weight_one():
    pf = Portfolio()
    assert isinstance(pf.sizer, TargetWeightSizer)
    assert pf.sizer.weight == 1.0


def test_fixed_quantity_buys_exactly_n_shares():
    frame = make_frame(10)
    data = HistoricBarHandler(frame, "TEST")
    pf = Portfolio(initial_capital=100_000.0,
                   sizer=FixedQuantitySizer(25))
    Backtest(data, BuyAndHoldStrategy(data), pf,
             NaiveExecutionHandler()).run()
    assert pf.positions["TEST"] == 25


def test_half_weight_holds_half_the_shares():
    frame = make_frame(10)
    full, half = [], []
    for sizer, out in ((TargetWeightSizer(1.0), full),
                       (TargetWeightSizer(0.5), half)):
        data = HistoricBarHandler(frame, "TEST")
        pf = Portfolio(initial_capital=100_000.0, sizer=sizer)
        Backtest(data, BuyAndHoldStrategy(data), pf,
                 NaiveExecutionHandler()).run()
        out.append(pf.positions["TEST"])
    assert half[0] == pytest.approx(full[0] / 2, abs=1)


def test_compounding_and_non_compounding_differ_when_resignalled():
    """The whole point of having two sizers.

    Buy-and-hold signals once, so both agree. A strategy that re-signals
    after the account has grown does not: target-weight scales the position
    up, fixed-fractional does not.
    """
    price, equity, capital = 100.0, 200_000.0, 100_000.0   # account doubled
    tw = TargetWeightSizer(1.0).target_quantity(
        direction=1, price=price, equity=equity, initial_capital=capital)
    ff = FixedFractionalSizer(1.0).target_quantity(
        direction=1, price=price, equity=equity, initial_capital=capital)
    assert tw == 2000
    assert ff == 1000
    assert tw > ff


def test_short_direction_is_not_over_sized():
    """int(x / p) truncates toward zero. Floor division would not."""
    qty = TargetWeightSizer(1.0).target_quantity(
        direction=-1, price=10.0, equity=1005.0, initial_capital=1005.0)
    assert qty == -100        # NOT -101

from backtester.costs import HalfSpreadSlippage, PerShareCommission
from backtester.execution import SimulatedExecutionHandler
from backtester.strategy import AlternatingStrategy


def _final_equity(frame, strategy_cls, execution):
    data = HistoricBarHandler(frame, "TEST")
    pf = Portfolio(initial_capital=100_000.0)
    Backtest(data, strategy_cls(data), pf, execution).run()
    return pf.equity_curve().iloc[-1]


def _costly():
    return SimulatedExecutionHandler(
        commission=PerShareCommission(),
        slippage=HalfSpreadSlippage(spread_bps=10.0),
    )


def test_costs_reduce_final_equity():
    frame = make_frame(50)
    free = _final_equity(frame, BuyAndHoldStrategy, NaiveExecutionHandler())
    paid = _final_equity(frame, BuyAndHoldStrategy, _costly())
    assert paid < free


def test_turnover_amplifies_the_same_cost_model():
    """THE lesson of Day 6.

    Identical cost model. The only difference is how often the strategy
    trades. Costs are nearly invisible on buy-and-hold and severe on a
    high-turnover strategy, which is why turnover belongs in every honest
    performance report.
    """
    frame = make_frame(50)

    bh_free = _final_equity(frame, BuyAndHoldStrategy, NaiveExecutionHandler())
    bh_paid = _final_equity(frame, BuyAndHoldStrategy, _costly())

    alt_free = _final_equity(frame, AlternatingStrategy, NaiveExecutionHandler())
    alt_paid = _final_equity(frame, AlternatingStrategy, _costly())

    bh_drag = 1.0 - bh_paid / bh_free
    alt_drag = 1.0 - alt_paid / alt_free

    assert alt_drag > bh_drag * 10