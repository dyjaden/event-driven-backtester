"""Day 9, Step 3: the formation tests.

The one that matters is the riser-vs-spiker test: a name that spiked only
inside the skip month must NOT be picked. That is the difference between
12-1 momentum and a 12-0 implementation that measures momentum minus
reversal -- the classic off-by-a-month -- and the same panel run with
skip=0 proves the skip is what does the work.
"""
import queue

import numpy as np
import pandas as pd
import pytest

from backtester.data import HistoricBarHandler, PanelBarHandler
from backtester.engine import Backtest
from backtester.execution import NaiveExecutionHandler
from backtester.portfolio import (FixedFractionalSizer, Portfolio,
                                  TargetWeightSizer)
from backtester.strategy import BuyAndHoldStrategy, CrossSectionalMomentumStrategy

N = 300
LOOKBACK, SKIP, REBAL = 252, 21, 21


def three_name_panel(n=N):
    """RISER climbs all year; FLAT does nothing; SPIKER quintuples, but
    only inside the final 15 bars -- entirely within the skip month."""
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    riser = 100.0 * (1.003 ** np.arange(n))
    flat = np.full(n, 100.0)
    spiker = np.full(n, 100.0)
    spiker[-15:] = 500.0
    close = pd.DataFrame({"RISER": riser, "FLAT": flat, "SPIKER": spiker},
                         index=idx)
    volume = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return close, volume


def run_momentum(close, volume, top_n=1, membership=None, skip=SKIP):
    data = PanelBarHandler(close, volume)
    pf = Portfolio(initial_capital=100_000.0,
                   sizer=TargetWeightSizer(1.0 / max(top_n, 1)))
    strat = CrossSectionalMomentumStrategy(
        data, membership=membership, top_n=top_n,
        lookback=LOOKBACK, skip=skip, rebalance=REBAL)
    Backtest(data, strat, pf, NaiveExecutionHandler()).run()
    return pf


# ----------------------------------------------------- 12-1, not 12-0
def test_12_1_picks_the_riser_and_never_touches_the_spiker():
    close, volume = three_name_panel()
    pf = run_momentum(close, volume, top_n=1)

    assert pf.positions.get("RISER", 0) > 0
    assert all(f.symbol == "RISER" for f in pf.fills)
    assert pf.equity_curve().notna().all()


def test_skip_zero_is_the_off_by_a_month_and_buys_the_spike():
    """The counterfactual that proves the skip does the work: the SAME
    panel with skip=0 chases the spike at the final formation."""
    close, volume = three_name_panel()
    pf = run_momentum(close, volume, top_n=1, skip=0)

    assert any(f.symbol == "SPIKER" and f.quantity > 0 for f in pf.fills)


def test_no_signals_before_a_full_lookback_exists():
    close, volume = three_name_panel(n=280)
    data = PanelBarHandler(close, volume)
    strat = CrossSectionalMomentumStrategy(data, top_n=1)
    q = queue.Queue()
    for _ in range(LOOKBACK):            # bars 1..252: window is short
        data.update_bars(q)
        event = q.get()
        strat.on_market(event, q)
        assert q.empty()                 # not one signal before the history


# ------------------------------------------------------------ membership
def test_membership_gates_selection():
    """The best score in the panel is worthless if the name is not a
    point-in-time member on the formation date."""
    close, volume = three_name_panel()
    membership = pd.DataFrame(True, index=close.index, columns=close.columns)
    membership["RISER"] = False          # the winner is not in the index

    pf = run_momentum(close, volume, top_n=1, membership=membership)

    assert pf.positions.get("RISER", 0) == 0
    assert pf.positions.get("FLAT", 0) > 0     # best eligible name wins


# ---------------------------------------------------------------- corpse
def test_a_dead_pick_still_receives_its_exit_signal():
    """The chosen name dies between formations. It prints no bar, so a
    symbols()-only exit rule would orphan it; the strategy tracks its own
    last selection and flattens it anyway."""
    n = 340
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    riser = pd.Series(100.0 * (1.003 ** np.arange(n)), index=idx)
    riser.iloc[290:] = np.nan            # dies after bar 290
    flat = pd.Series(100.0, index=idx)
    close = pd.DataFrame({"RISER": riser, "FLAT": flat})
    volume = pd.DataFrame(1e6, index=idx, columns=close.columns)
    volume["RISER"] = volume["RISER"].where(close["RISER"].notna())

    pf = run_momentum(close, volume, top_n=1)

    assert pf.positions.get("RISER", 0) == 0          # sold, not orphaned
    sells = [f for f in pf.fills if f.symbol == "RISER" and f.quantity < 0]
    assert len(sells) == 1
    assert sells[0].fill_price == pytest.approx(
        float(riser.dropna().iloc[-1]))               # the last known price
    assert pf.equity_curve().notna().all()


# ------------------------------------------------------ the rebalance band
def test_rebalance_band_suppresses_dust_orders():
    """FixedFractionalSizer on a drifting price retargets by a few shares
    every bar. With the band those dust orders are skipped; without it they
    trade. Same data, one knob."""
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    price = 100.0 + np.arange(n)         # drifts 1 dollar a bar
    frame = pd.DataFrame({"open": price, "high": price + 1,
                          "low": price - 1, "close": price,
                          "volume": np.full(n, 1e6)}, index=idx)

    def fills_with(band):
        data = HistoricBarHandler(frame, "TEST")
        pf = Portfolio(initial_capital=100_000.0,
                       sizer=FixedFractionalSizer(1.0),
                       min_trade_fraction=band)
        Backtest(data, BuyAndHoldStrategy(data), pf,
                 NaiveExecutionHandler()).run()
        return pf

    assert len(fills_with(0.0).fills) == 1     # one signal, one fill:
    assert len(fills_with(0.02).fills) == 1    # the band never blocks a
                                               # real position change

    # The dust case the band exists for: a strategy that re-signals +1
    # every bar while FixedFractionalSizer's target drifts by ~10 shares
    # (about 1% of equity) as the price climbs.
    from backtester.strategy import Strategy
    from backtester.events import SignalEvent

    class SignalEveryBar(Strategy):
        def __init__(self, data):
            self.data = data

        def on_market(self, event, events):
            events.put(SignalEvent(timestamp=event.timestamp,
                                   symbol="TEST", direction=1))

    def fills_dust(band):
        data = HistoricBarHandler(frame, "TEST")
        pf = Portfolio(initial_capital=100_000.0,
                       sizer=FixedFractionalSizer(1.0),
                       min_trade_fraction=band)
        Backtest(data, SignalEveryBar(data), pf,
                 NaiveExecutionHandler()).run()
        return pf

    free = fills_dust(0.0)
    banded = fills_dust(0.02)

    assert len(free.fills) == 30               # dust trades every single bar
    # The band does not freeze the book -- suppressed drift ACCUMULATES and
    # trades one material order when it crosses the threshold. Batching, not
    # blocking, is the point of the knob.
    assert 1 < len(banded.fills) < len(free.fills) / 2
    for f in banded.fills[1:]:                 # every post-entry trade is
        assert abs(f.quantity) * f.fill_price >= 0.02 * 100_000 * 0.99
                                               # material: >= band * equity
