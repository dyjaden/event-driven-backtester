"""Day 9, Step 2: the panel handler, and the four tests that earn it.

The regression test makes the old single-symbol world the known answer for
the new one. The two-priced book kills the holdings_value bug specifically
(a panel of clones cannot: identical prices make that bug accidentally
correct). The clone basket checks weights and accounting. The corpse checks
the delisting policy end to end: exit at the last known price, zero impact,
no NaN anywhere in the curve.
"""
import queue

import numpy as np
import pandas as pd
import pytest

from backtester.costs import SquareRootImpact
from backtester.crsp import to_panel
from backtester.data import HistoricBarHandler, PanelBarHandler
from backtester.engine import Backtest
from backtester.events import SignalEvent
from backtester.execution import NaiveExecutionHandler, SimulatedExecutionHandler
from backtester.portfolio import FixedQuantitySizer, Portfolio, TargetWeightSizer
from backtester.strategy import BuyAndHoldStrategy, Strategy

from make_fake_crsp import fake_universe


# ------------------------------------------------------------------ helpers
def make_frame(n=60, start=100.0, step=1.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.arange(n, dtype=float) * step + start
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": np.full(n, 1e6)},
        index=idx,
    )


def panel_of(**series):
    """Build (close, volume) frames from {symbol: (close_list, volume_list)}."""
    n = max(len(c) for c, _ in series.values())
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.DataFrame(index=idx)
    volume = pd.DataFrame(index=idx)
    for sym, (c, v) in series.items():
        close[sym] = pd.Series(list(c) + [np.nan] * (n - len(c)), index=idx)
        volume[sym] = pd.Series(list(v) + [np.nan] * (n - len(v)), index=idx)
    return close, volume


class BuyNamed(Strategy):
    """Test instrument: goes long every named symbol on the first bar."""

    def __init__(self, data, symbols):
        self.data = data
        self.names = list(symbols)
        self.done = False

    def on_market(self, event, events):
        if self.done:
            return
        for s in self.names:
            events.put(SignalEvent(timestamp=event.timestamp,
                                   symbol=s, direction=1))
        self.done = True


class BuyThenDrop(BuyNamed):
    """Buys every name on bar 1; emits a flat signal for `drop` on bar `at`.

    This is the rebalance-sell of a dead name, driven the same way the
    momentum strategy will drive it: a direction-0 signal, no special path.
    """

    def __init__(self, data, symbols, drop, at):
        super().__init__(data, symbols)
        self.drop = drop
        self.at = at
        self.bars = 0

    def on_market(self, event, events):
        self.bars += 1
        super().on_market(event, events)
        if self.bars == self.at:
            events.put(SignalEvent(timestamp=event.timestamp,
                                   symbol=self.drop, direction=0))


def run_panel(close, volume, strategy_cls, execution, sizer=None,
              capital=100_000.0, **kw):
    data = PanelBarHandler(close, volume)
    pf = Portfolio(initial_capital=capital, sizer=sizer)
    Backtest(data, strategy_cls(data, **kw), pf, execution).run()
    return pf, data


# ------------------------------------------------------------ 1. regression
def test_single_column_panel_matches_historic_handler():
    """The old world is the known answer for the new one: one symbol through
    the panel must reproduce the single-symbol handler bar for bar."""
    frame = make_frame(60)

    data = HistoricBarHandler(frame, "TEST")
    pf_old = Portfolio(initial_capital=100_000.0)
    Backtest(data, BuyAndHoldStrategy(data), pf_old,
             NaiveExecutionHandler()).run()

    close = frame[["close"]].rename(columns={"close": "TEST"})
    volume = frame[["volume"]].rename(columns={"volume": "TEST"})
    pf_new, _ = run_panel(close, volume, BuyNamed, NaiveExecutionHandler(),
                          symbols=["TEST"])

    pd.testing.assert_series_equal(pf_old.equity_curve(),
                                   pf_new.equity_curve(),
                                   check_names=False)


# ------------------------------------------- 2. the two-priced book (THE bug)
def test_two_priced_book_prices_each_position_at_its_own_close():
    """One share of A at 100 and one of B at 10 is a 110-dollar book.

    The pre-refactor holdings_value multiplied ONE close across every
    position and would report 200 or 20. A panel of clones cannot catch
    that; different prices can.
    """
    close, volume = panel_of(A=([100.0] * 20, [1e6] * 20),
                             B=([10.0] * 20, [1e6] * 20))
    pf, _ = run_panel(close, volume, BuyNamed, NaiveExecutionHandler(),
                      sizer=FixedQuantitySizer(1), symbols=["A", "B"])

    assert pf.positions == {"A": 1, "B": 1}
    assert pf.cash == pytest.approx(100_000.0 - 110.0)
    # flat prices, zero cost: swapping 110 dollars for a 110-dollar book
    # cannot move equity.
    assert pf.equity_curve().iloc[-1] == pytest.approx(100_000.0)


# --------------------------------------------------------- 3. clone basket
def test_clone_basket_matches_the_single_name():
    """Three copies of the same series at a third each equals one copy at
    full weight, to integer-share tolerance. This checks the weights and
    the accounting; the two-priced test above checks the pricing."""
    frame = make_frame(60)

    data = HistoricBarHandler(frame, "TEST")
    pf_one = Portfolio(initial_capital=100_000.0,
                       sizer=TargetWeightSizer(1.0))
    Backtest(data, BuyAndHoldStrategy(data), pf_one,
             NaiveExecutionHandler()).run()

    c = frame["close"]
    v = frame["volume"]
    close = pd.DataFrame({"A": c, "B": c, "C": c})
    volume = pd.DataFrame({"A": v, "B": v, "C": v})
    pf_three, _ = run_panel(close, volume, BuyNamed, NaiveExecutionHandler(),
                            sizer=TargetWeightSizer(1.0 / 3.0),
                            symbols=["A", "B", "C"])

    one = pf_one.equity_curve()
    three = pf_three.equity_curve()
    assert three.notna().all()
    assert np.isclose(three.iloc[-1], one.iloc[-1], rtol=1e-2)


# --------------------------------------------------------------- 4. corpse
def test_corpse_exits_at_last_price_with_zero_impact():
    """B dies at 1.00 on bar 30 of 60 (its -98% is in the closes) and is
    sold by an ordinary direction-0 signal 25 bars later, when every
    trailing statistic is NaN. The fill must happen at the last known
    price, pay zero impact, and leave no NaN anywhere in the curve."""
    a = [100.0] * 60
    b = [50.0] * 29 + [1.0]                 # bar 30 is the grave marker
    close, volume = panel_of(A=(a, [1e6] * 60),
                             B=(b, [1e6] * 30))

    execution = SimulatedExecutionHandler(impact=SquareRootImpact(1.0))
    pf, _ = run_panel(close, volume, BuyThenDrop, execution,
                      sizer=FixedQuantitySizer(100),
                      symbols=["A", "B"], drop="B", at=55)

    sells = [f for f in pf.fills if f.symbol == "B" and f.quantity < 0]
    assert len(sells) == 1
    corpse = sells[0]
    assert corpse.quantity == -100
    assert corpse.fill_price == pytest.approx(1.0)   # the last known close
    assert corpse.impact_cost == 0.0                 # NaN stats charge nothing
    assert pf.positions["B"] == 0

    curve = pf.equity_curve()
    assert curve.notna().all()                       # the NaN-poison guard
    # 100 sh of A (flat) plus 100 sh of B bought at 50 and worth 1.00:
    # the 4,900-dollar loss is real, and the exit does not change equity.
    assert curve.iloc[-1] == pytest.approx(100_000.0 - 100 * 49.0)


def test_dead_name_contract_mid_run():
    """The handler's corpse vocabulary, checked directly."""
    b = [50.0] * 29 + [1.0]
    close, volume = panel_of(A=([100.0] * 60, [1e6] * 60),
                             B=(b, [1e6] * 30))
    data = PanelBarHandler(close, volume)
    q = queue.Queue()
    for _ in range(55):                      # far enough past B's death that
        data.update_bars(q)                  # the 21-bar window is all NaN

    assert data.has_bar("A") and not data.has_bar("B")
    assert data.is_dead("B") and not data.is_dead("A")
    assert data.symbols() == ["A"]
    assert data.current_price("B") == pytest.approx(1.0)   # last known
    assert np.isnan(data.adv("B"))
    assert np.isnan(data.trailing_volatility("B"))


# ------------------------------------------------------------- validation
def test_panel_rejects_malformed_input_and_unknown_symbols():
    close, volume = panel_of(A=([100.0] * 10, [1e6] * 10))
    with pytest.raises(ValueError):
        PanelBarHandler(close, volume.rename(columns={"A": "B"}))
    with pytest.raises(ValueError):
        PanelBarHandler(close, volume.iloc[:-1])

    data = PanelBarHandler(close, volume)
    with pytest.raises(RuntimeError):
        data.current_price("A")              # no bar yet
    data.update_bars(queue.Queue())
    with pytest.raises(KeyError):
        data.current_price("ZZZ")
    with pytest.raises(ValueError):
        data.current_price()                 # a panel needs a symbol
    with pytest.raises(ValueError):
        data.current_price("A", field="open")


# ------------------------------------------------------------- to_panel
def test_to_panel_invariants_on_fabricated_ciz():
    """to_panel on the fixture universe: membership filter, str columns,
    per-bar dollar-volume invariance, delisting compounded into the price,
    and the SVB case (flag with no return) priced flat into the grave."""
    dates = pd.date_range("2015-01-02", periods=120, freq="B")
    uni, members = fake_universe(dates, n=12, dead=3, svb=1)

    close, volume = to_panel(uni, members)

    assert list(close.columns) == [str(p) for p in sorted(uni["permno"].unique())]
    assert isinstance(close.index, pd.DatetimeIndex)

    # dollar volume invariant, on every bar where both worlds have data
    raw = uni.pivot(index="dlycaldt", columns="permno", values="dlyclose")
    rawv = uni.pivot(index="dlycaldt", columns="permno", values="dlyvol")
    raw.columns = raw.columns.map(str)
    rawv.columns = rawv.columns.map(str)
    lhs = (close * volume).to_numpy()
    rhs = (raw * rawv).to_numpy()
    both = np.isfinite(lhs) & np.isfinite(rhs)
    assert both.any()
    np.testing.assert_allclose(lhs[both], rhs[both], rtol=1e-9)

    # the -93% deaths are inside the price series, then NaN forever
    dead_permno = str(uni.loc[uni["dlydelflg"] == "Y", "permno"].iloc[-1])
    col = close[dead_permno].dropna()
    assert col.iloc[-1] == pytest.approx(col.iloc[-2] * (1 - 0.93), rel=1e-9)
    assert close[dead_permno].iloc[-1] is not None
    assert np.isnan(close[dead_permno].iloc[-1])

    # the SVB case: delisted with NO return -- flat into the grave, not NaN
    svb_permno = str(uni.loc[uni["dlydelflg"] == "Y", "permno"].iloc[0])
    svb = close[svb_permno].dropna()
    assert svb.iloc[-1] == pytest.approx(svb.iloc[-2])

    # membership filter: a permno missing from members has no column
    cut = members.iloc[1:]
    close2, _ = to_panel(uni, cut)
    assert str(members["permno"].iloc[0]) not in close2.columns


def test_to_panel_feeds_the_handler():
    """End to end: fixture rows -> panel -> handler -> a priced book."""
    dates = pd.date_range("2015-01-02", periods=90, freq="B")
    uni, members = fake_universe(dates, n=6, dead=1)
    close, volume = to_panel(uni, members)

    alive = [c for c in close.columns if pd.notna(close[c].iloc[-1])][:3]
    pf, data = run_panel(close, volume, BuyNamed, NaiveExecutionHandler(),
                         sizer=TargetWeightSizer(1.0 / 3.0), symbols=alive)

    curve = pf.equity_curve()
    assert len(curve) == len(close)
    assert curve.notna().all()
    assert set(pf.positions) == set(alive)
