import numpy as np
import pandas as pd
import pytest

from backtester.crsp import (
    delisting_summary,
    equal_weight_index,
    membership_mask,
    to_engine_frame,
    validate_engine_frame,
)
from backtester.data import HistoricBarHandler


def make_ciz(n=40, permno=10107, start="2015-01-02", ret=0.001, div=0.0,
             price=100.0, volume=1_000_000, split_on=None):
    """CIZ-shaped daily rows for one permno.

    Deliberately built the way WRDS serves them, which is the way that breaks
    naive code: RAW prices that fall on a split and lag on a dividend, TOTAL
    returns that do neither, and a null return on the security's first day.

    `div` is a daily dividend yield, so the raw price compounds at `ret - div`
    while `dlyret` compounds at `ret`.
    """
    dates = pd.bdate_range(start, periods=n)
    rets = np.full(n, ret)
    rets[0] = np.nan                       # CRSP has no return on day one
    raw = np.full(n, price, dtype=float)
    vol = np.full(n, volume, dtype=float)
    for i in range(1, n):
        raw[i] = raw[i - 1] * (1 + ret - div)
    if split_on is not None:               # 2-for-1 on bar `split_on`
        raw[split_on:] /= 2.0
        vol[split_on:] *= 2.0
    return pd.DataFrame({
        "permno": permno,
        "dlycaldt": dates,
        "dlyopen": raw * 0.995,
        "dlyhigh": raw * 1.005,
        "dlylow": raw * 0.990,
        "dlyclose": raw,
        "dlyvol": vol,
        "dlyret": rets,
        "dlydelflg": "N",
    })


# ------------------------------------------------------------------ shape
def test_output_is_what_the_engine_already_eats():
    frame = to_engine_frame(make_ciz())
    HistoricBarHandler(frame, "TEST")      # constructor is the contract test


def test_missing_ciz_column_is_rejected():
    rows = make_ciz().drop(columns=["dlyret"])
    with pytest.raises(ValueError, match="dlyret"):
        to_engine_frame(rows)


def test_two_permnos_without_a_selector_is_rejected():
    rows = pd.concat([make_ciz(permno=10107), make_ciz(permno=14593)])
    with pytest.raises(ValueError, match="one permno"):
        to_engine_frame(rows)


def test_selecting_a_permno_from_a_stacked_frame():
    rows = pd.concat([make_ciz(permno=10107), make_ciz(permno=14593, ret=0.002)])
    frame = to_engine_frame(rows, permno=14593)
    assert frame.attrs["permno"] == 14593


def test_unsorted_input_comes_back_sorted():
    rows = make_ciz().sample(frac=1.0, random_state=0)
    frame = to_engine_frame(rows)
    assert frame.index.is_monotonic_increasing


# ------------------------------------------------------- the return series
def test_close_compounds_dlyret_not_dlyclose():
    rows = make_ciz(n=30, ret=0.001)
    frame = to_engine_frame(rows)
    expected = rows["dlyclose"].iloc[0] * (1.001 ** 29)
    assert frame["close"].iloc[-1] == pytest.approx(expected)


def test_a_split_does_not_show_up_as_a_return():
    """The one that makes the whole module necessary.

    Raw CRSP prices halve on a 2-for-1. Feed them to the engine and the
    strategy sees a -50% day that never happened.
    """
    rows = make_ciz(n=30, ret=0.001, split_on=15)
    frame = to_engine_frame(rows)
    daily = frame["close"].pct_change().dropna()
    assert daily.min() > 0                      # no fabricated crash
    assert daily.max() == pytest.approx(0.001)

    raw_daily = rows["dlyclose"].pct_change().dropna()
    assert raw_daily.min() == pytest.approx(-0.4995)   # what you would have got


def test_first_bar_anchors_to_the_raw_close():
    rows = make_ciz(price=250.0)
    assert to_engine_frame(rows)["close"].iloc[0] == pytest.approx(250.0)


# ------------------------------------------------------- dollar volume law
def test_dollar_volume_is_invariant():
    """close * volume == dlyclose * dlyvol, on EVERY bar.

    Share counts depend on the price basis; dollars do not. This is the
    conservation test for the same class of bug as cash conservation.
    """
    rows = make_ciz(n=60, ret=0.002, div=0.0002, split_on=30)
    frame = to_engine_frame(rows)
    got = frame["close"].to_numpy() * frame["volume"].to_numpy()
    want = rows["dlyclose"].to_numpy() * rows["dlyvol"].to_numpy()
    assert got == pytest.approx(want)


def test_dividends_rescale_volume_too():
    """The yfinance failure mode, reproduced.

    A back-adjusted price series and an untouched volume series multiply to a
    dollar volume that never existed. Here the adapter shrinks volume as the
    total-return index outgrows the raw price, so the product still holds.
    """
    rows = make_ciz(n=60, ret=0.002, div=0.0002)     # a dividend payer
    frame = to_engine_frame(rows)

    assert frame["volume"].iloc[0] == pytest.approx(rows["dlyvol"].iloc[0])
    assert frame["volume"].iloc[-1] < rows["dlyvol"].iloc[-1] * 0.99

    got = frame["close"].to_numpy() * frame["volume"].to_numpy()
    want = rows["dlyclose"].to_numpy() * rows["dlyvol"].to_numpy()
    assert got == pytest.approx(want)


def test_a_dividend_is_a_return_not_a_price_move():
    """Total return outruns the raw price by exactly the dividend stream."""
    rows = make_ciz(n=60, ret=0.002, div=0.0002)
    frame = to_engine_frame(rows)
    tr = frame["close"].iloc[-1] / frame["close"].iloc[0]
    px = rows["dlyclose"].iloc[-1] / rows["dlyclose"].iloc[0]
    assert tr > px
    assert tr / px == pytest.approx((1.002 / 1.0018) ** 59)


# ------------------------------------------------------------- delistings
def test_delisting_return_reaches_the_price_series():
    rows = make_ciz(n=20)
    rows.loc[rows.index[-1], "dlyret"] = -1.0
    rows.loc[rows.index[-1], "dlydelflg"] = "Y"

    frame = to_engine_frame(rows)
    assert frame["close"].iloc[-1] < frame["close"].iloc[-2] * 1e-6
    assert frame.attrs["delisted"] is True


def test_a_hundred_percent_loss_does_not_produce_a_zero_price():
    """-100% is a valid return and an invalid price. trailing_volatility takes
    a log; log(0) is -inf and every downstream number becomes NaN."""
    rows = make_ciz(n=20)
    rows.loc[rows.index[-1], "dlyret"] = -1.0
    frame = to_engine_frame(rows)
    assert frame["close"].iloc[-1] > 0
    assert np.isfinite(np.log(frame["close"].to_numpy())).all()


def test_delisting_summary_finds_the_flagged_row():
    rows = make_ciz(n=20)
    rows.loc[rows.index[-1], "dlyret"] = -0.87
    rows.loc[rows.index[-1], "dlydelflg"] = "Y"

    hits = delisting_summary(rows)
    assert len(hits) == 1
    assert hits["dlyret"].iloc[0] == pytest.approx(-0.87)


def test_delisting_summary_refuses_non_ciz_input():
    with pytest.raises(ValueError, match="not CIZ"):
        delisting_summary(make_ciz().drop(columns=["dlydelflg"]))


# ------------------------------------------------------------- validation
def test_validator_rejects_a_nan():
    frame = to_engine_frame(make_ciz())
    frame.iloc[3, frame.columns.get_loc("close")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_engine_frame(frame)


def test_validator_rejects_a_negative_price():
    """SIZ used a negative sign to mean 'this is a bid/ask average, not a
    trade'. Whatever the current convention is, assert it rather than read it."""
    frame = to_engine_frame(make_ciz())
    frame.iloc[3, frame.columns.get_loc("close")] = -101.0
    with pytest.raises(ValueError, match="non-positive price"):
        validate_engine_frame(frame)


def test_validator_rejects_negative_volume():
    frame = to_engine_frame(make_ciz())
    frame.iloc[3, frame.columns.get_loc("volume")] = -1.0
    with pytest.raises(ValueError, match="negative volume"):
        validate_engine_frame(frame)


# -------------------------------------------------------- the measurement
def _universe():
    dates = pd.bdate_range("2015-01-02", periods=10)
    # SURVIVOR returns +0.2%/day for the whole window.
    # CASUALTY does the same, then loses everything on the last day.
    returns = pd.DataFrame({
        101: [0.002] * 10,
        202: [0.002] * 9 + [-1.0],
    }, index=dates)
    return dates, returns


def test_survivor_only_basket_beats_the_full_one():
    """The whole thesis of the project, in four lines and no engine."""
    dates, returns = _universe()
    survivor_only = equal_weight_index(returns[[101]])
    everyone = equal_weight_index(returns)
    assert survivor_only.iloc[-1] > everyone.iloc[-1]


def test_membership_mask_is_point_in_time():
    dates, returns = _universe()
    membership = pd.DataFrame({
        "permno": [101, 202],
        "mbrstartdt": [dates[0], dates[0]],
        "mbrenddt": [dates[-1], dates[4]],     # 202 leaves the index early
    })
    mask = membership_mask(dates, returns.columns, membership)
    assert mask.loc[dates[0], 202]
    assert not mask.loc[dates[-1], 202]
    assert mask[101].all()


def test_leaving_the_index_before_the_blowup_hides_it():
    """Point-in-time membership is not automatically more conservative. It is
    more CORRECT, which is a different claim and the one to make out loud."""
    dates, returns = _universe()
    membership = pd.DataFrame({
        "permno": [101, 202],
        "mbrstartdt": [dates[0], dates[0]],
        "mbrenddt": [dates[-1], dates[4]],
    })
    mask = membership_mask(dates, returns.columns, membership)
    pit = equal_weight_index(returns, mask)
    naive = equal_weight_index(returns)
    assert pit.iloc[-1] > naive.iloc[-1]


def test_an_empty_basket_is_flat_not_nan():
    dates, returns = _universe()
    mask = pd.DataFrame(False, index=dates, columns=returns.columns)
    curve = equal_weight_index(returns, mask)
    assert curve.iloc[-1] == pytest.approx(1.0)


# ------------------------------------------------------------ integration
def test_a_crsp_frame_runs_through_the_engine_and_pays_impact():
    """Day 7's lesson, applied to Day 8's data. Unit tests prove a component
    is correct; only this proves the new data path is CONNECTED."""
    from backtester.costs import HalfSpreadSlippage, PerShareCommission, SquareRootImpact
    from backtester.engine import Backtest
    from backtester.execution import SimulatedExecutionHandler
    from backtester.portfolio import Portfolio
    from backtester.strategy import BuyAndHoldStrategy

    frame = to_engine_frame(make_ciz(n=120, ret=0.001, div=0.0002, split_on=60))
    data = HistoricBarHandler(frame, "TEST")
    pf = Portfolio(initial_capital=1e9)
    execution = SimulatedExecutionHandler(
        commission=PerShareCommission(),
        slippage=HalfSpreadSlippage(spread_bps=1.0),
        impact=SquareRootImpact(1.0),
    )
    Backtest(data, BuyAndHoldStrategy(data, warmup=21), pf, execution).run()

    assert len(pf.fills) == 1
    assert pf.fills[0].impact_cost > 0
    assert pf.equity_curve().notna().all()