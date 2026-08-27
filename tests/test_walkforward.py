"""Day 10, Step 1: the walk-forward framework's tests.

The one that matters is the poisoned-test test: make the test window absurd
and assert the selection cannot tell. Selection that can see the test set is
the entire disease the framework exists to prevent.
"""
import numpy as np
import pandas as pd
import pytest

from backtester.execution import NaiveExecutionHandler
from backtester.metrics import sharpe_ratio
from backtester.walkforward import (Config, calendar_folds, embargoed_sharpe,
                                    run_window, select_on_train, stitch)

IDX = pd.date_range("2015-01-02", "2025-12-31", freq="B")


# ------------------------------------------------------------------- folds
def test_calendar_folds_are_disjoint_ordered_and_cover_2018_to_2025():
    folds = calendar_folds(IDX, train_years=3, test_years=1)

    assert len(folds) == 8
    assert [f.test_start.year for f in folds] == list(range(2018, 2026))
    for f in folds:
        assert f.train_start < f.train_end < f.test_start < f.test_end
        assert f.test_start.year - f.train_start.year == 3   # 3y of training
    for a, b in zip(folds, folds[1:]):
        assert b.test_start > a.test_end or b.test_start.year == a.test_end.year + 1


def test_folds_refuse_an_index_that_is_too_short():
    short = pd.date_range("2020-01-02", "2021-12-31", freq="B")
    with pytest.raises(ValueError):
        calendar_folds(short, train_years=3, test_years=1)


# -------------------------------------------------------------- run_window
def two_name_panel(n=len(IDX)):
    """UP compounds gently forever; FLAT sits still. Enough history that any
    lookback in the test grids has context before every window."""
    idx = IDX[:n]
    up = 100.0 * (1.0015 ** np.arange(len(idx)))
    close = pd.DataFrame({"UP": up, "FLAT": np.full(len(idx), 100.0)},
                         index=idx)
    volume = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return close, volume


def test_run_window_rebases_and_measures_only_the_window():
    close, volume = two_name_panel()
    start, end = pd.Timestamp("2017-01-02"), pd.Timestamp("2017-12-29")
    curve = run_window(close, volume, None, Config(126, 21, 1), start, end,
                       execution_factory=NaiveExecutionHandler)

    assert curve.iloc[0] == pytest.approx(1.0)       # rebased
    assert curve.index[0] >= start and curve.index[-1] <= end
    assert curve.notna().all()
    # formations exist from bar one (the lookback context precedes the
    # window), so the book is long UP from the start and the curve grows
    assert curve.iloc[-1] > 1.10


# ------------------------------------------------------------------ stitch
def test_stitch_compounds_and_never_adds():
    a = pd.Series([1.0, 1.1],
                  index=pd.to_datetime(["2018-01-02", "2018-12-28"]))
    b = pd.Series([1.0, 1.2],
                  index=pd.to_datetime(["2019-01-02", "2019-12-27"]))
    out = stitch([a, b])

    assert out.iloc[-1] == pytest.approx(1.1 * 1.2)   # 1.32, not 0.30
    assert list(out.index) == list(a.index) + list(b.index)
    with pytest.raises(ValueError):
        stitch([b, a])                                # out of order


# ----------------------------------------------------------------- embargo
def test_selection_metric_is_the_embargoed_sharpe():
    """The score used for selection equals the Sharpe of the train curve
    with its final embargo bars cut -- computed here independently."""
    close, volume = two_name_panel()
    fold = calendar_folds(close.index, 1, 1)[1]
    grid = (Config(63, 5, 1),)

    _, scores = select_on_train(close, volume, None, fold, grid,
                                embargo_bars=21,
                                execution_factory=NaiveExecutionHandler)

    curve = run_window(close, volume, None, grid[0],
                       fold.train_start, fold.train_end,
                       execution_factory=NaiveExecutionHandler)
    expected = sharpe_ratio(curve.iloc[:-21].pct_change().dropna())
    assert scores[grid[0]] == pytest.approx(expected)
    assert embargoed_sharpe(curve, 21) == pytest.approx(expected)


# ------------------------------------------------- THE poisoned-test test
def test_a_poisoned_test_window_cannot_change_the_selection():
    """Selection must read only the train window. The test year is made
    absurd (+30% a day, a printing press) and the choice and every score
    must come back identical. If this test ever fails, the walk-forward is
    an in-sample backtest wearing an out-of-sample costume."""
    close, volume = two_name_panel()
    fold = calendar_folds(close.index, 2, 1)[0]      # train 15-16, test 17
    grid = (Config(63, 5, 1), Config(63, 0, 1), Config(126, 21, 1))

    chosen_clean, scores_clean = select_on_train(
        close, volume, None, fold, grid,
        execution_factory=NaiveExecutionHandler)

    poisoned = close.copy()
    test_rows = poisoned.index >= fold.test_start
    poisoned.loc[test_rows, "FLAT"] = (
        100.0 * (1.30 ** np.arange(test_rows.sum())))
    chosen_poisoned, scores_poisoned = select_on_train(
        poisoned, volume, None, fold, grid,
        execution_factory=NaiveExecutionHandler)

    assert chosen_poisoned == chosen_clean
    for config in grid:
        assert scores_poisoned[config] == pytest.approx(scores_clean[config])
