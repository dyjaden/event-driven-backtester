import numpy as np
import pandas as pd
import pytest

from backtester import metrics as m


def idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D")


def test_simple_vs_log():
    p = pd.Series([100.0, 110.0, 121.0], index=idx(3))
    s, l = m.simple_returns(p), m.log_returns(p)
    assert np.allclose(s.values, [0.10, 0.10])
    assert np.allclose(l.values, [np.log(1.1), np.log(1.1)])
    # log returns ADD across time, simple returns COMPOUND
    assert np.isclose(l.sum(), np.log(1.21))
    assert np.isclose((1 + s).prod() - 1, 0.21)


def test_gap_is_not_a_fabricated_flat_day():
    """A missing price must vanish, not pad forward into a fake 0% return."""
    p = pd.Series([100.0, np.nan, 110.0, 121.0], index=idx(4))
    r = m.simple_returns(p)
    assert len(r) == 1                      # the gap swallows both adjacent returns
    assert np.isclose(r.iloc[0], 0.10)      # only 121/110 survives
    assert not (r == 0.0).any()             # nothing fabricated


def test_total_and_cagr():
    p = pd.Series([100.0, 121.0], index=idx(2))
    assert np.isclose(m.total_return(p), 0.21)
    assert np.isclose(m.cagr(p, 252), 1.21 ** 252 - 1)
    # exactly one year of data: CAGR must equal total return
    p2 = pd.Series(np.linspace(100, 110, 253), index=idx(253))
    assert np.isclose(m.cagr(p2, 252), 0.10, atol=1e-9)


def test_sharpe_hand_computed():
    r = pd.Series([0.01, -0.01, 0.02, 0.00, 0.01])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert np.isclose(m.sharpe_ratio(r), expected)


def test_sharpe_zero_vol_is_nan():
    """A riskless series has no Sharpe. Must not return a huge number."""
    r = pd.Series([0.01] * 10)
    assert np.isnan(m.sharpe_ratio(r))


def test_drawdown_hand_computed():
    eq = pd.Series([100.0, 120.0, 90.0, 110.0], index=idx(4))
    assert np.isclose(m.max_drawdown(eq), -0.25)     # 120 -> 90
    ds = m.drawdown_series(eq)
    assert np.isclose(ds.iloc[0], 0.0)
    assert np.isclose(ds.iloc[2], -0.25)


def test_drawdown_is_path_dependent():
    """Same values, different order, different answer. This is the point."""
    a = pd.Series([100.0, 120.0, 90.0, 110.0])
    b = pd.Series([100.0, 90.0, 120.0, 110.0])
    assert m.max_drawdown(a) != m.max_drawdown(b)


def test_hit_rate():
    r = pd.Series([0.01, -0.01, 0.02, 0.00])
    assert np.isclose(m.hit_rate(r), 0.5)   # a flat day is not a win


def test_summary_keys():
    rng = np.random.default_rng(0)
    eq = pd.Series(np.cumprod(1 + rng.normal(0.0003, 0.01, 300)) * 100, index=idx(300))
    s = m.summary(eq)
    assert set(s) == {"total_return", "cagr", "ann_volatility", "sharpe",
                      "max_drawdown", "calmar", "hit_rate", "n_periods"}
    assert -1 <= s["max_drawdown"] <= 0