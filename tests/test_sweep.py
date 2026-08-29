"""Day 11, Step 1: the sweep runner and the multiple-testing math.

Three things are load-bearing. The cache must round-trip (a second sweep of
the same grid runs zero backtests -- and the cache is also the trial
registry, so what it remembers decides what the DSR forgives). The
rebalance knob must actually reach the strategy (a cadence nobody varies is
a constant wearing a parameter's costume). And the deflated Sharpe must
tell the best of a hundred lottery tickets from a real edge -- that
distinction is the entire point of Day 11.
"""
import numpy as np
import pandas as pd
import pytest

from backtester.execution import NaiveExecutionHandler
from backtester.metrics import deflated_sharpe, expected_max_sharpe
from backtester.walkforward import Config, run_window
from sweep_report import registry_trial_count, sweep

IDX = pd.date_range("2020-01-02", "2021-12-31", freq="B")


def leadership_flip_panel():
    """A leads for the first half, then rolls over while B takes off. A
    ranking that never changes cannot distinguish rebalance cadences; this
    one flips exactly once, so faster re-ranking reacts sooner."""
    n = len(IDX)
    half = n // 2
    a = np.empty(n)
    a[:half] = 100.0 * 1.004 ** np.arange(half)
    a[half:] = a[half - 1] * 0.997 ** np.arange(1, n - half + 1)
    b = np.empty(n)
    b[:half] = 100.0
    b[half:] = 100.0 * 1.004 ** np.arange(1, n - half + 1)
    close = pd.DataFrame({"A": a, "B": b}, index=IDX)
    volume = pd.DataFrame(1e6, index=IDX, columns=close.columns)
    return close, volume


# -------------------------------------------------------------- the cache
def test_sweep_caches_and_a_second_pass_runs_nothing(tmp_path):
    """Round-trip: every config runs once, lands in the CSV, and the same
    request again is served entirely from disk with identical numbers. The
    request deliberately lists one config twice -- SURFACE and BREADTH
    genuinely overlap on a cell, and the overlap must run once, not twice
    (the bug that crashed the first real sweep of Day 11)."""
    close, volume = leadership_flip_panel()
    cache = tmp_path / "cache.csv"
    configs = (Config(21, 0, 1, rebalance=5), Config(21, 0, 1, rebalance=63),
               Config(21, 0, 1, rebalance=5))         # the duplicate

    df1, ran1 = sweep(close, volume, None, configs,
                      start=IDX[30], end=IDX[-1], tag="t", capital=1e6,
                      execution_factory=NaiveExecutionHandler,
                      cache_path=cache)
    assert ran1 == 2                                  # duplicate ran once
    assert len(df1) == 3                              # but three rows back
    assert cache.exists()

    df2, ran2 = sweep(close, volume, None, configs,
                      start=IDX[30], end=IDX[-1], tag="t", capital=1e6,
                      execution_factory=NaiveExecutionHandler,
                      cache_path=cache)
    assert ran2 == 0                                  # zero backtests run
    assert np.allclose(df1["sharpe"], df2["sharpe"])
    assert list(df2["rebalance"]) == [5, 63, 5]       # rows in config order

    # rebalance is part of the key: two distinct rows on disk, no third
    on_disk = pd.read_csv(cache)
    assert len(on_disk) == 2
    assert sorted(on_disk["rebalance"]) == [5, 63]


def test_registry_counts_every_trial_plus_the_walkforwards_twelve(tmp_path):
    """The DSR denominator: Day 10's twelve configs happened whether or not
    this file remembers them, and a cached re-request is not a new look."""
    cache = tmp_path / "cache.csv"
    assert registry_trial_count(cache) == 12          # empty registry

    close, volume = leadership_flip_panel()
    configs = (Config(21, 0, 1, rebalance=5), Config(21, 0, 1, rebalance=63))
    for _ in range(2):                                # second pass is cached
        sweep(close, volume, None, configs,
              start=IDX[30], end=IDX[-1], tag="t", capital=1e6,
              execution_factory=NaiveExecutionHandler, cache_path=cache)
        assert registry_trial_count(cache) == 2 + 12


# ------------------------------------------------------ the rebalance knob
def test_config_rebalance_reaches_the_strategy():
    """Weekly re-ranking catches the mid-window leadership flip months
    before quarterly does, so the curves must differ -- and the default must
    reproduce Day 10's fixed cadence of 21 exactly."""
    close, volume = leadership_flip_panel()
    start, end = IDX[30], IDX[-1]

    def run(config):
        return run_window(close, volume, None, config, start, end,
                          execution_factory=NaiveExecutionHandler)

    weekly = run(Config(21, 0, 1, rebalance=5))
    quarterly = run(Config(21, 0, 1, rebalance=63))
    assert abs(weekly.iloc[-1] - quarterly.iloc[-1]) > 0.01

    assert run(Config(21, 0, 1)).equals(run(Config(21, 0, 1, rebalance=21)))


# ------------------------------------------- expected max of N noise trials
def test_expected_max_sharpe_is_monotone_in_trials_and_zero_for_one():
    v = 0.001
    assert expected_max_sharpe(1, v) == 0.0           # one trial: no search
    assert expected_max_sharpe(16, 0.0) == 0.0        # no spread: no max lift
    vals = [expected_max_sharpe(n, v) for n in (2, 4, 16, 64, 256)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    with pytest.raises(ValueError):
        expected_max_sharpe(0, v)


# --------------------------------------------------- the deflated Sharpe
def test_deflated_sharpe_tells_noise_from_edge():
    """Best of 100 pure-noise trials: a ~0.99 'discovery' if you hide the
    trial count, ~0.5 once you confess it. A true per-period Sharpe of 0.3
    survives the same deflation at ~1.0. This asymmetry is the reason the
    cache doubles as a registry."""
    rng = np.random.default_rng(11)
    n_trials, t = 100, 1000
    rets = rng.normal(0.0, 0.01, size=(n_trials, t))
    srs = rets.mean(axis=1) / rets.std(axis=1, ddof=1)
    best = pd.Series(rets[int(np.argmax(srs))])
    var_trials = float(srs.var(ddof=1))

    confessed = deflated_sharpe(best, n_trials, var_trials)
    hidden = deflated_sharpe(best, 1, var_trials)
    assert 0.2 < confessed < 0.7                      # what noise looks like
    assert hidden > 0.95                              # what hiding N buys
    assert confessed < hidden

    genuine = pd.Series(rng.normal(0.003, 0.01, size=t))
    assert deflated_sharpe(genuine, n_trials, var_trials) > 0.999


def test_deflated_sharpe_refuses_degenerate_inputs():
    flat = pd.Series([0.01, 0.01, 0.01, 0.01])        # zero volatility
    assert np.isnan(deflated_sharpe(flat, 10, 0.001))
    short = pd.Series([0.01, -0.02])                  # t < 3
    assert np.isnan(deflated_sharpe(short, 10, 0.001))
