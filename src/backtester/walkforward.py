"""Walk-forward validation. Parameters chosen on the past, judged on data
the choice never saw.

The structural rule that everything here serves: `select_on_train` never
receives a test window. It takes a Fold and reads ONLY the train fields, and
`tests/test_walkforward.py` poisons a test window to prove the selection
cannot notice. A walk-forward whose selection can see the test set is an
in-sample backtest wearing an out-of-sample costume, which is the disease
this module exists to prevent (Lopez de Prado, ch. 11).

One path through history, not many: CPCV (ch. 12) is the stronger design
and is named future work rather than silently skipped.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .costs import HalfSpreadSlippage, PerShareCommission, SquareRootImpact
from .data import PanelBarHandler
from .engine import Backtest
from .execution import SimulatedExecutionHandler
from .metrics import sharpe_ratio
from .portfolio import Portfolio, TargetWeightSizer
from .strategy import CrossSectionalMomentumStrategy

REBALANCE = 21          # fixed across the grid; Week 11 varies it deliberately
BAND = 0.001


@dataclass(frozen=True)
class Config:
    """The momentum knobs. Day 10 varied the first three; Day 11's sweep
    adds the rebalance cadence, defaulted so every Day 10 call is
    unchanged."""
    lookback: int
    skip: int
    top_n: int
    rebalance: int = REBALANCE


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


DEFAULT_CONFIG = Config(lookback=252, skip=21, top_n=50)

# Twelve configurations, frozen before the first run. Two are deliberate
# traps: skip=0 is the off-by-a-month, top_n=20 is concentration. If a train
# window picks one, that is selection doing what selection does, and it goes
# in the report -- the trial count is part of any backtest result.
GRID = tuple(Config(lookback=lb, skip=sk, top_n=n)
             for lb in (126, 252) for sk in (0, 21) for n in (20, 50, 100))


def loaded_execution() -> SimulatedExecutionHandler:
    """The fully loaded arm, identical to momentum_report's: IB-shaped
    commission, 1 bp half-spread, square-root impact Y=1.0, 10% cap."""
    return SimulatedExecutionHandler(
        commission=PerShareCommission(),
        slippage=HalfSpreadSlippage(spread_bps=1.0),
        impact=SquareRootImpact(coefficient=1.0),
        max_participation=0.10,
    )


def calendar_folds(index: pd.DatetimeIndex, train_years: int = 3,
                   test_years: int = 1) -> list[Fold]:
    """Rolling calendar folds over the years the index actually covers.

    2015-2025 with 3/1 gives eight folds: 2015-17 -> 18, ... 2022-24 -> 25.
    Boundaries are real bars from the index, so everything downstream
    aligns without asof games. Train always precedes test; the windows of
    one fold never overlap.
    """
    if train_years < 1 or test_years < 1:
        raise ValueError("train_years and test_years must be at least 1")
    years = sorted(set(index.year))
    span = train_years + test_years
    if len(years) < span:
        raise ValueError(f"need at least {span} calendar years, "
                         f"index covers {len(years)}")

    folds = []
    for i in range(len(years) - span + 1):
        tr = years[i:i + train_years]
        te = years[i + train_years:i + span]
        tr_bars = index[(index.year >= tr[0]) & (index.year <= tr[-1])]
        te_bars = index[(index.year >= te[0]) & (index.year <= te[-1])]
        if tr_bars.empty or te_bars.empty:
            continue
        folds.append(Fold(train_start=tr_bars[0], train_end=tr_bars[-1],
                          test_start=te_bars[0], test_end=te_bars[-1]))
    return folds


def run_window(close: pd.DataFrame, volume: pd.DataFrame,
               mask: pd.DataFrame | None, config: Config,
               start: pd.Timestamp, end: pd.Timestamp, *,
               capital: float = 10_000_000.0,
               execution_factory=loaded_execution,
               return_portfolio: bool = False):
    """The strategy over [start, end], returned rebased to 1.0.

    With return_portfolio=True, returns (curve, portfolio) so callers that
    need fills -- turnover, costs paid -- can have them without a second
    run. The curve is identical either way.

    The run actually begins `lookback` bars EARLIER, so formations exist
    from the first bar of the window and you walk in with a live book --
    which is what deployment looks like. That prefix is history, not
    leakage: a deployed strategy also knows the past. The returned curve is
    sliced to [start, end] and rebased, so metrics count only the window.
    """
    idx = close.index
    pos_start = idx.searchsorted(start)
    pos_end = idx.searchsorted(end, side="right")
    ctx_start = max(0, pos_start - config.lookback)

    c = close.iloc[ctx_start:pos_end]
    v = volume.iloc[ctx_start:pos_end]
    m = mask.loc[c.index] if mask is not None else None

    data = PanelBarHandler(c, v)
    pf = Portfolio(initial_capital=capital,
                   sizer=TargetWeightSizer(1.0 / config.top_n),
                   min_trade_fraction=BAND)
    strat = CrossSectionalMomentumStrategy(
        data, membership=m, top_n=config.top_n,
        lookback=config.lookback, skip=config.skip,
        rebalance=config.rebalance)
    Backtest(data, strat, pf, execution_factory()).run()

    curve = pf.equity_curve().loc[start:end]
    if curve.empty:
        raise ValueError(f"no bars in [{start}, {end}]")
    curve = curve / curve.iloc[0]
    return (curve, pf) if return_portfolio else curve


def embargoed_sharpe(curve: pd.Series, embargo_bars: int) -> float:
    """Annualised Sharpe of the curve with its final `embargo_bars` bars cut,
    so a formation straddling the train/test boundary cannot flatter the
    selection. NaN (flat or too-short curve) compares as -inf downstream."""
    if embargo_bars > 0:
        curve = curve.iloc[:-embargo_bars]
    returns = curve.pct_change().dropna()
    return sharpe_ratio(returns)


def select_on_train(close, volume, mask, fold: Fold, grid=GRID, *,
                    embargo_bars: int = REBALANCE,
                    capital: float = 10_000_000.0,
                    execution_factory=loaded_execution
                    ) -> tuple[Config, dict[Config, float]]:
    """Argmax embargoed net Sharpe over the TRAIN window only.

    Reads fold.train_start and fold.train_end and nothing else -- the test
    fields exist on the Fold but are structurally ignored, and the poisoned
    -test test holds this function to that. Exact ties break toward
    DEFAULT_CONFIG when it is among the leaders, else toward grid order.
    """
    scores: dict[Config, float] = {}
    for config in grid:
        curve = run_window(close, volume, mask, config,
                           fold.train_start, fold.train_end,
                           capital=capital,
                           execution_factory=execution_factory)
        s = embargoed_sharpe(curve, embargo_bars)
        scores[config] = float("-inf") if pd.isna(s) else s

    best = max(scores.values())
    leaders = [c for c in grid if scores[c] == best]
    chosen = DEFAULT_CONFIG if DEFAULT_CONFIG in leaders else leaders[0]
    return chosen, scores


def stitch(segments: list[pd.Series]) -> pd.Series:
    """Growth-of-1 compounding across ordered, disjoint segments.

    Each segment is rebased and scaled by the level the previous segments
    reached: equity compounds, it never adds. Summing would overweight the
    early years and produce a curve no account could have held.
    """
    if not segments:
        return pd.Series(dtype=float, name="oos_equity")
    parts = []
    level = 1.0
    last_ts = None
    for seg in segments:
        if seg.empty:
            raise ValueError("empty segment")
        if last_ts is not None and seg.index[0] <= last_ts:
            raise ValueError("segments must be ordered and disjoint")
        rebased = seg / seg.iloc[0]
        parts.append(rebased * level)
        level *= float(rebased.iloc[-1])
        last_ts = seg.index[-1]
    out = pd.concat(parts)
    out.name = "oos_equity"
    return out
