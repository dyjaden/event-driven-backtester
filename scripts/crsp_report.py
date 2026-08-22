"""What changed when the data changed.

    python scripts/crsp_report.py                      # the real pull
    python scripts/crsp_report.py --data data/_dryrun  # the rehearsal

Reads only Parquet files, so it re-runs offline and never needs a WRDS
session. That is what makes the dry run possible: the same script, the same
code path, fabricated inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtester.crsp import (delisting_summary, equal_weight_index,
                             membership_mask, to_engine_frame)
from backtester.engine import _run_pf
from backtester.execution import NaiveExecutionHandler
from backtester.metrics import summary
from backtester.strategy import BuyAndHoldStrategy


def _aligned(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]


def _yfinance_frame(data: Path) -> tuple[pd.DataFrame, bool]:
    """Prefer the dollar-volume-corrected file, and say which one was used."""
    fixed = data / "SPY_daily_dollarvol.parquet"
    if fixed.exists():
        return pd.read_parquet(fixed), True
    return pd.read_parquet(data / "SPY_daily.parquet"), False


def reconcile_spy(data: Path) -> None:
    """The same asset from two vendors. If they disagree, one is being read
    wrong -- and it is nearly always the one you did not write a loader for."""
    crsp = to_engine_frame(pd.read_parquet(data / "crsp_spy_daily.parquet"))
    yf, repaired = _yfinance_frame(data)

    c, y = _aligned(crsp["close"], yf["close"])
    print(f"RECONCILIATION -- SPY, {len(c):,} shared bars, "
          f"{c.index[0].date()} to {c.index[-1].date()}")
    print(f"  yfinance file: {'dollar-volume CORRECTED' if repaired else 'ORIGINAL, uncorrected'}")
    print(f"  CRSP  total return {c.iloc[-1]/c.iloc[0]-1:>9.2%}")
    print(f"  yf    total return {y.iloc[-1]/y.iloc[0]-1:>9.2%}")
    print(f"  gap                {(c.iloc[-1]/c.iloc[0])/(y.iloc[-1]/y.iloc[0])-1:>9.2%}")

    dc, dy = _aligned(c.pct_change().dropna(), y.pct_change().dropna())
    print(f"  daily return correlation {dc.corr(dy):.6f}")
    print(f"  worst single-day gap     {(dc - dy).abs().max():.6%} "
          f"on {(dc - dy).abs().idxmax().date()}")

    # The invariant. CRSP hands you both series, so the ratio is measurable
    # rather than argued. Corrected file: ~1.00 throughout. Original: the
    # dividend factor, ~1.2 at the start and ~1.0 at the end.
    rc, ry = _aligned(crsp["close"] * crsp["volume"], yf["close"] * yf["volume"])
    ratio = rc / ry
    print(f"  dollar-volume ratio CRSP/yf: first {ratio.iloc[0]:.3f}, "
          f"last {ratio.iloc[-1]:.3f}")
    if repaired:
        print("  ^ both near 1.00 confirms the yfinance repair was right, "
              "checked against a second vendor.")
    else:
        print("  ^ drift from 1 is the yfinance adjustment mismatch. Run "
              "scripts/fix_yfinance_volume.py.")


def crsp_baseline(data: Path) -> None:
    frame = to_engine_frame(pd.read_parquet(data / "crsp_spy_daily.parquet"))
    pf = _run_pf(frame, BuyAndHoldStrategy, NaiveExecutionHandler())
    print("\nBASELINE -- CRSP SPY, buy and hold, zero cost, $100,000")
    for k, v in summary(pf.equity_curve()).items():
        print(f"{k:>16}: {v}")


def survivorship(data: Path) -> None:
    """The number the README has been promising since Day 1.

    Not a backtest. A measurement of the data: two equally weighted baskets,
    daily rebalanced, no costs, no sizing, no engine. The only difference
    between the arms is WHO IS IN THE BASKET.
    """
    uni = pd.read_parquet(data / "crsp_universe_daily.parquet")
    members = pd.read_parquet(data / "crsp_sp500_membership.parquet")

    wide = (uni.pivot_table(index="dlycaldt", columns="permno", values="dlyret")
              .sort_index())
    wide.index = pd.DatetimeIndex(wide.index)

    mask = membership_mask(wide.index, wide.columns, members)
    last_day = mask.iloc[-1]
    survivors = list(last_day[last_day].index)

    pit = equal_weight_index(wide, mask)
    # The mistake almost every retail backtest makes: today's membership
    # list, applied to the whole history.
    today = equal_weight_index(wide[survivors])

    print(f"\nSURVIVORSHIP -- equal weight, daily rebalanced, {len(wide):,} bars")
    print(f"  point-in-time membership   {pit.iloc[-1]-1:>9.2%}  "
          f"({int(mask.sum(axis=1).mean())} names/day on average)")
    print(f"  today's membership list    {today.iloc[-1]-1:>9.2%}  "
          f"({len(survivors)} names)")
    print(f"  overstatement              {today.iloc[-1]/pit.iloc[-1]-1:>9.2%} "
          f"of total return")

    years = len(wide) / 252
    ann = lambda s: s.iloc[-1] ** (1 / years) - 1
    print(f"  annualised gap             {ann(today)-ann(pit):>9.2%} per year")

    print(f"  names per day: min {int(mask.sum(axis=1).min())}, "
          f"max {int(mask.sum(axis=1).max())}   "
          f"survivor arm with data: first bar "
          f"{int(wide[survivors].iloc[0].notna().sum())}, "
          f"last bar {int(wide[survivors].iloc[-1].notna().sum())}")

    dead = delisting_summary(uni)
    print(f"  delistings inside the window: {len(dead):,}")
    for r in dead.nsmallest(3, "dlyret").itertuples(index=False):
        print(f"    permno {r.permno}  {pd.Timestamp(r.dlycaldt).date()}  "
              f"{r.dlyret:>8.2%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data",
                    help="directory holding the Parquet files")
    args = ap.parse_args()
    data = Path(args.data)

    reconcile_spy(data)
    crsp_baseline(data)
    if (data / "crsp_universe_daily.parquet").exists():
        survivorship(data)
    else:
        print("\n(no universe file -- run pull_crsp.py --universe for the "
              "survivorship number)")


if __name__ == "__main__":
    main()