"""The first full costed backtest: 12-1 momentum on the point-in-time S&P 500.

    python scripts/momentum_report.py                      # the real pull
    python scripts/momentum_report.py --data data/_dryrun  # the rehearsal

Reads only Parquet, so it re-runs offline and never needs a WRDS session.
Every strategy number here came through the event loop -- panel handler,
portfolio, execution, costs. The benchmark is the same measurement Day 8-2
published (equal weight, daily rebalanced, no costs, point-in-time members),
printed beside the strategy so the comparison is on the page, not implied.

The deliverable is the measurement, not a victory: momentum net of costs is
allowed to lose to the basket it picks from, and if it does, that IS the
finding.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from backtester.costs import (HalfSpreadSlippage, PerShareCommission,
                              SquareRootImpact)
from backtester.crsp import equal_weight_index, membership_mask, to_panel
from backtester.data import PanelBarHandler
from backtester.engine import Backtest
from backtester.execution import NaiveExecutionHandler, SimulatedExecutionHandler
from backtester.portfolio import Portfolio, TargetWeightSizer
from backtester.strategy import CrossSectionalMomentumStrategy

# The knobs, stated once. 12-1 formation, monthly rebalance, long-only top
# 50, equal weight. The band skips per-name rebalance orders under 0.1% of
# equity (dust batches until material -- see test_momentum). No separate
# warm-up flag: the 253-bar lookback gate means the first order arrives with
# every trailing statistic already in existence.
TOP_N = 50
LOOKBACK, SKIP, REBALANCE = 252, 21, 21
BAND = 0.001
SIZES = (100_000.0, 10_000_000.0)


def costed_execution() -> SimulatedExecutionHandler:
    """The fully loaded arm: IB-shaped commission, 1 bp half-spread,
    square-root impact at Y=1.0, 10% participation cap."""
    return SimulatedExecutionHandler(
        commission=PerShareCommission(),
        slippage=HalfSpreadSlippage(spread_bps=1.0),
        impact=SquareRootImpact(coefficient=1.0),
        max_participation=0.10,
    )


def run(close, volume, mask, capital, execution) -> Portfolio:
    data = PanelBarHandler(close, volume)
    pf = Portfolio(initial_capital=capital,
                   sizer=TargetWeightSizer(1.0 / TOP_N),
                   min_trade_fraction=BAND)
    strat = CrossSectionalMomentumStrategy(
        data, membership=mask, top_n=TOP_N,
        lookback=LOOKBACK, skip=SKIP, rebalance=REBALANCE)
    Backtest(data, strat, pf, execution).run()
    return pf


def describe(pf: Portfolio, capital: float) -> dict:
    eq = pf.equity_curve()
    years = len(eq) / 252
    growth = eq.iloc[-1] / capital
    traded = sum(abs(f.gross_value) for f in pf.fills)
    return {
        "eq": eq,
        "total": growth - 1.0,
        "cagr": growth ** (1.0 / years) - 1.0,
        "turnover": traded / eq.mean() / years,   # both sides, annualised
        "commission": sum(f.commission for f in pf.fills),
        "impact": sum(f.impact_cost for f in pf.fills),
        "fills": len(pf.fills),
        "names": sum(1 for q in pf.positions.values() if q),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data",
                    help="directory holding the Parquet files")
    args = ap.parse_args()
    data = Path(args.data)

    uni = pd.read_parquet(data / "crsp_universe_daily.parquet")
    members = pd.read_parquet(data / "crsp_sp500_membership.parquet")

    close, volume = to_panel(uni, members)
    mask = (membership_mask(close.index, [int(c) for c in close.columns],
                            members)
            .rename(columns=str))
    years = len(close) / 252

    print(f"MOMENTUM -- 12-1, monthly, long-only top {TOP_N}, equal weight, "
          f"band {BAND:.1%}")
    print(f"  panel: {len(close):,} bars x {close.shape[1]:,} names, "
          f"{close.index[0].date()} to {close.index[-1].date()}  "
          f"({int(mask.sum(axis=1).mean())} members/day on average)")

    # ---------------------------------------------------------- benchmark
    # The Day 8-2 measurement, recomputed the same way it was published:
    # equal weight, daily rebalanced, zero cost, point-in-time membership.
    wide = uni.pivot(index="dlycaldt", columns="permno",
                     values="dlyret").sort_index()
    bench_mask = membership_mask(pd.DatetimeIndex(wide.index), wide.columns,
                                 members)
    pit = equal_weight_index(wide, bench_mask)
    pit_years = len(wide) / 252
    pit_cagr = pit.iloc[-1] ** (1.0 / pit_years) - 1.0
    print(f"\nBENCHMARK -- point-in-time equal weight, daily, zero cost "
          f"(the Day 8-2 arm)")
    print(f"  total {pit.iloc[-1] - 1.0:>9.2%}   annualised {pit_cagr:>7.2%}")

    # ---------------------------------------------------------- the runs
    checks: list[tuple[str, bool]] = []
    for capital in SIZES:
        rows = {}
        for label, execution in (("zero cost", NaiveExecutionHandler()),
                                 ("all costs", costed_execution())):
            t0 = time.time()
            pf = run(close, volume, mask, capital, execution)
            rows[label] = describe(pf, capital)
            rows[label]["secs"] = time.time() - t0

        free, full = rows["zero cost"], rows["all costs"]
        drag = free["cagr"] - full["cagr"]
        print(f"\nMOMENTUM at ${capital:,.0f}")
        for label in ("zero cost", "all costs"):
            r = rows[label]
            print(f"  {label:>9}: total {r['total']:>9.2%}   "
                  f"cagr {r['cagr']:>7.2%}   "
                  f"turnover {r['turnover']:>6.1%}/yr   "
                  f"{r['fills']:,} fills   {r['names']} names held   "
                  f"({r['secs']:.0f}s)")
        print(f"  costs paid: commission ${full['commission']:,.0f}   "
              f"impact ${full['impact']:,.0f}   "
              f"(slippage rides in the fill prices)")
        print(f"  cost drag : {drag * 100:.2f} pp/yr")
        print(f"  vs the benchmark, net: "
              f"{(full['cagr'] - pit_cagr) * 100:+.2f} pp/yr")

        checks.append((f"${capital:,.0f}: no NaN in either curve",
                       bool(free["eq"].notna().all()
                            and full["eq"].notna().all())))
        checks.append((f"${capital:,.0f}: costs only ever hurt",
                       full["eq"].iloc[-1] <= free["eq"].iloc[-1]))
        checks.append((f"${capital:,.0f}: book near {TOP_N} names",
                       0 < full["names"] <= TOP_N))

    # ------------------------------------------------------------- sanity
    print("\nSANITY")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        raise SystemExit("sanity check failed -- do not quote these numbers")


if __name__ == "__main__":
    main()
