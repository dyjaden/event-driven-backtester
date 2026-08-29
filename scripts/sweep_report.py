"""Robustness sweeps: the sensitivity surface, capacity, cost sensitivity,
and the deflated Sharpe of the best cell.

    python scripts/sweep_report.py                      # the real pull
    python scripts/sweep_report.py --data data/_dryrun  # the rehearsal

Every backtest this script runs lands in results/sweep_cache.csv, which is
both a cache (re-runs are free) and the TRIAL REGISTRY: the deflated Sharpe
needs an honest count of every configuration ever tried, and undisclosed
trials are the one thing that number cannot forgive.

THE CACHE DIES WITH THE ENGINE. A cached Sharpe from before an engine change
is a measurement of software that no longer exists -- delete
results/sweep_cache.csv in any commit that touches src/backtester/, and let
the sweeps rerun. Slow honesty beats fast fiction.

The grids are frozen in the Day 11 guide before the first run. Extending
them is allowed; the registry counts it and the deflated Sharpe pays for it.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from backtester.costs import (HalfSpreadSlippage, PerShareCommission,
                              SquareRootImpact, ZeroImpact, ZeroSlippage)
from backtester.crsp import membership_mask, to_panel
from backtester.execution import SimulatedExecutionHandler
from backtester.metrics import deflated_sharpe, expected_max_sharpe, summary
from backtester.walkforward import Config, loaded_execution, run_window

CACHE = Path("results/sweep_cache.csv")
CAPITAL = 10_000_000.0

KEY = ["tag", "lookback", "skip", "top_n", "rebalance",
       "start", "end", "capital"]
COLS = KEY + ["total", "cagr", "sharpe", "max_drawdown", "turnover",
              "commission", "impact", "ran_at"]

# ------------------------------------------------- the frozen Day 11 grids
SURFACE = tuple(Config(lookback=lb, skip=21, top_n=50, rebalance=rb)
                for lb in (63, 126, 189, 252) for rb in (5, 21, 42, 63))
BREADTH = tuple(Config(lookback=252, skip=21, top_n=n) for n in (20, 50, 100, 150))
AUM_LADDER = (1e5, 1e6, 1e7, 1e8, 1e9)
SPREADS_BP = (0.0, 1.0, 2.0, 5.0, 10.0)
IMPACT_YS = (0.0, 0.3, 1.0, 1.5)


def execution_with(spread_bps: float = 1.0, impact_y: float = 1.0):
    """The loaded arm with two auditable knobs exposed."""
    def factory():
        return SimulatedExecutionHandler(
            commission=PerShareCommission(),
            slippage=(ZeroSlippage() if spread_bps == 0
                      else HalfSpreadSlippage(spread_bps=spread_bps)),
            impact=(ZeroImpact() if impact_y == 0
                    else SquareRootImpact(coefficient=impact_y)),
            max_participation=0.10,
        )
    return factory


def _load_cache(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=COLS)


def _key_of(tag, config, start, end, capital) -> tuple:
    return (tag, config.lookback, config.skip, config.top_n,
            config.rebalance, str(pd.Timestamp(start).date()),
            str(pd.Timestamp(end).date()), float(capital))


def sweep(close, volume, mask, configs, *, start, end, tag,
          capital: float = CAPITAL, execution_factory=loaded_execution,
          cache_path: Path = CACHE) -> tuple[pd.DataFrame, int]:
    """Run every config not already in the cache; return (rows, n_ran).

    Rows come back in `configs` order whether cached or fresh. Costs and
    turnover are measured over the whole run, which for full-window sweeps
    (start = first bar) is exactly the measured window.
    """
    cache = _load_cache(cache_path)
    have = {tuple(r) for r in cache[KEY].itertuples(index=False)} if len(cache) else set()

    ran = 0
    new_rows = []
    for config in configs:
        key = _key_of(tag, config, start, end, capital)
        if key in have:
            continue
        t0 = time.time()
        curve, pf = run_window(close, volume, mask, config, start, end,
                               capital=capital,
                               execution_factory=execution_factory,
                               return_portfolio=True)
        m = summary(curve)
        eq = pf.equity_curve()
        years = len(eq) / 252
        traded = sum(abs(f.gross_value) for f in pf.fills)
        new_rows.append(dict(zip(KEY, key)) | {
            "total": m["total_return"], "cagr": m["cagr"],
            "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
            "turnover": traded / eq.mean() / years,
            "commission": sum(f.commission for f in pf.fills),
            "impact": sum(f.impact_cost for f in pf.fills),
            "ran_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        })
        ran += 1
        print(f"    ran {tag} L{config.lookback}/S{config.skip}"
              f"/N{config.top_n}/R{config.rebalance} @ ${capital:,.0f}: "
              f"sharpe {m['sharpe']:.2f}  ({time.time() - t0:.0f}s)",
              flush=True)

    if new_rows:
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)

    keys = [_key_of(tag, c, start, end, capital) for c in configs]
    idx = cache.set_index(KEY)
    rows = pd.DataFrame([idx.loc[k] for k in keys])
    out = pd.DataFrame(keys, columns=KEY).join(rows.reset_index(drop=True))
    return out, ran


def registry_trial_count(cache_path: Path = CACHE,
                         walkforward_trials: int = 12) -> int:
    """Every distinct configuration ever tried, per the registry, PLUS the
    walk-forward's twelve -- those trials happened whether or not this file
    remembers them. Cost-knob and AUM re-runs of one config count once per
    distinct (config, capital, tag): a re-pricing is still a look."""
    cache = _load_cache(cache_path)
    if len(cache) == 0:
        return walkforward_trials
    return int(cache[KEY].drop_duplicates().shape[0]) + walkforward_trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    data = Path(args.data)

    uni = pd.read_parquet(data / "crsp_universe_daily.parquet")
    members = pd.read_parquet(data / "crsp_sp500_membership.parquet")
    close, volume = to_panel(uni, members)
    mask = (membership_mask(close.index, [int(c) for c in close.columns],
                            members).rename(columns=str))
    start, end = close.index[0], close.index[-1]

    print(f"SWEEP RUNNER -- cache {CACHE} "
          f"({'exists' if CACHE.exists() else 'new'})")
    df, ran = sweep(close, volume, mask, SURFACE + BREADTH,
                    start=start, end=end, tag="net")
    print(f"\n  surface+breadth: {len(df)} configs, {ran} freshly run, "
          f"{len(df) - ran} from cache")
    print(f"  registry trial count (incl. walk-forward's 12): "
          f"{registry_trial_count()}")
    # Steps 2-4 read this cache and draw the pictures.


if __name__ == "__main__":
    main()
