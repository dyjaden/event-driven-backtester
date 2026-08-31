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

import matplotlib
matplotlib.use("Agg")                    # headless; PNGs, never windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from backtester.costs import (HalfSpreadSlippage, PerShareCommission,
                              SquareRootImpact, ZeroImpact, ZeroSlippage)
from backtester.crsp import membership_mask, to_panel
from backtester.execution import NaiveExecutionHandler, SimulatedExecutionHandler
from backtester.metrics import deflated_sharpe, expected_max_sharpe, summary
from backtester.walkforward import (DEFAULT_CONFIG, Config, loaded_execution,
                                    run_window)

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
    """The append-only log, deduplicated on read: the same key measured
    twice is the same measurement (the engine is deterministic), so the
    first occurrence wins and the duplicate is inert."""
    if path.exists():
        return pd.read_csv(path).drop_duplicates(subset=KEY, keep="first",
                                                 ignore_index=True)
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

    Each row is APPENDED TO DISK the moment its backtest completes, not when
    the batch ends -- an interrupted sweep loses nothing, and the registry
    records every backtest that actually ran, which is the only kind of
    registry the deflated Sharpe can trust.
    """
    cache = _load_cache(cache_path)
    have = {tuple(r) for r in cache[KEY].itertuples(index=False)} if len(cache) else set()

    ran = 0
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
        row = dict(zip(KEY, key)) | {
            "total": m["total_return"], "cagr": m["cagr"],
            "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
            "turnover": traded / eq.mean() / years,
            "commission": sum(f.commission for f in pf.fills),
            "impact": sum(f.impact_cost for f in pf.fills),
            "ran_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
        }
        row_df = pd.DataFrame([row], columns=COLS)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        row_df.to_csv(cache_path, mode="a",
                      header=not cache_path.exists(), index=False)
        cache = pd.concat([cache, row_df], ignore_index=True)
        have.add(key)            # a config listed twice in one call runs once
        ran += 1
        print(f"    ran {tag} L{config.lookback}/S{config.skip}"
              f"/N{config.top_n}/R{config.rebalance} @ ${capital:,.0f}: "
              f"sharpe {m['sharpe']:.2f}  ({time.time() - t0:.0f}s)",
              flush=True)

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


def draw_heatmap(surface: pd.DataFrame,
                 png: Path = Path("results/sensitivity_heatmap.png"),
                 csv: Path = Path("results/sensitivity_heatmap.csv")) -> None:
    """Lookback x rebalance -> net Sharpe. Every cell carries its number (a
    color gradient without printed values is decoration) and the pivot table
    is saved beside the PNG. One hue when the surface is one-signed; a
    red/blue split around zero when it is not -- polarity is information.
    White figure background on purpose: GitHub dark mode renders PNGs as-is.
    """
    grid = (surface.pivot(index="lookback", columns="rebalance",
                          values="sharpe")
            .sort_index().sort_index(axis=1).astype(float))
    csv.parent.mkdir(parents=True, exist_ok=True)
    grid.round(4).to_csv(csv)

    lo, hi = float(grid.min().min()), float(grid.max().max())
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    if lo < 0 < hi:
        span = max(abs(lo), abs(hi))
        norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
        im = ax.imshow(grid.values, cmap="RdBu", norm=norm, aspect="auto")
    else:
        im = ax.imshow(grid.values, cmap="Blues", vmin=lo, vmax=hi,
                       aspect="auto")

    for i, lb in enumerate(grid.index):
        for j, rb in enumerate(grid.columns):
            v = grid.iloc[i, j]
            r, g, b, _ = im.cmap(im.norm(v))
            ink = "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5 else "#1a1a1a"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=ink, fontsize=13, fontweight="bold")

    ax.set_xticks(range(len(grid.columns)),
                  [f"{c}" for c in grid.columns], fontsize=11)
    ax.set_yticks(range(len(grid.index)),
                  [f"{i}" for i in grid.index], fontsize=11)
    ax.set_xlabel("rebalance every N bars", fontsize=11)
    ax.set_ylabel("lookback (bars)", fontsize=11)
    ax.set_title("Net Sharpe at $10M -- lookback x rebalance\n"
                 "(top 50, skip 21, 2015-2025, all costs on)",
                 fontsize=12, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("net Sharpe", fontsize=10)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"  wrote {png} and {csv}")


def draw_capacity(ladder: pd.DataFrame,
                  png: Path = Path("results/capacity_curve.png"),
                  csv: Path = Path("results/capacity_curve.csv")
                  ) -> tuple[float, float]:
    """Cost drag (gross CAGR minus net CAGR, pp/yr) against AUM, log-x.

    The drag is measured RUNG BY RUNG -- the gross arm reruns at every
    capital, because share rounding at $100k is not the rounding at $1B and
    the baseline must carry the same granularity as the run it absolves.
    This strategy has no edge over its benchmark to kill, so this curve is a
    property of the implementation, not of an alpha: the question is where
    the cost curve steepens, not where profit dies. Returns the log-
    interpolated AUMs where drag crosses 1 and 2 pp/yr (nan if never).
    """
    t = ladder.sort_values("capital").reset_index(drop=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    t.round(6).to_csv(csv, index=False)

    def crossing(level: float) -> float:
        x, y = np.log10(t["capital"].values), t["drag_pp"].values
        for i in range(len(y) - 1):
            lo, hi = y[i] - level, y[i + 1] - level
            if lo * hi <= 0 and y[i] != y[i + 1]:
                f = (level - y[i]) / (y[i + 1] - y[i])
                return float(10 ** (x[i] + f * (x[i + 1] - x[i])))
        return float("nan")

    aum1, aum2 = crossing(1.0), crossing(2.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_xscale("log")
    ax.plot(t["capital"], t["drag_pp"], color="#2166ac", lw=2,
            marker="o", ms=7, zorder=3)
    for _, r in t.iterrows():
        ax.annotate(f"{r['drag_pp']:.2f}", (r["capital"], r["drag_pp"]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9.5, color="#1a1a1a")
    for level, aum in ((1.0, aum1), (2.0, aum2)):
        ax.axhline(level, color="#999999", lw=1, ls="--", zorder=1)
        if np.isfinite(aum):
            ax.axvline(aum, color="#999999", lw=1, ls=":", zorder=1)
            ax.annotate(f"{level:.0f} pp/yr at ${aum / 1e6:,.0f}M",
                        (aum, level), textcoords="offset points",
                        xytext=(6, -14), fontsize=9.5, color="#444444")
    ax.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9],
                  ["$100k", "$1M", "$10M", "$100M", "$1B"], fontsize=10)
    ax.set_xlabel("AUM (log scale)", fontsize=11)
    ax.set_ylabel("cost drag (pp of CAGR per year)", fontsize=11)
    ax.set_title("Where costs eat the implementation -- drag vs AUM\n"
                 "(12-1 momentum, top 50, monthly, 2015-2025, "
                 "gross minus net CAGR)", fontsize=12, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cccccc")
    ax.tick_params(colors="#444444")
    ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"  wrote {png} and {csv}")
    return aum1, aum2


def draw_cost_sensitivity(spread: pd.DataFrame, impact: pd.DataFrame,
                          png: Path = Path("results/cost_sensitivity.png"),
                          csv: Path = Path("results/cost_sensitivity.csv")
                          ) -> None:
    """Net Sharpe against the two cost assumptions nobody audits: the
    half-spread (bp) and the impact coefficient Y. Side-by-side panels on a
    shared y-axis -- the x units differ, and a twin-axis chart is how
    readers get lied to."""
    both = pd.concat([spread.assign(slice="spread_bp"),
                      impact.assign(slice="impact_y")], ignore_index=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    both.round(6).to_csv(csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), dpi=150,
                             sharey=True)
    fig.patch.set_facecolor("white")
    panels = ((axes[0], spread, "assumed half-spread (bp)",
               "spread, at impact Y=1.0"),
              (axes[1], impact, "impact coefficient Y",
               "impact, at spread 1 bp"))
    for ax, t, xlabel, sub in panels:
        ax.plot(t["x"], t["sharpe"], color="#2166ac", lw=2,
                marker="o", ms=7, zorder=3)
        for _, r in t.iterrows():
            ax.annotate(f"{r['sharpe']:.2f}", (r["x"], r["sharpe"]),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9.5, color="#1a1a1a")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(sub, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")
        ax.tick_params(colors="#444444")
        ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
    axes[0].set_ylabel("net Sharpe", fontsize=11)
    fig.suptitle("Net Sharpe vs the cost assumptions "
                 "(default config, $10M, 2015-2025)", fontsize=12)
    fig.tight_layout()
    fig.savefig(png, facecolor="white")
    plt.close(fig)
    print(f"  wrote {png} and {csv}")


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

    # ---------------------------------------- Step 2: the surface, drawn
    surface_df = df.iloc[:len(SURFACE)]
    breadth_df = df.iloc[len(SURFACE):]
    draw_heatmap(surface_df)

    print("\nBREADTH  (L252/S21/R21, $10M, all costs on)")
    print(f"  {'top_n':>6} {'sharpe':>7} {'cagr':>7} {'maxDD':>8} "
          f"{'turnover':>9}")
    for _, r in breadth_df.iterrows():
        print(f"  {int(r['top_n']):>6} {r['sharpe']:>7.2f} {r['cagr']:>7.2%} "
              f"{r['max_drawdown']:>8.2%} {r['turnover']:>8.2f}x")

    by_n = breadth_df.set_index("top_n")["sharpe"]
    if by_n[100] > by_n[50]:
        verdict = ("top-100 beats top-50 on the full window too: the folds "
                   "were seeing real diversification of a weak signal.")
    else:
        verdict = ("top-100 does NOT beat top-50 on the full window: the "
                   "folds' preference was fit to three-year noise.")
    print(f"  BREADTH VERDICT: {verdict}")

    # -------------------------------------------- Step 3: the AUM ladder
    print(f"\nCAPACITY  (default config, gross rerun at every rung)")
    rungs = []
    for cap in AUM_LADDER:
        n, _ = sweep(close, volume, mask, (DEFAULT_CONFIG,),
                     start=start, end=end, tag="net", capital=cap)
        g, _ = sweep(close, volume, mask, (DEFAULT_CONFIG,),
                     start=start, end=end, tag="gross", capital=cap,
                     execution_factory=NaiveExecutionHandler)
        rungs.append({
            "capital": cap,
            "gross_cagr": float(g["cagr"].iloc[0]),
            "net_cagr": float(n["cagr"].iloc[0]),
            "drag_pp": (float(g["cagr"].iloc[0])
                        - float(n["cagr"].iloc[0])) * 100,
            "net_sharpe": float(n["sharpe"].iloc[0]),
            "net_turnover": float(n["turnover"].iloc[0]),
            "gross_turnover": float(g["turnover"].iloc[0]),
            "commission": float(n["commission"].iloc[0]),
            "impact": float(n["impact"].iloc[0]),
        })
    ladder = pd.DataFrame(rungs)
    aum1, aum2 = draw_capacity(ladder)

    print(f"  {'AUM':>9} {'gross':>7} {'net':>7} {'drag':>8} {'sharpe':>7} "
          f"{'turnover':>9}")
    for _, r in ladder.iterrows():
        label = f"${r['capital'] / 1e6:,.1f}M"
        print(f"  {label:>9} "
              f"{r['gross_cagr']:>7.2%} {r['net_cagr']:>7.2%} "
              f"{r['drag_pp']:>6.2f}pp {r['net_sharpe']:>7.2f} "
              f"{r['net_turnover']:>8.2f}x")
    for level, aum in ((1.0, aum1), (2.0, aum2)):
        where = (f"${aum / 1e6:,.0f}M" if np.isfinite(aum)
                 else "never inside the ladder")
        print(f"  drag crosses {level:.0f} pp/yr at: {where}")
    tw = ladder.set_index("capital")
    shrink = 1 - tw.loc[1e9, "net_turnover"] / tw.loc[1e9, "gross_turnover"]
    if shrink > 0.05:
        print(f"  at $1B the capped book completes {1 - shrink:.0%} of the "
              f"turnover the uncapped one wants -- the Day 7 participation "
              f"wall, reappearing at the illiquid tail")

    # ------------------------------------------ Step 4a: the cost slices
    print(f"\nCOST SENSITIVITY  (default config, $10M)")

    def slice_row(x, res):
        return {"x": x, "sharpe": float(res["sharpe"].iloc[0]),
                "cagr": float(res["cagr"].iloc[0]),
                "commission": float(res["commission"].iloc[0]),
                "impact": float(res["impact"].iloc[0])}

    spread_rows, impact_rows = [], []
    for bp in SPREADS_BP:
        if bp == 1.0:                      # the loaded arm IS 1 bp / Y=1.0
            r, _ = sweep(close, volume, mask, (DEFAULT_CONFIG,),
                         start=start, end=end, tag="net")
        else:
            r, _ = sweep(close, volume, mask, (DEFAULT_CONFIG,),
                         start=start, end=end, tag=f"spread{bp:g}bp",
                         execution_factory=execution_with(spread_bps=bp,
                                                          impact_y=1.0))
        spread_rows.append(slice_row(bp, r))
    for y in IMPACT_YS:
        if y == 1.0:
            r, _ = sweep(close, volume, mask, (DEFAULT_CONFIG,),
                         start=start, end=end, tag="net")
        else:
            r, _ = sweep(close, volume, mask, (DEFAULT_CONFIG,),
                         start=start, end=end, tag=f"impactY{y:g}",
                         execution_factory=execution_with(spread_bps=1.0,
                                                          impact_y=y))
        impact_rows.append(slice_row(y, r))
    spread_t = pd.DataFrame(spread_rows)
    impact_t = pd.DataFrame(impact_rows)
    draw_cost_sensitivity(spread_t, impact_t)

    slope = ((spread_t["sharpe"].iloc[-1] - spread_t["sharpe"].iloc[0])
             / (spread_t["x"].iloc[-1] - spread_t["x"].iloc[0]))
    print(f"  spread slice : " + "  ".join(
        f"{r['x']:g}bp {r['sharpe']:.2f}" for _, r in spread_t.iterrows()))
    print(f"  impact slice : " + "  ".join(
        f"Y={r['x']:g} {r['sharpe']:.2f}" for _, r in impact_t.iterrows()))
    print(f"  SLOPE: each basis point of assumed half-spread costs "
          f"~{abs(slope):.3f} Sharpe here. The expectation was that this "
          f"assumption dominates; at ~7x/yr turnover it does not -- the "
          f"sensitivity lives in the turnover knob (the rebalance=5 "
          f"column), and a spread assumption only matters as fast as you "
          f"trade against it.")

    # --------------------- Step 4b: the deflated Sharpe of the best cell
    # The discovery set whose spread feeds the null: the $10M net
    # strategy variants (surface + breadth). N is larger -- every look the
    # registry knows about, plus the walk-forward's twelve.
    strat = df.drop_duplicates(subset=KEY)
    per_period = strat["sharpe"] / np.sqrt(252)
    var_trials = float(per_period.var(ddof=1))
    best = strat.loc[strat["sharpe"].idxmax()]
    best_cfg = Config(int(best["lookback"]), int(best["skip"]),
                      int(best["top_n"]), int(best["rebalance"]))
    n = registry_trial_count()
    curve = run_window(close, volume, mask, best_cfg, start, end,
                       capital=CAPITAL)      # re-measure for daily returns
    rets = curve.pct_change().dropna()
    dsr_n = deflated_sharpe(rets, n, var_trials)
    dsr_2n = deflated_sharpe(rets, 2 * n, var_trials)
    sr0_ann = expected_max_sharpe(n, var_trials) * np.sqrt(252)

    print(f"\nDEFLATED SHARPE of the best cell "
          f"(L{best_cfg.lookback}/S{best_cfg.skip}/N{best_cfg.top_n}"
          f"/R{best_cfg.rebalance})")
    print(f"  raw Sharpe {best['sharpe']:.2f}   trials N={n} "
          f"(registry {n - 12} + walk-forward 12)   "
          f"cross-trial spread from {len(strat)} strategy variants")
    print(f"  expected max Sharpe of N pure-noise trials: {sr0_ann:.2f} "
          f"(annualised)")
    print(f"  DSR at N:  {dsr_n:.2f}     DSR at 2N: {dsr_2n:.2f}")
    print(f"  SENTENCE: the best cell's Sharpe is {best['sharpe']:.2f}; "
          f"after deflating for {n} trials, the evidence it is real "
          f"is {dsr_n:.2f} (and {dsr_2n:.2f} if the true trial count "
          f"were double).")


if __name__ == "__main__":
    main()
