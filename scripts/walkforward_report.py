"""Walk-forward momentum: parameters chosen on the past, judged on the future.

    python scripts/walkforward_report.py                      # the real pull
    python scripts/walkforward_report.py --data data/_dryrun  # the rehearsal

Offline, Parquet in, sections out. Eight folds (train three calendar years,
test the next), twelve frozen configurations, selection by embargoed net
Sharpe on the train window only. Two stitched out-of-sample curves come out:
per-fold selection, and the fixed default (12-1, top 50) that never selects
anything -- against the point-in-time equal-weight benchmark over the same
stitched years.

Pre-commitments, so the numbers get read rather than argued with: test below
train is the expected direction and its size is the finding; the fixed
default beating per-fold selection is a completely plausible outcome (twelve
tickets on a flat surface buys mostly noise); a fold that picks skip=0 is
chapter 11 happening live. The trial count -- 12 configurations, 8 folds --
is part of the result.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from backtester.crsp import equal_weight_index, membership_mask, to_panel
from backtester.metrics import sharpe_ratio, summary
from backtester.walkforward import (DEFAULT_CONFIG, GRID, calendar_folds,
                                    run_window, select_on_train, stitch)

CAPITAL = 10_000_000.0


def tag(c) -> str:
    return f"L{c.lookback}/S{c.skip}/N{c.top_n}"


def test_sharpe(curve: pd.Series) -> float:
    return sharpe_ratio(curve.pct_change().dropna())


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

    folds = calendar_folds(close.index, train_years=3, test_years=1)
    print(f"WALK-FORWARD -- {len(folds)} folds (train 3y, test 1y), "
          f"{len(GRID)} configs, ${CAPITAL:,.0f}, all costs on")
    print(f"  panel: {len(close):,} bars x {close.shape[1]:,} names   "
          f"trial count: {len(GRID)} x {len(folds)} = "
          f"{len(GRID) * len(folds)} train runs -- part of the result\n")

    hdr = (f"  {'test':>4} {'chosen':>13} {'trainSR':>8} {'testSR':>7} "
           f"{'default testSR':>14}")
    print("PER FOLD  (trainSR is embargoed; testSR is the full test year)")
    print(hdr)

    seg_sel, seg_def = [], []
    rows = []
    for fold in folds:
        t0 = time.time()
        chosen, scores = select_on_train(close, volume, mask, fold)
        sel_curve = run_window(close, volume, mask, chosen,
                               fold.test_start, fold.test_end,
                               capital=CAPITAL)
        if chosen == DEFAULT_CONFIG:
            def_curve = sel_curve
        else:
            def_curve = run_window(close, volume, mask, DEFAULT_CONFIG,
                                   fold.test_start, fold.test_end,
                                   capital=CAPITAL)
        row = {
            "year": fold.test_start.year,
            "chosen": chosen,
            "train_sr": scores[chosen],
            "test_sr": test_sharpe(sel_curve),
            "def_test_sr": test_sharpe(def_curve),
        }
        rows.append(row)
        seg_sel.append(sel_curve)
        seg_def.append(def_curve)
        flag = "  <- non-default" if chosen != DEFAULT_CONFIG else ""
        print(f"  {row['year']:>4} {tag(chosen):>13} {row['train_sr']:>8.2f} "
              f"{row['test_sr']:>7.2f} {row['def_test_sr']:>14.2f}"
              f"{flag}   ({time.time() - t0:.0f}s)", flush=True)

    oos_sel = stitch(seg_sel)
    oos_def = stitch(seg_def)

    # the benchmark over exactly the stitched years
    wide = uni.pivot(index="dlycaldt", columns="permno",
                     values="dlyret").sort_index()
    bench_mask = membership_mask(pd.DatetimeIndex(wide.index), wide.columns,
                                 members)
    pit = equal_weight_index(wide, bench_mask)
    pit_oos = pit.loc[oos_sel.index[0]:oos_sel.index[-1]]
    pit_oos = pit_oos / pit_oos.iloc[0]

    print(f"\nSTITCHED OUT OF SAMPLE, "
          f"{oos_sel.index[0].date()} to {oos_sel.index[-1].date()}")
    for label, curve in (("per-fold selection", oos_sel),
                        ("fixed default     ", oos_def),
                        ("PIT equal weight  ", pit_oos)):
        m = summary(curve)
        print(f"  {label}: total {curve.iloc[-1] - 1:>8.2%}   "
              f"cagr {m['cagr']:>7.2%}   sharpe {m['sharpe']:>5.2f}   "
              f"maxDD {m['max_drawdown']:>7.2%}")

    # the degradation, stated
    mean_train = sum(r["train_sr"] for r in rows) / len(rows)
    mean_test = sum(r["test_sr"] for r in rows) / len(rows)
    sel_vs_def = summary(oos_sel)["sharpe"] - summary(oos_def)["sharpe"]
    n_nondef = sum(1 for r in rows if r["chosen"] != DEFAULT_CONFIG)
    picked_skip0 = sum(1 for r in rows if r["chosen"].skip == 0)
    print("\nDEGRADATION")
    print(f"  mean train Sharpe of chosen configs : {mean_train:>5.2f}")
    print(f"  mean test  Sharpe of chosen configs : {mean_test:>5.2f}   "
          f"(IS -> OOS: {mean_test - mean_train:+.2f})")
    print(f"  selection vs fixed default, stitched OOS Sharpe: "
          f"{sel_vs_def:+.2f}")
    print(f"  non-default picks: {n_nondef}/{len(rows)}   "
          f"skip=0 picks: {picked_skip0}")
    verdict = ("added" if sel_vs_def > 0 else "cost")
    print(f"  VERDICT: per-fold selection {verdict} "
          f"{abs(sel_vs_def):.2f} Sharpe out of sample "
          f"vs never selecting at all.")

    # ------------------------------------------------------------- sanity
    checks = [
        ("every train window ends before its test window begins",
         all(f.train_end < f.test_start for f in folds)),
        ("stitched index is exactly the union of the test windows",
         list(oos_sel.index) == [t for s in seg_sel for t in s.index]),
        ("no NaN in any stitched curve",
         bool(oos_sel.notna().all() and oos_def.notna().all()
              and pit_oos.notna().all())),
        ("stitched index strictly increasing",
         oos_sel.index.is_monotonic_increasing
         and not oos_sel.index.has_duplicates),
    ]
    print("\nSANITY")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not all(ok for _, ok in checks):
        raise SystemExit("sanity check failed -- do not quote these numbers")


if __name__ == "__main__":
    main()
