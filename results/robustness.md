# Robustness — 30 August 2026

Is anything in this repo's momentum result real, or is it the best of a
pile of lottery tickets? Four screens: the sensitivity surface, breadth,
capacity, cost sensitivity — then chapter 11's math turned on our own best
cell. Grids were frozen in the Day 11 guide before the first run.

Reproduce: `python scripts/sweep_report.py` (~45 min cold; seconds warm —
every backtest lands in `results/sweep_cache.csv`, which is both the cache
and the **trial registry**. The cache dies with the engine: delete it in
any commit that changes behaviour under `src/backtester/`; a
docstring-only commit keeps it and says so in its message.)

## The sensitivity surface

Net Sharpe at $10M, lookback × rebalance (top 50, skip 21, all costs on).
`sensitivity_heatmap.png` / `.csv`:

| lookback \ rebalance | 5 | 21 | 42 | 63 |
|---|---|---|---|---|
| 63  | 0.38 | 0.47 | 0.61 | 0.58 |
| 126 | 0.57 | 0.61 | 0.63 | 0.71 |
| 189 | 0.59 | 0.66 | **0.73** | **0.75** |
| 252 | 0.55 | 0.58 | 0.52 | 0.58 |

- **The rebalance=5 column is eaten alive, as pre-committed**: worst or
  near-worst cell in every row. Weekly re-ranking turns the book over
  21.5×/yr against 5.0× at quarterly and pays $2.6M against $0.8M in costs
  on $10M over the window. Costs bind; the cost path works.
- **The surface is a smooth ridge, not a lone hot cell** — Sharpe rises
  toward slower rebalancing and mid-length formation, peaking at
  L189/R63 = 0.75 beside neighbours at 0.73 and 0.71.
- **The canonical 12-1 row (L252) is the weakest lookback at every
  cadence.** Nine-month formation beat the textbook *on this one window*.
  That is promotion bait, and the cell stays unpromoted: the deflated
  Sharpe below is the only way this repo reads its own maximum.

## The breadth verdict

Day 10's folds preferred the top-100 book in seven of eight train windows.
Full-window answer (L252/S21/R21, $10M, net):

| top_n | Sharpe | CAGR | maxDD | turnover |
|---|---|---|---|---|
| 20  | 0.53 | 10.66% | -38.65% | 7.90× |
| 50  | 0.58 | 10.45% | -37.06% | 7.33× |
| 100 | **0.67** | 11.32% | -36.15% | 5.77× |
| 150 | 0.65 | 10.39% | -36.87% | 4.58× |

**Top-100 beats top-50 on the full window too: the folds were seeing real
diversification of a weak signal, not three-year noise.** Breadth helps
until it dilutes (150 gives back a little), and it trades less along the
way — broader books churn their rank boundary less per dollar.

## Capacity

Default config, gross arm rerun at every rung (share rounding at $100k is
not the rounding at $1B; the baseline must carry the granularity of the
run it absolves). Drag = gross CAGR − net CAGR. `capacity_curve.png`:

| AUM | gross | net | drag | net Sharpe | turnover |
|---|---|---|---|---|---|
| $100k | 10.28% | 9.84% | 0.44 pp | 0.57 | 7.03× |
| $1M   | 10.96% | 10.71% | 0.25 pp | 0.59 | 7.31× |
| $10M  | 11.02% | 10.45% | 0.57 pp | 0.58 | 7.33× |
| $100M | 11.03% | 9.44%  | 1.58 pp | 0.53 | 7.32× |
| $1B   | 11.03% | 8.16%  | 2.87 pp | 0.48 | 6.29× |

- **Drag crosses 1 pp/yr at ≈ $26M and 2 pp/yr at ≈ $211M** (log-
  interpolated). No wall inside the ladder — the square-root climb, as the
  50-name monthly book predicts.
- **The curve is J-shaped**: $100k pays more drag than $1M because
  per-order commission minimums bite small orders. The retail account is
  eaten from the other end; the sweet spot is around $1M.
- **The Day 7 wall reappears at the illiquid tail**: at $1B the 10%
  participation cap binds and the book completes only 86% of the turnover
  the uncapped arm wants. SPY buy-and-hold hit its wall at $3.27B in one
  name; fifty names push the wall out, they do not abolish it.
- This strategy has no edge over its benchmark to kill, so these are
  capacity numbers for the *implementation*, not for an alpha.

## Cost sensitivity

Net Sharpe at $10M against the two assumptions nobody audits
(`cost_sensitivity.png`):

| half-spread | 0 bp | 1 bp | 2 bp | 5 bp | 10 bp |
|---|---|---|---|---|---|
| Sharpe | 0.58 | 0.58 | 0.58 | 0.57 | 0.56 |

| impact Y | 0.0 | 0.3 | 1.0 | 1.5 |
|---|---|---|---|---|
| Sharpe | 0.60 | 0.59 | 0.58 | 0.57 |

**The pre-committed reading was wrong, and the data gets the last word.**
The expectation — written in the guide before the run — was that the
spread assumption moves the answer more than most strategy debates. At
this book's ~7×/yr turnover it does not: each assumed basis point of
half-spread costs ~0.002 Sharpe, and even 10 bp only costs 0.02. A cost
assumption matters exactly as fast as you trade against it — the
sensitivity lives in the turnover knob (see the rebalance=5 column), not
in the spread guess. Day 6's lesson, resurfacing at panel scale.

## The deflated Sharpe of the best cell

Best cell: **L189/S21/N50/R63, raw net Sharpe 0.75**. Trial count
**N = 47** — every distinct configuration in the registry (35: the
surface and breadth grids, the capacity rungs and gross arms, the cost
slices) plus the walk-forward's 12, which happened whether or not this
file remembers them. Cross-trial spread from the 19 strategy variants at
$10M net.

Expected max Sharpe of 47 pure-noise trials at that spread: **0.20**
(annualised). Deflated Sharpe of the observed best against it:

> **The best cell's Sharpe is 0.75; after deflating for 47 trials, the
> evidence it is real is 0.96 — and still 0.96 at double the trial
> count.**

Two honesty notes, without which that number overstates itself:

1. **The 2N sensitivity barely moves because E[max] grows like √(ln N).**
   Undisclosed trials hurt logarithmically; what makes discoveries
   dissolve is a *wide* spread of trial Sharpes, and this grid's spread is
   tight (0.38–0.75 over eleven years). The registry still matters — at
   N in the thousands the deflation is material — but doubling a
   double-digit count is survivable arithmetic.
2. **DSR tests the null "true Sharpe = 0", and for a long-only,
   fully-invested book over 2015–2025 that null is weak.** The
   point-in-time equal-weight benchmark earned Sharpe 0.63 over the same
   window; market beta alone clears zero. So 0.96 says the 0.75 is not a
   selection artifact of the grid — it does NOT say the strategy beats
   its benchmark, which it does not (`momentum_baseline.md`,
   `walkforward.md`). Deflation guards against multiple testing; choosing
   the right benchmark is a separate, older discipline, and both
   sentences belong beside any number this repo reports.

## Limitations carried forward

Every surface and screen in this file shares one 2015–2025 window;
robustness across knobs is not robustness across regimes. Walk-forward
remains one path through history (CPCV is named future work). The impact
model is a calibrated square-root law, not measured fills; capacity
numbers inherit its coefficient.
