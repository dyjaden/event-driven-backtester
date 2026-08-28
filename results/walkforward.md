# Walk-forward momentum — 28 August 2026

Eight folds (train three calendar years, test the next, rolled annually),
twelve frozen configurations (`lookback ∈ {126, 252} × skip ∈ {0, 21} ×
top_n ∈ {20, 50, 100}`, rebalance fixed at 21, band 0.1%), selection by
embargoed net Sharpe on the train window only, $10,000,000, all costs on.
96 training runs. **The trial count — 12 × 8 — is part of the result.**

Reproduce: `python scripts/walkforward_report.py` (~1 hour; prints its own
sanity checks and refuses a clean exit if any fail).

## Per fold

trainSR is the embargoed train-window Sharpe of the chosen config; testSR
is its full test year; the last column is the never-selecting default
(12-1, top 50) on the same test year.

| test | chosen | trainSR | testSR | default testSR |
|---|---|---|---|---|
| 2018 | L252/S21/N100 | 0.95 | -0.20 | -0.34 |
| 2019 | L126/S21/N100 | 1.04 | 1.42 | 1.61 |
| 2020 | L252/S21/N100 | 0.85 | 0.53 | 0.44 |
| 2021 | L252/S21/N100 | 0.49 | 1.04 | 0.75 |
| 2022 | L126/S0/N100 | 0.86 | -0.23 | 0.14 |
| 2023 | L126/S0/N100 | 0.56 | 0.86 | 0.90 |
| 2024 | L126/S21/N100 | 0.56 | 1.04 | 1.36 |
| 2025 | L252/S0/N20 | 0.88 | 0.89 | 0.66 |

## Stitched out of sample, 2018-01-02 to 2025-12-31

| | Total | CAGR | Sharpe | Max drawdown |
|---|---|---|---|---|
| Per-fold selection | 128.88% | 10.94% | 0.59 | -35.98% |
| Fixed default (12-1, top 50) | 133.73% | 11.23% | 0.57 | -37.89% |
| PIT equal weight (benchmark) | 134.56% | 11.28% | 0.63 | -39.55% |

## The degradation, stated

Mean Sharpe of the chosen configs shrank from **0.77 in train to 0.67 in
test** (-0.10 of selection optimism). Stitched, **per-fold selection added
+0.02 Sharpe versus never selecting at all** — a wash. Eight years of
annual re-optimization over twelve configurations bought two basis points
of Sharpe, which is what an honest small-grid search on a weak signal
should show.

What the folds chose is more interesting than what it earned:

- **Every fold went non-default, and seven of eight picked the broadest
  book (top 100)** — in-sample, diversifying the momentum bet beat
  concentrating it nearly every year. A consistent opinion, not churn.
- **The skip=0 trap was bought three times** (2022, 2023, 2025). In 2022 it
  was punished exactly as the theory predicts: -0.23 test Sharpe while the
  default made +0.14, reversal-chasing into a bear market. In 2023 it was a
  wash; in 2025 it won (0.89 vs 0.66) — traps pay just often enough to
  keep getting bought. Chapter 11, live.
- **Out of sample 2018-2025, default momentum nearly matches the benchmark
  on CAGR** (11.23% vs 11.28%) while trailing on Sharpe (0.57 vs 0.63).
  The full-window shortfall reported in `momentum_baseline.md` was
  concentrated in 2015-2017, which these folds only ever train on. The two
  windows are different; both sentences are true, and each names its
  window.

## Limitations carried forward

Walk-forward is ONE path through history; CPCV (Lopez de Prado ch. 12) is
the known stronger design and is future work, named rather than skipped.
Per-fold test Sharpes carry one-year error bars and are read for parameter
stability, not significance; conclusions come from the stitched eight
years, stated with humility.

Sanity, all PASS: every train window ends before its test window begins;
the stitched index is exactly the union of the test windows, strictly
increasing; no NaN in any curve.
