# Momentum baseline — v0.5, 26 August 2026

12-1 cross-sectional momentum on the point-in-time S&P 500, run end to end
through the event loop (panel handler, portfolio, execution, costs) against
CRSP CIZ daily data, 2015-01-02 to 2025-12-31. 2,766 bars x 677 names; 451
point-in-time members per day on average.

Formation: rank by `close[t-21] / close[t-252] - 1` every 21 bars; long-only
top 50, equal weight (`TargetWeightSizer(1/50)`); rebalance band 0.1% of
equity per name. Cost models on the loaded arm: per-share commission (IB
shape), 1 bp half-spread, square-root impact Y=1.0, 10% participation cap.

Reproduce: `python scripts/momentum_report.py` (~5 minutes; prints its own
sanity checks and exits non-zero if any fail). Numbers below were produced
independently on two machines and match digit for digit.

## Benchmark

Point-in-time equal weight, daily rebalanced, zero cost (the Day 8-2
measurement, reproduced here by an independent script):

| | Total 2015-2025 | Annualised |
|---|---|---|
| Point-in-time S&P 500, equal weight | 224.36% | 11.32% |

## Momentum

| | $100,000 | $10,000,000 |
|---|---|---|
| Zero cost, total | 192.60% | 214.90% |
| Zero cost, annualised | 10.28% | 11.02% |
| All costs, total | 180.02% | 197.54% |
| All costs, annualised | 9.84% | 10.44% |
| Turnover (both sides, annualised) | 703.7% | 733.4% |
| Cost drag | 0.44 pp/yr | 0.57 pp/yr |
| Commission paid | $5,552 | $65,057 |
| Impact paid | $738 | $808,519 |
| Names held (target 50) | 47 | 50 |
| **Net vs benchmark** | **-1.48 pp/yr** | **-0.87 pp/yr** |

## The finding

The strategy lost to the basket it picks from — after costs at both sizes,
and before costs at the small one. Long-only large-cap 12-1 momentum over a
window containing the 2016 and 2020 momentum reversals underperformed
point-in-time equal weight. That is the deliverable: the measurement, not a
victory. The literature effect is strongest long-short and in smaller
names; neither is modelled here (no borrow), and saying so is part of the
result.

Implementation findings that came with it:

- **A $100k account cannot implement equal-weight top 50 on the S&P.** A 2%
  slot is $2,000 and an expensive stock rounds to zero shares: the book
  held 47 of its 50 picks. The size penalty (10.28% vs 11.02% zero-cost) is
  integer-share granularity, not signal.
- **The cost structure flips with size.** Commission dominates at $100k
  ($5.6k vs $738 impact over eleven years); impact dominates at $10M
  ($809k vs $65k). The square-root law behaving as published.
- **Turnover is ~700%/yr both sides** — monthly re-ranking of a ~450-name
  field churns the bottom of the top-50 constantly. This is why the cost
  drag is material here when it rounded away for buy-and-hold.

Sanity, all PASS at both sizes: no NaN in either equity curve; the costed
arm never beats the zero-cost arm; the book stays at or under 50 names.
