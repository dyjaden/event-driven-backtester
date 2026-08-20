# event-driven-backtester
Event-driven backtesting engine with transaction cost and market impact modeling

## Status: v0.1

Event-driven engine with portfolio accounting, Zero transaction costs so far

![Equity curve](results/equity_curve.png)

| Metric | Buy and hold, SPY |
|---|---|
| Total return | 302.56% |
| CAGR | 13.54% |
| Volatility | 17.79% |
| Sharpe | 0.80 |
| Max drawdown | -33.71% |
| Periods | 2765 |

## Installation

```bash
git clone https://github.com/dyjaden/event-driven-backtester.git
cd event-driven-backtester
python -m venv .venv && .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

To reproduce the exact figures below, install the pinned versions instead:
`python -m pip install -r requirements.txt`

### Limitations so far

- **Costs modelled: commission and half-spread. Market impact is not yet
  included**, so results overstate achievable performance at size. Day 7.
- **The spread assumption is a parameter, not a measurement.** Results are
  reported across 1-5 bp rather than at a single value.
- **Survivorship-biased data.** Yahoo Finance serves currently listed tickers
  only. Migrating to CRSP for delisting returns. Day 8
- **Same-bar fills.** Orders fill at the close the decision was made on, which
  is mildly optimistic. A conservative design fills at the next open
- **Cash drag.** Integer share counts leave a small uninvested remainder that
  never compounds, so total return sits slightly below the asset's own
- **One symbol.** Multi-asset support will be around Day 9