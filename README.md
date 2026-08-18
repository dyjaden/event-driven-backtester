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

### Limitations so far

- **No transaction costs.** No commission, no spread, no market impact. Day 6-7
- **Survivorship-biased data.** Yahoo Finance serves currently listed tickers
  only. Migrating to CRSP for delisting returns. Day 8
- **Same-bar fills.** Orders fill at the close the decision was made on, which
  is mildly optimistic. A conservative design fills at the next open
- **Cash drag.** Integer share counts leave a small uninvested remainder that
  never compounds, so total return sits slightly below the asset's own
- **One symbol.** Multi-asset support will be around Day 9