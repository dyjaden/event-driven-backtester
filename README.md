# event-driven-backtester

An event-driven backtesting engine in pure Python, built around one idea: **most
backtests are wrong, and the interesting engineering is in the parts that make
them wrong.**

Look-ahead bias, survivorship bias, transaction costs and market impact are
modelled explicitly rather than assumed away, and every result states the
capital it was produced at.

```
DataHandler --MarketEvent--> Strategy --SignalEvent--> Portfolio
     ^                                                    |
     |                                                OrderEvent
     |                                                    v
     +---------------FillEvent---- ExecutionHandler <-----+
```

Four event types through a single queue. **The queue is what makes look-ahead
bias structurally impossible** rather than something you have to remember to
avoid: an order cannot be filled with information that did not exist when it
was created, because time only advances when the queue is empty.

`SignalEvent` deliberately carries no quantity. The strategy decides *what*,
the portfolio decides *how much*, which is why risk rules can change without
touching strategy logic.

## Status

Engine, portfolio accounting, pluggable position sizing, commission and
half-spread costs, square-root market impact and a participation cap. 59 tests.

## Headline result

Buy and hold, SPY, **$100,000**, zero costs. This is the known-answer baseline:
with no costs, a buy-and-hold backtest must reproduce the asset's own return.
If it does not, the engine is wrong, not the strategy.

![Equity curve](results/equity_curve.png)

| Metric | Buy and hold, SPY, $100,000 |
|---|---|
| Total return | 302.56% |
| CAGR | 13.54% |
| Volatility | 17.79% |
| Sharpe | 0.80 |
| Max drawdown | -33.71% |
| Periods | 2765 |

The -33.71% drawdown is the COVID crash, correctly reproduced.

## Costs are decided by turnover, not by the cost model

The same commission and half-spread model applied to two strategies. The only
variable is how often they trade.

At 5 bp, buy and hold gives up **$27.93** on a $402,563 account, which rounds
away entirely at two decimal places. The identical model applied to a strategy
that round-trips every other bar:

| Spread | Share of the costs-off result that survives |
|---|---|
| 1 bp | 83.0% |
| 2 bp | 72.3% |
| 5 bp | 47.7% |

*Alternating strategy, roughly 1,382 round trips over the same 2,765 bars.*

This is why **"I included transaction costs" is a weaker claim than it sounds.**
Include them on a strategy that trades once a decade and you have demonstrated
nothing. Turnover belongs next to Sharpe in any honest performance report.

## Capacity: Sharpe is a property of a strategy *at a size*

Market impact is modelled with the square-root law:

```
price move / price  =  Y * sigma_daily * sqrt(Q / ADV)
```

At `Q = ADV` and `Y = 1` the move is exactly one daily standard deviation,
which is the standard rule of thumb: consuming a full day's volume costs about
one sigma. ADV and volatility are **trailing** estimates taken from bars up to
and including the current one, never the full sample.

| AUM | Total return | Impact paid | bp of AUM | Ratio to previous row |
|---|---|---|---|---|
| $100,000 | 303.87% | $2 | 0.22 | — |
| $1,000,000 | 303.86% | $69 | 0.69 | 3.16x |
| $10,000,000 | 303.85% | $2,169 | 2.17 | 3.16x |
| $100,000,000 | 303.81% | $68,581 | 6.86 | 3.16x |
| $1,000,000,000 | 303.66% | $2,168,713 | 21.69 | 3.16x |
| $10,000,000,000 | 303.19% | $68,580,717 | 68.58 | 3.16x |

*SPY, buy and hold, coefficient 1.0, 21-bar warm-up. The warmed arm enters 21
bars later than the zero-cost baseline above, which is why its total return
differs slightly.*

Because impact per share goes as the square root of participation, **cost in
basis points scales as the square root of AUM**: every tenfold increase in size
multiplies it by 3.16, not by 10. In dollar terms the total goes as `Q^1.5`,
which is why capacity behaves like a wall rather than a slope.

### The coefficient is a pure multiplier

Impact is linear in `Y`, so the choice of coefficient scales the entire cost
line. At $1bn:

| Y | Impact paid |
|---|---|
| 0.3 | $650,614 |
| 0.5 | $1,084,356 |
| 1.0 | $2,168,713 |
| 1.5 | $3,253,069 |

Moving from 1.5 to 0.3 is one keystroke and a **five-fold** reduction in
modelled cost, with no external check on which value is right. That is the
argument for reporting a range rather than a point.

### Where the wall actually is

Impact on a one-shot position is a single toll, and even 68 bp barely dents an
eleven-year return. The binding constraint is different: with a participation
cap of 10% of trailing ADV, above roughly **$2.7bn** the desired SPY position
no longer fits inside one day's permitted volume.

| AUM | Shares acquired | Total return |
|---|---|---|
| $100,000,000 | 591,014 of 591,014 (100%) | 303.81% |
| $1,000,000,000 | 5,910,149 of 5,910,149 (100%) | 303.66% |
| $10,000,000,000 | 15,949,333 of 59,101,490 (27%) | **81.91%** |

At $10bn the strategy never gets invested. Seventy-three percent of the account
sits in cash for eleven years, and the return collapses from 303% to 82%. **The
strategy did not stop working because it got expensive. It stopped working
because it could not be executed.**

A strategy that round-trips instead of holding would pay the impact toll on
every trip rather than once, so turnover and size compound: the wall arrives far
earlier than this.

**A backtest that does not name its assumed AUM has not reported a result.** It
has reported the limit as capital goes to zero, which is the one size nobody
trades.

## Installation

```bash
git clone https://github.com/dyjaden/event-driven-backtester.git
cd event-driven-backtester
python -m venv .venv && .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

To reproduce the exact figures in this README, install the pinned versions
instead: `python -m pip install -r requirements.txt`

Run the demo with `python -m backtester.engine`.

## Limitations

Kept deliberately explicit. A backtest without this section should not be
believed.

- **Survivorship-biased data.** Yahoo Finance serves currently listed tickers
  only, so every delisting is silently absent and the final near-total loss
  never enters the calculation. Separately, choosing a historical universe from
  today's index membership is look-ahead, because additions typically follow a
  run-up. Migrating to CRSP for point-in-time membership and delisting returns.
- **The impact coefficient is an assumption, not a measurement.** Empirical
  estimates span roughly 0.3 to 1.5. Nobody audits this number and halving it
  roughly halves the modelled cost, so results are reported across a range.
- **The same is true of the spread.** Reported across 1-5 bp rather than at a
  single value.
- **Impact is a participation cap, not an execution schedule.** Almgren-Chriss
  solves for an optimal trajectory that trades impact against timing risk. This
  models the impact law and clips the order; it does not schedule it.
- **A 21-bar warm-up is required when impact is enabled.** Trailing ADV and
  volatility do not exist on the first bar, so a strategy that trades there
  would silently pay zero impact. The alternative, estimating them from the
  full sample, is look-ahead.
- **Same-bar fills.** Orders fill at the close of the bar the decision was made
  on, which is mildly optimistic. A conservative design fills at the next open.
- **Cash drag.** Integer share counts leave a small uninvested remainder that
  never compounds, so total return sits slightly below the asset's own.
- **One symbol.** Multi-asset support is next.

## Tests

```bash
python -m pytest -q
```

59 tests. The ones worth reading are the invariants rather than the happy
paths: that a zero-cost fill cannot change equity, that impact scales as the
square root of size, that a cost is never charged twice, and that a bar-zero
trade pays no impact, which pins the warm-up boundary so that "fixing" it with
a full-sample estimate fails loudly instead of quietly introducing look-ahead.

## License

MIT.