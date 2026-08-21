"""The event loop. Time advances here and nowhere else."""
from __future__ import annotations
import queue

import pandas as pd

from .data import HistoricBarHandler
from .events import FillEvent, MarketEvent, OrderEvent, SignalEvent
from .strategy import Strategy


class Backtest:
    """Drives a single backtest from the first bar to the last.

    Outer loop advances one bar. Inner loop drains every event that bar
    produced. Time cannot advance while the queue is non-empty, which is what
    prevents an order from being filled with information that did not exist
    when it was created.
    """

    def __init__(
        self,
        data: HistoricBarHandler,
        strategy: Strategy,
        portfolio=None,      # Day 4
        execution=None,      # Day 6
        verbose: bool = False,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.verbose = verbose

        self.events: queue.Queue = queue.Queue()
        self.counts = {"MarketEvent": 0, "SignalEvent": 0,
                       "OrderEvent": 0, "FillEvent": 0}

    def _handle(self, event) -> None:
        name = type(event).__name__
        self.counts[name] = self.counts.get(name, 0) + 1

        if isinstance(event, MarketEvent):
            self.strategy.on_market(event, self.events)
            if self.portfolio is not None:
                self.portfolio.mark_to_market(event, self.data)

        elif isinstance(event, SignalEvent):
            if self.portfolio is not None:
                self.portfolio.on_signal(event, self.events, self.data)

        elif isinstance(event, OrderEvent):
            if self.execution is not None:
                self.execution.on_order(event, self.events, self.data)

        elif isinstance(event, FillEvent):
            if self.portfolio is not None:
                self.portfolio.on_fill(event)

        else:
            raise TypeError(f"unknown event type: {name}")

        if self.verbose:
            print(f"  {event.timestamp.date()}  {name}")

    def run(self) -> dict:
        """Outer loop: one iteration per bar."""
        while self.data.continue_backtest:
            self.data.update_bars(self.events)

            # Inner loop: drain everything this bar produced.
            while True:
                try:
                    event = self.events.get(block=False)
                except queue.Empty:
                    break
                self._handle(event)

        return self.counts


def _run_pf(frame, strategy_cls, execution, capital=100_000.0):
    """One full backtest. Returns the Portfolio, which holds BOTH the equity
    curve and the fills, so no caller ever has to run the same thing twice."""
    from .portfolio import Portfolio
    data = HistoricBarHandler(frame, symbol="SPY")
    pf = Portfolio(initial_capital=capital)
    Backtest(data, strategy_cls(data), pf, execution).run()
    return pf


def _run(frame, strategy_cls, execution, capital=100_000.0):
    """Equity curve only. Most callers want just this."""
    return _run_pf(frame, strategy_cls, execution, capital).equity_curve()


def _demo() -> None:
    """Day 6 asked whether costs kill the strategy. Day 7 asks how big it can get."""
    from functools import partial

    from .costs import (HalfSpreadSlippage, PerShareCommission,
                        SquareRootImpact)
    from .execution import NaiveExecutionHandler, SimulatedExecutionHandler
    from .metrics import summary
    from .strategy import BuyAndHoldStrategy

    frame = pd.read_parquet("data/SPY_daily.parquet")

    # Trailing ADV and volatility do not exist on bar 0. A strategy that
    # trades there gets NaN inputs and therefore zero impact, silently.
    WARMUP = 21
    BH = partial(BuyAndHoldStrategy, warmup=WARMUP)

    def sized(coef, cap=None):
        return SimulatedExecutionHandler(
            commission=PerShareCommission(),
            slippage=HalfSpreadSlippage(spread_bps=1.0),
            impact=SquareRootImpact(coefficient=coef),
            max_participation=cap,
        )

    print(f"CAPACITY CURVE -- buy and hold, impact coefficient 1.0, "
          f"warmup {WARMUP} bars")
    print(f"{'AUM':>16} {'total ret':>11} {'impact $':>16} {'bp of AUM':>10} {'ratio':>7}")
    prev = None
    for aum in (1e5, 1e6, 1e7, 1e8, 1e9, 1e10):
        pf = _run_pf(frame, BH, sized(1.0), capital=aum)
        eq = pf.equity_curve()
        paid = sum(f.impact_cost for f in pf.fills)
        bp = paid / aum * 1e4
        ratio = f"{bp / prev:.2f}x" if prev else "  --"
        print(f"${aum:>15,.0f} {eq.iloc[-1]/aum-1:>11.2%} ${paid:>15,.0f} "
              f"{bp:>10.2f} {ratio:>7}")
        prev = bp

    print("\nSAME AUM, COEFFICIENT SWEEP at $1B (the parameter nobody audits)")
    for coef in (0.3, 0.5, 1.0, 1.5):
        pf = _run_pf(frame, BH, sized(coef), capital=1e9)
        paid = sum(f.impact_cost for f in pf.fills)
        print(f"  Y = {coef:>3.1f} : total return "
              f"{pf.equity_curve().iloc[-1]/1e9-1:>8.2%}   impact ${paid:>14,.0f}")

    print("\nPARTICIPATION CAP at 10% of trailing ADV")
    for aum in (1e8, 1e9, 1e10):
        free = _run_pf(frame, BH, sized(1.0), capital=aum)
        cap = _run_pf(frame, BH, sized(1.0, 0.10), capital=aum)
        got = cap.positions.get("SPY", 0)
        want = free.positions.get("SPY", 0)
        print(f"  ${aum:>14,.0f} : uncapped {free.equity_curve().iloc[-1]/aum-1:>8.2%}   "
              f"capped {cap.equity_curve().iloc[-1]/aum-1:>8.2%}   "
              f"shares {got:,} of {want:,} ({got/want if want else 0:.0%})")

    # Headline baseline stays UNWARMED so it remains comparable to Days 1-6.
    print("\nBASELINE -- zero cost, no warmup (the Day 4 result, unchanged)")
    base = _run(frame, BuyAndHoldStrategy, NaiveExecutionHandler())
    for k, v in summary(base).items():
        print(f"{k:>16}: {v}")


if __name__ == "__main__":
    _demo()