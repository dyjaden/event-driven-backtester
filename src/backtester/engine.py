"""The event loop. Time advances here and nowhere else."""
from __future__ import annotations
import queue
from typing import Optional

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

def _demo() -> None:
    from .metrics import summary
    from .portfolio import Portfolio
    from .execution import NaiveExecutionHandler
    from .strategy import BuyAndHoldStrategy

    frame = pd.read_parquet("data/SPY_daily.parquet")
    data = HistoricBarHandler(frame, symbol="SPY")
    portfolio = Portfolio(initial_capital=100_000.0)

    bt = Backtest(
        data=data,
        strategy=BuyAndHoldStrategy(data),
        portfolio=portfolio,
        execution=NaiveExecutionHandler(),
    )
    counts = bt.run()

    equity = portfolio.equity_curve()
    print(f"bars     : {counts['MarketEvent']}")
    print(f"fills    : {counts['FillEvent']}")
    print(f"final eq : ${equity.iloc[-1]:,.2f}")
    print()
    for k, v in summary(equity).items():
        if k == "n_periods":
            print(f"{k:>16}: {v}")
        elif "return" in k or "drawdown" in k or k in ("cagr", "hit_rate",
                                                       "ann_volatility"):
            print(f"{k:>16}: {v:>8.2%}")
        else:
            print(f"{k:>16}: {v:>8.2f}")

    from .plotting import plot_equity_vs_benchmark
    path = plot_equity_vs_benchmark(equity, frame["close"])
    print(f"\nchart -> {path}")


if __name__ == "__main__":
    _demo()