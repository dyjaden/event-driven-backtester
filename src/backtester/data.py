"""Bar streaming. The ONLY component allowed to touch raw data."""
from __future__ import annotations
import queue
import pandas as pd
from .events import MarketEvent

REQUIRED = ["open", "high", "low", "close", "volume"]


class HistoricBarHandler:
    """Streams bars one at a time and refuses to reveal the future.

    `latest_bars` can never return data past the current cursor, which is
    what makes look-ahead bias structurally impossible rather than merely
    something you tried to remember not to do.
    """

    def __init__(self, frame: pd.DataFrame, symbol: str) -> None:
        missing = [c for c in REQUIRED if c not in frame.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError("index must be a DatetimeIndex")
        if not frame.index.is_monotonic_increasing:
            raise ValueError("index must be sorted ascending")

        self.frame = frame
        self.symbol = symbol
        self._i = -1
        self.continue_backtest = True

    @property
    def current_time(self):
        if self._i < 0:
            return None
        return self.frame.index[self._i]

    def update_bars(self, events: queue.Queue) -> None:
        """Advance one bar and emit a MarketEvent."""
        self._i += 1
        if self._i >= len(self.frame):
            self.continue_backtest = False
            return
        events.put(MarketEvent(timestamp=self.frame.index[self._i]))

    def latest_bars(self, n: int = 1) -> pd.DataFrame:
        """The most recent n bars, inclusive of the current one. Never more."""
        if self._i < 0:
            return self.frame.iloc[0:0]
        start = max(0, self._i - n + 1)
        return self.frame.iloc[start:self._i + 1]

    def current_price(self, field: str = "close") -> float:
        return float(self.frame.iloc[self._i][field])