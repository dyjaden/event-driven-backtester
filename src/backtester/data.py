"""Bar streaming. The ONLY component allowed to touch raw data."""
from __future__ import annotations
import queue

import numpy as np
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

    def current_price(self, symbol: str | None = None, field: str = "close") -> float:
        if symbol is not None and symbol != self.symbol:
            raise KeyError(f"this handler serves {self.symbol!r}, not {symbol!r}")
        return float(self.frame.iloc[self._i][field])

    def adv(self, symbol: str | None = None, window: int = 21) -> float:
        """Trailing average daily volume, in shares.

        Uses bars up to and INCLUDING the current one. Never the full sample.
        """
        if symbol is not None and symbol != self.symbol:
            raise KeyError(f"this handler serves {self.symbol!r}, not {symbol!r}")
        bars = self.latest_bars(window)
        if bars.empty or "volume" not in bars.columns:
            return float("nan")
        return float(bars["volume"].mean())

    def trailing_volatility(self, symbol: str | None = None, window: int = 21) -> float:
        """Trailing daily log-return standard deviation."""
        if symbol is not None and symbol != self.symbol:
            raise KeyError(f"this handler serves {self.symbol!r}, not {symbol!r}")
        bars = self.latest_bars(window + 1)
        if len(bars) < 3:
            return float("nan")
        returns = np.log(bars["close"]).diff().dropna()
        return float(returns.std(ddof=1))


class PanelBarHandler:
    """N symbols on one shared DatetimeIndex. Day 9's cross-sectional handler.

    Same contract as HistoricBarHandler -- `update_bars`, `continue_backtest`,
    `current_time`, and the symbol-aware trio -- plus the cross-sectional
    queries a strategy needs. The cursor is the ONLY source of time, and this
    class is the only component allowed to hold the full matrix: history
    reaches a strategy through `latest_closes(n)` or not at all.

    Columns are symbols (strings). `close` is a total-return price per name
    (see `crsp.to_panel`); `volume` is share volume on the SAME price basis,
    so close * volume is the dollars that actually traded. NaN means "no
    bar": before listing, after delisting, or a genuine gap.

    The corpse policy lives here: `current_price` falls back to the LAST
    KNOWN close, so a held name that has stopped printing is marked -- and
    exits -- at its final value, which in CIZ already contains the delisting
    return. The raw matrix keeps its NaNs so statistics see missing bars as
    missing; the execution handler charges no impact when they are.
    """

    def __init__(self, close: pd.DataFrame, volume: pd.DataFrame) -> None:
        if not isinstance(close.index, pd.DatetimeIndex):
            raise TypeError("index must be a DatetimeIndex")
        if not close.index.equals(volume.index):
            raise ValueError("close and volume must share the same index")
        if list(close.columns) != list(volume.columns):
            raise ValueError("close and volume must share the same columns")
        if not close.index.is_monotonic_increasing:
            raise ValueError("index must be sorted ascending")
        if close.index.has_duplicates:
            raise ValueError("duplicate timestamps")
        if close.columns.has_duplicates:
            raise ValueError("duplicate symbols")

        self._close = close
        self._volume = volume
        self._marks = close.ffill()          # last-known-price fallback
        self._last_valid = {s: close[s].last_valid_index()
                            for s in close.columns}
        # positional index per symbol: scalar .iloc[row, pos] lookups instead
        # of building a 1,000-wide row Series on every price query. Same
        # answers; the difference is minutes of wall clock on an 11-year run.
        self._pos = {s: i for i, s in enumerate(close.columns)}
        self._i = -1
        self.continue_backtest = True

    # ------------------------------------------------------------- time
    @property
    def current_time(self):
        if self._i < 0:
            return None
        return self._close.index[self._i]

    def update_bars(self, events: queue.Queue) -> None:
        """Advance one bar (one DATE, for every symbol at once) and emit ONE
        MarketEvent. 'A new date exists' is symbol-agnostic, which is why the
        event loop did not have to change for the cross-section."""
        self._i += 1
        if self._i >= len(self._close):
            self.continue_backtest = False
            return
        events.put(MarketEvent(timestamp=self._close.index[self._i]))

    # ---------------------------------------------------------- plumbing
    def _col(self, symbol) -> int:
        """Symbol -> column position, with the loud failures."""
        if symbol is None:
            raise ValueError("a panel handler needs a symbol; none was given")
        pos = self._pos.get(symbol)
        if pos is None:
            raise KeyError(f"unknown symbol {symbol!r}")
        return pos

    def _require_started(self) -> None:
        if self._i < 0 or self._i >= len(self._close):
            raise RuntimeError("no current bar: the run has not started or "
                               "has already finished")

    # ------------------------------------------------------ cross-section
    def symbols(self) -> list:
        """Names with a price on the current bar."""
        self._require_started()
        row = self._close.iloc[self._i]
        return list(row.index[row.notna()])

    def has_bar(self, symbol) -> bool:
        self._require_started()
        return bool(pd.notna(self._close.iloc[self._i, self._col(symbol)]))

    def is_dead(self, symbol) -> bool:
        """Past the last bar this name will ever print: the series has ended."""
        self._require_started()
        last = self._last_valid[self._close.columns[self._col(symbol)]]
        return last is None or self._close.index[self._i] > last

    def latest_closes(self, n: int = 1) -> pd.DataFrame:
        """The most recent n rows of the close matrix, current bar included.
        Never more. This is the only window a strategy is allowed to see."""
        if self._i < 0:
            return self._close.iloc[0:0]
        start = max(0, self._i - n + 1)
        return self._close.iloc[start:self._i + 1]

    # ------------------------------------------------------------ pricing
    def current_price(self, symbol: str | None = None,
                      field: str = "close") -> float:
        if field != "close":
            raise ValueError("a panel carries close and volume only")
        self._require_started()
        px = self._marks.iloc[self._i, self._col(symbol)]
        if pd.isna(px):
            raise ValueError(
                f"{symbol!r} has no price history by "
                f"{self._close.index[self._i].date()}")
        return float(px)

    def adv(self, symbol: str | None = None, window: int = 21) -> float:
        """Trailing average daily volume in shares. Missing bars are absent,
        not zero; an all-missing window returns NaN, and the execution
        handler's policy for NaN statistics is to charge no impact."""
        self._require_started()
        pos = self._col(symbol)
        start = max(0, self._i - window + 1)
        vol = self._volume.iloc[start:self._i + 1, pos]
        return float(vol.mean()) if vol.notna().any() else float("nan")

    def trailing_volatility(self, symbol: str | None = None,
                            window: int = 21) -> float:
        """Trailing daily log-return standard deviation, from RAW closes so
        that gaps stay gaps rather than fake zero-volatility days."""
        self._require_started()
        pos = self._col(symbol)
        start = max(0, self._i - window)
        closes = self._close.iloc[start:self._i + 1, pos]
        returns = np.log(closes).diff().dropna()
        if len(returns) < 2:
            return float("nan")
        return float(returns.std(ddof=1))