"""Repair the dollar volume in the yfinance SPY file. Run once.

yfinance with `auto_adjust=True` back-adjusts OHLC for dividends and leaves
`Volume` untouched. The two columns end up on different price bases, so

    close * volume

is a dollar volume that never traded -- roughly 17% low at the start of an
eleven-year SPY sample, exact at the end, because the adjustment is anchored
to the most recent price.

The impact model sizes orders in adjusted shares and compares them against
that ADV, so participation is overstated and the cap binds too early. This
script rebuilds `volume` so dollar volume is invariant, exactly as
`backtester.crsp.to_engine_frame` does for CRSP. Same law, two vendors.

    python scripts/fix_yfinance_volume.py

Writes data/SPY_daily_dollarvol.parquet. The original file is not touched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA = Path("data")
SRC = DATA / "SPY_daily.parquet"
OUT = DATA / "SPY_daily_dollarvol.parquet"


def corrected(adjusted: pd.DataFrame, raw_close: pd.Series) -> pd.DataFrame:
    """Rebuild `volume` so `close * volume` equals the dollars that traded.

    `adjusted` is the engine frame as it stands: back-adjusted OHLC, raw
    volume. `raw_close` is the unadjusted close for the same dates.

    Prices are left alone. They are already the right series for compounding
    returns -- the defect is that volume is on the other basis.
    """
    raw_close = raw_close.reindex(adjusted.index)
    if raw_close.isna().any():
        missing = int(raw_close.isna().sum())
        raise ValueError(f"no raw close for {missing} of {len(adjusted)} bars")
    if not (raw_close > 0).all():
        raise ValueError("non-positive raw close")

    factor = raw_close / adjusted["close"]        # raw basis -> adjusted basis
    out = adjusted.copy()
    out["volume"] = adjusted["volume"] * factor

    check = out["close"] * out["volume"]
    want = raw_close * adjusted["volume"]
    if not ((check - want).abs() <= want.abs() * 1e-9).all():
        raise AssertionError("dollar volume not preserved -- refusing to write")
    out.attrs["dollar_volume_corrected"] = True
    return out


def main() -> None:
    import yfinance as yf

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    adjusted = pd.read_parquet(args.src)
    start = adjusted.index[0].strftime("%Y-%m-%d")
    end = (adjusted.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # auto_adjust=False keeps BOTH series: `Close` raw, `Adj Close` adjusted.
    raw = yf.download(args.symbol, start=start, end=end,
                      auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw_close = raw["Close"]
    raw_close.index = pd.DatetimeIndex(raw_close.index).tz_localize(None)

    out = corrected(adjusted, raw_close)

    factor = (raw_close.reindex(adjusted.index) / adjusted["close"])
    print(f"{args.symbol}: {len(out):,} bars, "
          f"{out.index[0].date()} to {out.index[-1].date()}")
    print(f"  adjustment factor  first {factor.iloc[0]:.4f}   "
          f"last {factor.iloc[-1]:.4f}")
    print(f"  dollar volume was understated by "
          f"{(1 - 1/factor.iloc[0]) * 100:.1f}% on the first bar")
    print(f"  every ADV-derived number scales by ~{factor.iloc[0]:.2f}x, "
          f"so the capacity wall moves out by the same factor")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"  wrote {args.out}  (original untouched)")


if __name__ == "__main__":
    main()
