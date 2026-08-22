"""CRSP CIZ adapter. Turns WRDS rows into the frame the engine already eats.

Nothing in this module talks to WRDS. The pull lives in `scripts/pull_crsp.py`
because it needs credentials; this file needs only a DataFrame, which is what
makes every line of it testable without an account.

The central decision: CRSP's authoritative series is `dlyret`, not `dlyclose`.
`dlyclose` is the RAW price and drops by the split factor on a split. `dlyret`
is total return, including dividends and -- in CIZ -- the delisting return.
So the price series the engine sees is REBUILT from returns, and the volume
series is rescaled to keep dollar volume invariant. See `to_engine_frame`.
"""
from __future__ import annotations

import pandas as pd

# CIZ column names, as WRDS serves them (lowercase).
REQUIRED_CIZ = ["permno", "dlycaldt", "dlyclose", "dlyvol", "dlyret"]
OPTIONAL_CIZ = ["dlyopen", "dlyhigh", "dlylow", "dlydelflg", "ticker"]

ENGINE_COLUMNS = ["open", "high", "low", "close", "volume"]


def to_engine_frame(rows: pd.DataFrame, *, permno: int | None = None,
                    price_floor: float = 1e-8) -> pd.DataFrame:
    """One PERMNO of CIZ daily rows -> the OHLCV frame `HistoricBarHandler` wants.

    `close` is a TOTAL-RETURN INDEX anchored at the first raw close, not a
    price. Calling it `close` is a deliberate abuse: the engine's contract is
    five columns, and the quantity a strategy must compound is the return.

    `volume` is rescaled so that `close * volume` equals `dlyclose * dlyvol`
    on every single bar. Dollar volume is the physical quantity; share counts
    are an accounting convention that depends on the price basis. Get this
    wrong and the impact model prices participation against a market that
    never existed.
    """
    missing = [c for c in REQUIRED_CIZ if c not in rows.columns]
    if missing:
        raise ValueError(f"missing CIZ columns: {missing}")

    df = rows
    if permno is not None:
        df = df[df["permno"] == permno]
    permnos = df["permno"].unique()
    if len(permnos) != 1:
        raise ValueError(
            f"expected exactly one permno, got {len(permnos)}; "
            "pass permno= to select one")
    if df.empty:
        raise ValueError("no rows for that permno")

    df = df.sort_values("dlycaldt")
    if df["dlycaldt"].duplicated().any():
        raise ValueError("duplicate dates for a single permno")

    idx = pd.DatetimeIndex(pd.to_datetime(df["dlycaldt"].to_numpy()), name="date")

    raw_close = pd.Series(df["dlyclose"].to_numpy(dtype=float), index=idx)
    raw_volume = pd.Series(df["dlyvol"].to_numpy(dtype=float), index=idx)
    ret = pd.Series(df["dlyret"].to_numpy(dtype=float), index=idx)

    # CRSP has no return on a security's first day. That is a missing input,
    # not a zero-return day, but for an index anchored at that day it is the
    # same thing.
    ret = ret.fillna(0.0)

    anchor = raw_close.dropna()
    if anchor.empty or not (anchor.iloc[0] > 0):
        raise ValueError("no usable anchor price: first close is missing or <= 0")

    close = anchor.iloc[0] * (1.0 + ret).cumprod()

    # -100% is a valid RETURN and an invalid PRICE. Floor it so that the log
    # machinery downstream (trailing_volatility) never sees log(0) = -inf.
    close = close.clip(lower=price_floor)

    # factor converts a raw-basis quantity to the index basis.
    factor = (close / raw_close).where(raw_close > 0)

    out = pd.DataFrame(index=idx)
    for src, dst in (("dlyopen", "open"), ("dlyhigh", "high"), ("dlylow", "low")):
        if src in df.columns:
            col = pd.Series(df[src].to_numpy(dtype=float), index=idx) * factor
        else:
            col = close.copy()
        out[dst] = col.where(col > 0, close)
    out["close"] = close
    # close * volume == raw_close * raw_volume, exactly, every bar.
    out["volume"] = (raw_volume / factor).fillna(0.0)

    out.attrs["permno"] = int(permnos[0])
    out.attrs["delisted"] = bool(
        "dlydelflg" in df.columns and df["dlydelflg"].astype(str).str.upper()
        .isin(["Y", "TRUE", "1"]).any())
    return validate_engine_frame(out)

def validate_engine_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Assert the contract `HistoricBarHandler` will not assert for you.

    The handler checks columns and index ordering. It does not check for NaN,
    non-positive prices or negative volume, because on Day 1 the data came
    from one vendor and could not contain them. It can now.
    """
    missing = [c for c in ENGINE_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"missing engine columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending")
    if frame.index.has_duplicates:
        raise ValueError("duplicate timestamps")
    if frame[ENGINE_COLUMNS].isna().to_numpy().any():
        raise ValueError("NaN in OHLCV -- the engine has no gap handling")
    if not (frame[["open", "high", "low", "close"]].to_numpy() > 0).all():
        raise ValueError("non-positive price")
    if (frame["volume"].to_numpy() < 0).any():
        raise ValueError("negative volume")
    return frame


def delisting_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Every permno whose series ends in a delisting, and its final return.

    In CIZ the delisting return is already inside `dlyret`. The SIZ workflow
    merged `crsp.dsedelist` separately; copying that code into a CIZ pull
    either double-counts or -- more often -- silently finds nothing, because
    the table it reads no longer updates.
    """
    if "dlydelflg" not in rows.columns:
        raise ValueError("no dlydelflg column: this is not CIZ daily data")
    flag = rows["dlydelflg"].astype(str).str.upper().isin(["Y", "TRUE", "1"])
    hits = rows[flag]
    if hits.empty:
        return pd.DataFrame(columns=["permno", "dlycaldt", "dlyret"])
    return (hits[["permno", "dlycaldt", "dlyret"]]
            .sort_values("dlycaldt")
            .reset_index(drop=True))


def membership_mask(dates: pd.DatetimeIndex, permnos, membership: pd.DataFrame
                    ) -> pd.DataFrame:
    """Boolean frame: was this permno in the index on this date?

    `membership` is `crsp_a_indexes.dsp500list_v2`: permno, mbrstartdt,
    mbrenddt. This is the point-in-time universe, and it is the entire reason
    the migration was worth doing.
    """
    for col in ("permno", "mbrstartdt", "mbrenddt"):
        if col not in membership.columns:
            raise ValueError(f"membership is missing {col}")

    permnos = list(permnos)
    mask = pd.DataFrame(False, index=dates, columns=permnos)
    for permno, start, end in membership[
            ["permno", "mbrstartdt", "mbrenddt"]].itertuples(index=False):
        if permno not in mask.columns:
            continue
        start = pd.Timestamp(start)
        end = pd.Timestamp(end) if pd.notna(end) else dates[-1]
        sel = (dates >= start) & (dates <= end)
        mask.loc[sel, permno] |= True
    return mask


def equal_weight_index(returns: pd.DataFrame,
                       mask: pd.DataFrame | None = None) -> pd.Series:
    """Growth of 1 in an equally weighted, daily-rebalanced basket.

    `returns` is wide: one column per permno, one row per date. `mask` selects
    who is in the basket on each date; pass None to hold every column on every
    date, which is the survivor-biased arm.

    This is a measurement of the DATA, not a backtest. It has no costs, no
    sizing and no engine, and the guide says so out loud rather than letting
    the number look like a strategy result.
    """
    if mask is None:
        eligible = returns
    else:
        eligible = returns.where(mask.reindex_like(returns).fillna(False))
    daily = eligible.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + daily).cumprod()