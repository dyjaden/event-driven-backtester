"""Fabricate CIZ-shaped Parquet files so the Monday pipeline can be rehearsed.

This exists because WRDS access and the code that consumes it arrive on
different days. Waiting for the data to test the loader is the wrong order:
it puts every bug in the pull, the adapter and the report into the same
debugging session, on a clock, against a database that times out.

    python scripts/make_fake_crsp.py
    python scripts/crsp_report.py --data data/_dryrun

Writes into data/_dryrun/ so it can never be confused with the real pull.
The numbers are fiction. The SHAPES are the contract, and the shapes are
what the code is being tested against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/_dryrun")


def fake_spy(yf_frame: pd.DataFrame, permno: int = 84398,
             div_yield: float = 0.017, anchor: float = 205.43) -> pd.DataFrame:
    """CIZ daily rows for one ETF, derived from the yfinance frame.

    The point is the property that breaks naive code: `dlyret` is the total
    return, and `dlyclose` is a RAW price that lags it by the dividend
    stream. Anything that compounds `dlyclose` gets a visibly wrong answer.
    """
    dates = yf_frame.index
    ret = (yf_frame["close"] / yf_frame["close"].shift(1) - 1.0)
    ret.iloc[0] = np.nan

    daily_div = div_yield / 252
    r = ret.fillna(0.0).to_numpy()
    raw = np.empty(len(dates))
    raw[0] = anchor
    for i in range(1, len(dates)):
        raw[i] = raw[i - 1] * (1 + r[i] - daily_div)

    scale = raw / yf_frame["close"].to_numpy()
    return pd.DataFrame({
        "permno": permno,
        "dlycaldt": dates,
        "dlyopen": yf_frame["open"].to_numpy() * scale,
        "dlyhigh": yf_frame["high"].to_numpy() * scale,
        "dlylow": yf_frame["low"].to_numpy() * scale,
        "dlyclose": raw,
        "dlyvol": yf_frame["volume"].to_numpy().astype(float),
        "dlyret": ret.to_numpy(),
        "dlyretx": ret.to_numpy() - daily_div,
        "dlyfacprc": 1.0,
        "dlydelflg": "N",
        "shrout": 900_000.0,
    }).reset_index(drop=True)


def fake_universe(dates: pd.DatetimeIndex, n: int = 60, dead: int = 8,
                  seed: int = 8, svb: int = 0
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small universe where some names die inside the window.

    `dead` of them end early with a near-total loss and a delisting flag,
    and their index membership ends on the same day. That is the only
    structure the survivorship measurement actually needs, and it is the
    structure that makes the two arms differ.
    """
    rng = np.random.default_rng(seed)
    permnos = list(range(10_001, 10_001 + n))
    rets = pd.DataFrame(rng.normal(0.0004, 0.016, size=(len(dates), n)),
                        index=dates, columns=permnos)

    rows, members = [], []
    for j, permno in enumerate(permnos):
        end = len(dates) - 1
        if j < dead:
            end = int(len(dates) * (0.25 + 0.6 * j / dead))
            # `svb` of the deaths carry NO computable delisting return, the
            # way ~3% of real CIZ delistings (SVB included) do: flag Y,
            # dlyret NaN, price flat into the grave.
            rets.iloc[end, j] = np.nan if j < svb else -0.93
        series = rets.iloc[:end + 1, j]
        close = 50.0 * (1 + series.fillna(0.0)).cumprod()
        flags = ["N"] * end + (["Y"] if j < dead else ["N"])
        rows.append(pd.DataFrame({
            "permno": permno,
            "dlycaldt": dates[:end + 1],
            "dlyret": series.to_numpy(),
            "dlyclose": close.to_numpy(),
            "dlyvol": rng.uniform(1e5, 5e6, end + 1),
            "dlydelflg": flags,
        }))
        members.append({"permno": permno,
                        "mbrstartdt": dates[0], "mbrenddt": dates[end]})

    return (pd.concat(rows, ignore_index=True), pd.DataFrame(members))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/SPY_daily.parquet")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    yf_frame = pd.read_parquet(args.src)
    yf_frame.to_parquet(out / "SPY_daily.parquet")
    fixed = Path("data/SPY_daily_dollarvol.parquet")
    if fixed.exists():
        pd.read_parquet(fixed).to_parquet(out / "SPY_daily_dollarvol.parquet")

    spy = fake_spy(yf_frame)
    spy.to_parquet(out / "crsp_spy_daily.parquet")

    universe, members = fake_universe(pd.DatetimeIndex(yf_frame.index))
    universe.to_parquet(out / "crsp_universe_daily.parquet")
    members.to_parquet(out / "crsp_sp500_membership.parquet")

    print(f"wrote {out}/ -- FICTION, for rehearsal only")
    print(f"  crsp_spy_daily.parquet          {len(spy):,} rows, 1 permno")
    print(f"  crsp_universe_daily.parquet     {len(universe):,} rows, "
          f"{universe['permno'].nunique()} permnos, "
          f"{(universe['dlydelflg'] == 'Y').sum()} delistings")
    print(f"  crsp_sp500_membership.parquet   {len(members):,} spells")
    print(f"\nnow:  python scripts/crsp_report.py --data {out}")


if __name__ == "__main__":
    main()