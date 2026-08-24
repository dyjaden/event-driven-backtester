"""Pull CRSP CIZ data from WRDS and cache it to Parquet. Run once per window.

Deliberately OUTSIDE src/backtester/. The package must import with no
credentials and no network, and `pip install -e .` must not ship anything
that opens a database connection.

    python scripts/pull_crsp.py --start 2015-01-01 --end 2025-12-31

Everything lands in data/, which was gitignored on day one. CRSP is licensed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

OUT = Path("data")

# CIZ replaced SIZ in Feb 2025. crsp.dsf is the legacy SIZ daily file and no
# longer updates; crsp.dsf_v2 (a.k.a. crsp.stkdlysecuritydata) is the live one.
DAILY = "crsp.dsf_v2"
SECINFO = "crsp.stksecurityinfohist"
SP500 = "crsp_a_indexes.dsp500list_v2"

# The standard US common-stock screen, in CIZ's flag columns. SIZ said
# `shrcd in (10, 11) and exchcd in (1, 2, 3)`; that is five flags now.
#
# The screen must not delete the corpses (found 24 Aug 2026, live data).
# On the delisting day the security's info spell flips to placeholders --
# SVB's last row reads sharetype 'N/A', securitysubtype 'UNK',
# conditionaltype 'N/A', tradingstatusflg 'D' -- so EVERY condition below
# fails on exactly the row that carries dlydelflg = 'Y' and the delisting
# return. Screened naively, the file contains zero delistings and the
# survivorship number quietly loses the final day of every dead company.
# So: a row passes if it clears the screen, OR it is a delisting row for a
# security that cleared the screen in the 30 days before it died. The
# lookback covers halt gaps between last trade and formal delisting; a
# stock halted for more than 30 days before delisting would still be
# missed, which is rare and accepted.
COMMON_STOCK = """
      AND (
            (    s.sharetype       = 'NS'
             AND s.securitytype    = 'EQTY'
             AND s.securitysubtype = 'COM'
             AND s.usincflg        = 'Y'
             AND s.issuertype      IN ('ACOR', 'CORP')
             AND s.primaryexch     IN ('N', 'A', 'Q')
             AND s.conditionaltype IN ('RW', 'NW')
             AND s.tradingstatusflg = 'A')
         OR (d.dlydelflg = 'Y' AND EXISTS (
                 SELECT 1
                 FROM crsp.stksecurityinfohist s2
                 WHERE s2.permno = d.permno
                   AND s2.secinfostartdt <  d.dlycaldt
                   AND s2.secinfoenddt   >= d.dlycaldt - INTERVAL '30 days'
                   AND s2.sharetype       = 'NS'
                   AND s2.securitytype    = 'EQTY'
                   AND s2.securitysubtype = 'COM'
                   AND s2.usincflg        = 'Y'
                   AND s2.issuertype      IN ('ACOR', 'CORP')
                   AND s2.primaryexch     IN ('N', 'A', 'Q')
                   AND s2.conditionaltype IN ('RW', 'NW')
                   AND s2.tradingstatusflg = 'A'))
      )
"""


def find_permno(db, ticker: str) -> pd.DataFrame:
    """Every security that has EVER carried this ticker.

    More than one row is the normal case, not an error. Tickers are recycled
    after a delisting, which is why the rest of this file speaks PERMNO.
    """
    return db.raw_sql(f"""
        SELECT permno, ticker, issuernm, securitybegdt, securityenddt,
               sharetype, securitytype, securitysubtype, issuertype, primaryexch
        FROM {SECINFO}
        WHERE ticker = %(ticker)s
        ORDER BY securitybegdt
    """, params={"ticker": ticker.upper()},
        date_cols=["securitybegdt", "securityenddt"])


def pull_securities(db, permnos, start, end) -> pd.DataFrame:
    """Daily bars for specific PERMNOs, with NO common-stock screen.

    The screen is omitted on purpose: SPY is an ETF, so `securitysubtype='COM'`
    excludes it and you get an empty frame with no error.
    """
    return db.raw_sql(f"""
        SELECT permno, dlycaldt, dlyopen, dlyhigh, dlylow, dlyclose,
               dlyvol, dlyret, dlyretx, dlyfacprc, dlydelflg, shrout
        FROM {DAILY}
        WHERE permno IN %(permnos)s
          AND dlycaldt BETWEEN %(start)s AND %(end)s
        ORDER BY permno, dlycaldt
    """, params={"permnos": tuple(int(p) for p in permnos),
                 "start": start, "end": end},
        date_cols=["dlycaldt"])


def pull_universe(db, start, end) -> pd.DataFrame:
    """Every US common stock's daily return over the window.

    The join to stksecurityinfohist is date-bounded on BOTH sides. Join on
    permno alone and a company that was an ADR in 2015 and common stock in
    2020 is either wholly in or wholly out -- which is look-ahead, applied to
    the universe rather than to prices.

    Delisting rows are admitted through a carve-out in COMMON_STOCK (see the
    comment there): their spell values are placeholders, so the plain screen
    would silently drop every one. dlyret on a delisting row is CRSP's
    delisting return when computable, and NA when it is not.
    """
    return db.raw_sql(f"""
        SELECT d.permno, d.dlycaldt, d.dlyret, d.dlyclose, d.dlyvol, d.dlydelflg
        FROM {DAILY} AS d
        INNER JOIN {SECINFO} AS s
          ON  d.permno = s.permno
          AND s.secinfostartdt <= d.dlycaldt
          AND d.dlycaldt <= s.secinfoenddt
        WHERE d.dlycaldt BETWEEN %(start)s AND %(end)s
        {COMMON_STOCK}
        ORDER BY d.permno, d.dlycaldt
    """, params={"start": start, "end": end}, date_cols=["dlycaldt"])


def pull_sp500_membership(db, start, end) -> pd.DataFrame:
    """Point-in-time S&P 500 membership. The whole reason for the migration."""
    return db.raw_sql(f"""
        SELECT permno, mbrstartdt, mbrenddt
        FROM {SP500}
        WHERE mbrenddt >= %(start)s AND mbrstartdt <= %(end)s
        ORDER BY permno, mbrstartdt
    """, params={"start": start, "end": end},
        date_cols=["mbrstartdt", "mbrenddt"])


def main() -> None:
    import wrds

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--universe", action="store_true",
                    help="also pull every US common stock (large, slow)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    db = wrds.Connection()

    names = find_permno(db, args.ticker)
    print(f"{args.ticker}: {len(names)} security-info rows")
    print(names.to_string(index=False))
    if names.empty:
        raise SystemExit(f"no permno found for {args.ticker}")

    permno = int(names["permno"].iloc[-1])
    bars = pull_securities(db, [permno], args.start, args.end)
    bars.to_parquet(OUT / f"crsp_{args.ticker.lower()}_daily.parquet")
    print(f"  permno {permno}: {len(bars):,} bars -> "
          f"{OUT / f'crsp_{args.ticker.lower()}_daily.parquet'}")

    members = pull_sp500_membership(db, args.start, args.end)
    members.to_parquet(OUT / "crsp_sp500_membership.parquet")
    print(f"  S&P 500 membership: {len(members):,} spells, "
          f"{members['permno'].nunique():,} distinct permnos")

    if args.universe:
        uni = pull_universe(db, args.start, args.end)
        uni.to_parquet(OUT / "crsp_universe_daily.parquet")
        print(f"  universe: {len(uni):,} rows, "
              f"{uni['permno'].nunique():,} permnos")

    db.close()


if __name__ == "__main__":
    main()