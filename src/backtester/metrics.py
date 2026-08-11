"""Performance metrics. Written BEFORE any strategy exists, on purpose.

If you build the strategy first, you will unconsciously shape the metric to
flatter it. Defining "good" while you have nothing to defend is the only way
to keep these honest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- returns
def simple_returns(prices: pd.Series) -> pd.Series:
    """(P_t / P_{t-1}) - 1.  Use for anything that aggregates ACROSS assets.

    fill_method=None is load-bearing: the pandas default pads a missing price
    forward, which turns a data gap into a fabricated 0% return day.
    """
    return prices.pct_change(fill_method=None).dropna()


def log_returns(prices: pd.Series) -> pd.Series:
    """ln(P_t / P_{t-1}).  Use for anything that aggregates ACROSS TIME."""
    return np.log(prices / prices.shift(1)).dropna()


def total_return(prices: pd.Series) -> float:
    return float(prices.iloc[-1] / prices.iloc[0] - 1.0)


def cagr(prices: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Geometric mean annual growth. NOT the average of yearly returns."""
    n = len(prices) - 1
    if n <= 0:
        return np.nan
    growth = prices.iloc[-1] / prices.iloc[0]
    if growth <= 0:
        return np.nan
    return float(growth ** (periods_per_year / n) - 1.0)


# ------------------------------------------------------------ risk / ratio
def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Sample std (ddof=1) scaled by sqrt(time)."""
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                 periods_per_year: int = TRADING_DAYS) -> float:
    """Annualised Sharpe. `risk_free` is an ANNUAL rate, converted per-period.

    Returns nan when volatility is effectively zero: a riskless series has no
    Sharpe, and silently returning a huge number would poison every downstream
    comparison.
    """
    if len(returns) < 2:
        return np.nan
    rf_per_period = (1 + risk_free) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    sd = excess.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Percentage below the running peak, at every point in time."""
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline. Path dependent: order matters."""
    if len(equity) == 0:
        return np.nan
    return float(drawdown_series(equity).min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(cagr(equity, periods_per_year) / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Share of periods with a positive return. Says nothing about magnitude."""
    if len(returns) == 0:
        return np.nan
    return float((returns > 0).sum() / len(returns))


def summary(equity: pd.Series, risk_free: float = 0.0,
            periods_per_year: int = TRADING_DAYS) -> dict:
    r = simple_returns(equity)
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "ann_volatility": annualized_volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, risk_free, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, periods_per_year),
        "hit_rate": hit_rate(r),
        "n_periods": len(equity),
    }