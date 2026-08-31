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
    """Growth over the whole series: last over first, minus one."""
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
    """CAGR over |max drawdown|; NaN when the drawdown is zero or undefined."""
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
    """Every headline metric of an equity curve, in one dict."""
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

# ------------------------------------------------- multiple-testing honesty
# Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio". The best
# Sharpe among N tried configurations is the maximum of N noisy numbers, and
# the maximum of pure noise grows like sqrt(2 ln N). These two functions ask
# the only honest question about a swept result: is the observed best bigger
# than the best you would expect from N lottery tickets?
#
# Everything here works in PER-PERIOD (daily) Sharpe units, because the PSR
# variance formula is stated per observation. Convert annualised Sharpes by
# dividing by sqrt(periods_per_year) before calling.

_EULER_GAMMA = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """E[max Sharpe] under the null that every true Sharpe is zero.

    `var_trials` is the cross-trial variance of the (per-period) Sharpe
    estimates -- the spread of the lottery tickets. N=1 returns 0.0: one
    trial is not a search, so nothing is deflated.
    """
    from statistics import NormalDist
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1 or var_trials <= 0:
        return 0.0
    z = NormalDist().inv_cdf
    e = np.e
    return float(np.sqrt(var_trials)
                 * ((1 - _EULER_GAMMA) * z(1 - 1 / n_trials)
                    + _EULER_GAMMA * z(1 - 1 / (n_trials * e))))


def deflated_sharpe(returns: pd.Series, n_trials: int,
                    var_trials: float) -> float:
    """Probability that the observed Sharpe beats the expected max of N
    pure-noise trials, adjusted for the return series' skew and fat tails.

    ~0.5 means "exactly what the best of N lottery tickets looks like";
    near 1.0 means the result survives its own trial count. Report it
    BESIDE the raw Sharpe, never instead of it, and always with N.
    """
    from statistics import NormalDist
    r = returns.dropna()
    t = len(r)
    if t < 3:
        return np.nan
    sd = r.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.nan
    sr = float(r.mean() / sd)                       # per-period Sharpe
    sr0 = expected_max_sharpe(n_trials, var_trials)
    g3 = float(r.skew())
    g4 = float(r.kurt()) + 3.0                      # pandas kurt is excess
    denom = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return np.nan
    stat = (sr - sr0) * np.sqrt(t - 1) / np.sqrt(denom)
    return float(NormalDist().cdf(stat))
