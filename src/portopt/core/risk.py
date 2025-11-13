from __future__ import annotations
import numpy as np
import pandas as pd


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute portfolio return series from asset returns and weights."""
    w = np.asarray(weights).reshape(-1)
    if returns.shape[1] != len(w):
        raise ValueError("Weights length and number of assets do not match")
    r = returns.mul(w, axis=1).sum(axis=1)
    r.name = "PortfolioReturn"
    return r


def var_cvar_historical(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """
    Historical VaR and CVaR for a long portfolio.
    VaR and CVaR are returned as positive numbers representing losses.
    """
    if returns.empty:
        return np.nan, np.nan

    # lower tail for a long portfolio
    q = np.quantile(returns, 1.0 - alpha)
    tail = returns[returns <= q]

    var = -float(q)
    cvar = -float(tail.mean()) if len(tail) > 0 else np.nan
    return var, cvar


def rolling_sharpe(returns: pd.Series, rf_per_period: float = 0.0, window: int = 60) -> pd.Series:
    """
    Rolling Sharpe ratio for a return series.
    rf_per_period is the risk free rate per period (not annual).
    """
    if returns.empty:
        return pd.Series(dtype=float)

    excess = returns - rf_per_period

    def _sharpe(x: pd.Series) -> float:
        if x.std(ddof=1) == 0:
            return np.nan
        return x.mean() / x.std(ddof=1)

    return excess.rolling(window=window, min_periods=window).apply(_sharpe, raw=False)


def rolling_sortino(returns: pd.Series, rf_per_period: float = 0.0, window: int = 60) -> pd.Series:
    """
    Rolling Sortino ratio using downside deviation.
    """
    if returns.empty:
        return pd.Series(dtype=float)

    excess = returns - rf_per_period

    def _sortino(x: pd.Series) -> float:
        downside = x[x < 0]
        if downside.std(ddof=1) == 0:
            return np.nan
        return x.mean() / downside.std(ddof=1)

    return excess.rolling(window=window, min_periods=window).apply(_sortino, raw=False)


def corr_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Simple correlation matrix helper."""
    return returns.corr()
