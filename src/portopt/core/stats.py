from __future__ import annotations
import numpy as np
import pandas as pd

def annualize_mean_cov(returns: pd.DataFrame, periods_per_year: int) -> tuple[np.ndarray, np.ndarray]:
    mu = returns.mean().to_numpy() * periods_per_year
    cov = returns.cov().to_numpy() * periods_per_year
    return mu, cov

def shrink_cov_to_diag(cov: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    d = np.diag(np.diag(cov))
    return (1.0 - alpha) * cov + alpha * d

def portfolio_metrics(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> dict:
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return {"return": ret, "vol": vol, "sharpe": sharpe}

def equity_curve(returns: pd.Series | pd.DataFrame, start_value: float = 1.0) -> pd.Series | pd.DataFrame:
    """Compound returns to an equity curve starting at start_value."""
    if isinstance(returns, pd.DataFrame):
        return start_value * (1.0 + returns).cumprod()
    else:
        return start_value * (1.0 + returns).cumprod()

def drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute drawdown series in percentage terms."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd
