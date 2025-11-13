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

def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum drawdown of an equity curve. Returns a negative number for a loss.
    """
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr_from_equity(equity: pd.Series, periods_per_year: int) -> float:
    """
    Compound annual growth rate from an equity curve.
    """
    if equity.empty:
        return np.nan
    total_return = float(equity.iloc[-1] / equity.iloc[0]) - 1.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return np.nan
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def ann_return_vol_from_equity(equity: pd.Series, periods_per_year: int) -> tuple[float, float]:
    """
    Annualized return and volatility estimated from an equity curve.
    """
    if equity.empty:
        return np.nan, np.nan
    rets = equity.pct_change().dropna()
    if rets.empty:
        return np.nan, np.nan
    mu = float(rets.mean()) * periods_per_year
    vol = float(rets.std(ddof=1)) * np.sqrt(periods_per_year)
    return mu, vol

def bootstrap_sharpe_ci(
    returns: pd.Series,
    rf_per_period: float = 0.0,
    n_boot: int = 1000,
    block_size: int = 21,
    ci: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """
    Block bootstrap confidence interval for Sharpe ratio.

    returns: per-period returns (Series)
    rf_per_period: risk-free rate per period
    n_boot: number of bootstrap samples
    block_size: length of contiguous return blocks to preserve some autocorrelation
    ci: confidence level, e.g. 0.95
    """
    r = returns.dropna().to_numpy().astype(float)
    if r.size < max(block_size * 2, 10):
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "samples": np.array([])}

    excess = r - rf_per_period
    n = excess.size

    rng = np.random.default_rng(random_state)
    sharpe_samples = []

    for _ in range(n_boot):
        idx = []
        # simple overlapping block bootstrap
        while len(idx) < n:
            start = rng.integers(0, max(1, n - block_size + 1))
            idx.extend(range(start, min(start + block_size, n)))
        idx = np.array(idx[:n], dtype=int)

        sample = excess[idx]
        std = sample.std(ddof=1)
        if std <= 0 or np.isnan(std):
            continue
        s = sample.mean() / std
        if np.isfinite(s):
            sharpe_samples.append(s)

    if not sharpe_samples:
        return {"mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "samples": np.array([])}

    samples = np.array(sharpe_samples)
    alpha = 1.0 - ci
    ci_lower, ci_upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "mean": float(np.mean(samples)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "samples": samples,
    }
