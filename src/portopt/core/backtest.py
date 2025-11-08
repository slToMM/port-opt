from __future__ import annotations
import numpy as np
import pandas as pd
from .stats import equity_curve, drawdown_series

def portfolio_returns_static(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Static weights applied to per period returns. No rebalancing inside the period."""
    w = np.asarray(weights).reshape(-1)
    # ensure columns align
    cols = returns.columns
    if len(w) != len(cols):
        raise ValueError(f"Weight length {len(w)} does not match asset count {len(cols)}")
    r = returns.mul(w, axis=1).sum(axis=1)
    r.name = "Portfolio"
    return r

def backtest_static(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    weights: np.ndarray,
    include_equal_weight: bool = True,
    market_proxy: str | None = "SPY",
) -> dict:
    """
    Simple static backtest of a fixed-weight portfolio vs benchmarks.
    Returns dict with equity curves and drawdowns.
    """
    out: dict[str, pd.Series] = {}

    # portfolio
    r_p = portfolio_returns_static(returns, weights)
    eq_p = equity_curve(r_p)
    out["Portfolio"] = eq_p
    out["PortfolioDD"] = drawdown_series(eq_p)

    # equal weight benchmark
    if include_equal_weight:
        ew_w = np.ones(len(returns.columns)) / len(returns.columns)
        r_ew = portfolio_returns_static(returns, ew_w)
        eq_ew = equity_curve(r_ew)
        out["EqualWeight"] = eq_ew
        out["EqualWeightDD"] = drawdown_series(eq_ew)

    # market proxy if available in prices
    if market_proxy and market_proxy in prices.columns:
        r_mkt = prices[market_proxy].pct_change().dropna()
        eq_mkt = equity_curve(r_mkt)
        out["Market"] = eq_mkt
        out["MarketDD"] = drawdown_series(eq_mkt)

    return out
