from __future__ import annotations
import numpy as np
import pandas as pd
from .stats import equity_curve, drawdown_series, annualize_mean_cov, shrink_cov_to_diag
from .opt import trace_efficient_frontier, pick_max_sharpe


def portfolio_returns_static(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Static weights applied to per period returns. No rebalancing inside the period."""
    w = np.asarray(weights).reshape(-1)
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
    out["PortfolioR"] = r_p
    out["Portfolio"] = eq_p
    out["PortfolioDD"] = drawdown_series(eq_p)

    # equal weight benchmark
    if include_equal_weight:
        ew_w = np.ones(len(returns.columns)) / len(returns.columns)
        r_ew = portfolio_returns_static(returns, ew_w)
        eq_ew = equity_curve(r_ew)
        out["EqualWeightR"] = r_ew
        out["EqualWeight"] = eq_ew
        out["EqualWeightDD"] = drawdown_series(eq_ew)

    # market proxy if available in prices
    if market_proxy and market_proxy in prices.columns:
        r_mkt = prices[market_proxy].pct_change().dropna()
        eq_mkt = equity_curve(r_mkt)
        out["MarketR"] = r_mkt.rename("MarketR")
        out["Market"] = eq_mkt.rename("Market")
        out["MarketDD"] = drawdown_series(eq_mkt).rename("MarketDD")

    return out


def backtest_walkforward(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rf: float,
    ppy: int,
    window: int = 252,
    step: int = 21,
    tc_bps: float = 0.0,
    slippage_bps: float = 0.0,
    long_only: bool = True,
    lb: float = 0.0,
    ub: float = 1.0,
    alpha: float = 0.1,
    n_pts: int = 25,
    include_equal_weight: bool = True,
    market_proxy: str | None = "SPY",
) -> dict:
    """
    Walk-forward backtest:
    - Rolling window estimation of mean/cov
    - Re-optimization every `step` periods
    - Max Sharpe portfolio on each rebalance date
    - Transaction costs + slippage based on turnover

    tc_bps and slippage_bps are in decimal form (e.g. 0.0005 = 5 bps),
    applied per unit turnover (sum |w_new - w_old|).
    The total cost is applied as a one-period return drag at each rebalance.
    """
    returns = returns.dropna()
    if returns.shape[0] <= window + step:
        raise ValueError("Not enough data for walk-forward backtest with the chosen window and step.")

    n_assets = returns.shape[1]
    idx = returns.index

    # portfolio returns over time
    r_p = pd.Series(index=idx, dtype=float, name="Portfolio")
    current_w = np.zeros(n_assets)

    total_cost_rate = tc_bps + slippage_bps  # both already decimals, e.g. 0.0005

    i = window
    weights_hist: list[np.ndarray] = []
    rebalance_dates: list[pd.Timestamp] = []

    while i < len(idx):
        # training window
        train = returns.iloc[i - window:i].dropna()
        if train.shape[0] < 20:
            break

        # annualized mean/cov
        mu, cov = annualize_mean_cov(train, ppy)
        cov_use = shrink_cov_to_diag(cov, alpha=alpha)

        # build frontier and pick max Sharpe
        W, _ = trace_efficient_frontier(
            mu, cov_use, n_pts=n_pts, long_only=long_only, lb=lb, ub=ub
        )
        best_idx, _ = pick_max_sharpe(mu, cov_use, rf, W)
        if best_idx is None:
            w_new = np.ones(n_assets) / n_assets
        else:
            w_new = W[best_idx]

        # compute turnover and total cost rate for this rebalance
        cost_rate = 0.0
        if weights_hist:
            turnover = float(np.abs(w_new - current_w).sum())
            cost_rate = total_cost_rate * turnover  # e.g. 0.001 for -0.1% hit
            # ensure we don't do crazy things if parameters are extreme
            cost_rate = max(min(cost_rate, 0.90), 0.0)

        current_w = w_new
        weights_hist.append(current_w.copy())
        rebalance_dates.append(idx[i])

        # apply this weight until next rebalance (or end)
        j_end = min(i + step, len(idx))
        for j in range(i, j_end):
            r_gross = float(returns.iloc[j].to_numpy() @ current_w)
            if j == i and cost_rate > 0.0:
                # apply cost as a drag on this first-period return after rebalance
                r_net = (1.0 - cost_rate) * (1.0 + r_gross) - 1.0
            else:
                r_net = r_gross
            r_p.iloc[j] = r_net

        i += step

    # drop leading NaNs
    r_p = r_p.dropna()
    eq_p = equity_curve(r_p)
    dd_p = drawdown_series(eq_p)

    out: dict[str, pd.Series] = {}
    out["PortfolioR"] = r_p
    out["Portfolio"] = eq_p
    out["PortfolioDD"] = dd_p

    # equal weight benchmark over full period (no walk-forward, simple static equal weight)
    if include_equal_weight:
        ew_w = np.ones(len(returns.columns)) / len(returns.columns)
        r_ew = portfolio_returns_static(returns, ew_w)
        eq_ew = equity_curve(r_ew)
        out["EqualWeightR"] = r_ew
        out["EqualWeight"] = eq_ew
        out["EqualWeightDD"] = drawdown_series(eq_ew)

    # market proxy
    if market_proxy and market_proxy in prices.columns:
        r_mkt = prices[market_proxy].pct_change().dropna()
        eq_mkt = equity_curve(r_mkt)
        out["MarketR"] = r_mkt.rename("MarketR")
        out["Market"] = eq_mkt.rename("Market")
        out["MarketDD"] = drawdown_series(eq_mkt).rename("MarketDD")

    # store weight history for potential future visualization
    if weights_hist:
        weights_df = pd.DataFrame(
            weights_hist,
            index=pd.Index(rebalance_dates, name="Date"),
            columns=returns.columns,
        )
        out["WeightsHistory"] = weights_df

    return out