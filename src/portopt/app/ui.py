from pathlib import Path
import sys

# ensure src is importable if PYTHONPATH is not set by the shell
ROOT = Path(__file__).resolve().parents[2]  # this is /src directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import io

from portopt.core.risk import (
    portfolio_returns,
    var_cvar_historical,
    rolling_sharpe,
    rolling_sortino,
    corr_matrix,
)
from portopt.core.mc import simulate_gbm_portfolio, summarize_terminal_distribution
from portopt.core.hrp import hrp_weights
from portopt.core.data import fetch_prices, align_and_clean
from portopt.core.stats import (
    annualize_mean_cov,
    shrink_cov_to_diag,
    portfolio_metrics,
    equity_curve,
    drawdown_series,
    ann_return_vol_from_equity,
    max_drawdown,
    bootstrap_sharpe_ci,
)
from portopt.core.opt import (
    trace_efficient_frontier,
    pick_max_sharpe,
    risk_parity_weights,
    solve_cvar_min,
    robust_return_weights,
)
from portopt.core.backtest import backtest_static, backtest_walkforward

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization App")

# small cache so repeated loads are fast
@st.cache_data(show_spinner=False)
def cached_fetch(tickers, start, end, interval):
    if not tickers:
        raise ValueError("No tickers provided.")
    raw = fetch_prices(tickers, start=start, end=end, interval=interval)
    prices = align_and_clean(raw)
    if prices.empty:
        raise ValueError("Downloaded prices are empty. Try Daily frequency, widen the date range, and ensure tickers are valid (e.g., AAPL, MSFT, TLT, GLD, IWM, SPY).")
    return prices

with st.sidebar:
    st.header("Data")
    tickers_text = st.text_input("Tickers", value="AAPL, MSFT, TLT, GLD, IWM, SPY")
    end = st.date_input("End date", value=date.today())
    start = st.date_input("Start date", value=date.today() - timedelta(days=365 * 5))
    freq = st.selectbox("Data frequency", ["Daily", "Weekly", "Monthly"])
    ret_method = st.selectbox("Return model", ["Log", "Simple"])
    load_btn = st.button("Load data", type="primary")

    st.header("Optimization")
    rf = st.number_input("Risk free rate (annual)", value=0.03, format="%.4f")
    long_only = st.checkbox("Long only", value=True)
    lb = st.number_input("Lower weight bound", value=0.0, min_value=-1.0, max_value=1.0, step=0.05)
    ub = st.number_input("Upper weight bound", value=0.6, min_value=0.0, max_value=1.0, step=0.05)
    alpha = st.slider("Covariance shrinkage alpha", 0.0, 1.0, 0.1, 0.05)
    n_pts = st.slider("Frontier points", 10, 60, 25, 1)
    solve_btn = st.button("Solve frontier")
    hrp_btn = st.button("Compute HRP weights")
    rp_btn = st.button("Compute risk-parity weights")
    cvar_btn = st.button("Minimize CVaR portfolio")
    robust_gamma = st.slider("Robustness level γ (mu uncertainty)", 0.0, 5.0, 1.0, 0.1)
    robust_btn = st.button("Compute robust portfolio")

tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

def compute_returns(prices: pd.DataFrame, method: str = "Log") -> pd.DataFrame:
    if method == "Log":
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna()

def periods_per_year(freq: str) -> int:
    return {"Daily": 252, "Weekly": 52, "Monthly": 12}[freq]

# session state
if "prices" not in st.session_state:
    st.session_state.prices = None
if "rets" not in st.session_state:
    st.session_state.rets = None
if "frontier" not in st.session_state:
    st.session_state.frontier = None
if "best_idx" not in st.session_state:
    st.session_state.best_idx = None
if "weights" not in st.session_state:
    st.session_state.weights = None

if load_btn:
    try:
        prices = cached_fetch(tickers, start, end, freq)
        rets = compute_returns(prices, method=ret_method)
        st.session_state.prices = prices
        st.session_state.rets = rets
        st.success(f"Loaded {len(tickers)} tickers with {prices.shape[0]} rows")
        st.write(f"Prices shape: {prices.shape}")
        st.write(f"Columns: {list(prices.columns)}"[:200])

    except Exception as e:
        st.error(f"Failed to load data: {e}")

tabs = st.tabs(["Prices", "Returns", "Summary", "Optimization", "Backtest", "Risk analysis", "Downloads"])

# Prices
with tabs[0]:
    if st.session_state.prices is None:
        st.info("Load data to see prices")
    else:
        prices = st.session_state.prices
        st.dataframe(prices.tail())
        wide = prices.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Adj Close")
        fig = px.line(wide, x="Date", y="Adj Close", color="Ticker", title="Adjusted Close")
        st.plotly_chart(fig, use_container_width=True)

# Returns
with tabs[1]:
    if st.session_state.rets is None:
        st.info("Load data to see returns")
    else:
        rets = st.session_state.rets
        st.dataframe(rets.tail())
        ret_wide = rets.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Return")
        fig2 = px.line(ret_wide, x="Date", y="Return", color="Ticker", title="Periodic Returns")
        st.plotly_chart(fig2, use_container_width=True)

# Summary
with tabs[2]:
    if st.session_state.rets is None:
        st.info("Load data to see summary")
    else:
        rets = st.session_state.rets
        ppy = periods_per_year(freq)
        mu, cov = annualize_mean_cov(rets, ppy)
        vol = np.sqrt(np.diag(cov))
        df = pd.DataFrame({
            "Annual Return": mu,
            "Annual Volatility": vol,
        }, index=rets.columns)
        st.dataframe(df.style.format("{:.3f}"))

# Optimization
with tabs[3]:
    if st.session_state.rets is None:
        st.info("Load data first")
    else:
        rets = st.session_state.rets

        # Validate before solving
        if rets.empty:
            st.error("No overlapping data for the chosen tickers and dates.")
        elif rets.shape[1] < 2:
            st.error("Need at least 2 assets to trace an efficient frontier.")
        else:
            ppy = periods_per_year(freq)
            mu, cov = annualize_mean_cov(rets, ppy)

            if mu.size == 0 or np.isnan(mu).any() or np.isnan(cov).any():
                st.error("Mean or covariance is invalid. Try Daily frequency and a wider date range.")
            else:
                cov_use = shrink_cov_to_diag(cov, alpha=alpha)

                # 1) If user clicked Solve, compute and store results
                if solve_btn:
                    with st.spinner("Solving frontier..."):
                        try:
                            W, targets = trace_efficient_frontier(
                                mu, cov_use, n_pts=n_pts, long_only=long_only, lb=lb, ub=ub
                            )
                            risks = [float(np.sqrt(w @ cov_use @ w)) for w in W]
                            rets_ann = [float(w @ mu) for w in W]
                            best_idx, best_sharpe = pick_max_sharpe(mu, cov_use, rf, W)

                            st.session_state.frontier = {
                                "W": W, "risks": risks, "rets": rets_ann,
                                "mu": mu, "cov": cov_use, "best_idx": best_idx,
                                "tickers": list(rets.columns), "rf": rf, "best_sharpe": best_sharpe,
                            }
                            st.session_state.weights = W[best_idx] if best_idx is not None else None
                            st.success("Frontier solved. See the chart below.")
                        except Exception as e:
                            st.error(f"Optimization failed: {e}")

                if hrp_btn:
                    with st.spinner("Computing HRP allocation..."):
                        try:
                            weights_hrp = hrp_weights(rets)
                            st.session_state.frontier = None    # clear old frontier display
                            st.session_state.weights = weights_hrp.values
                            st.session_state.weights_labels = list(weights_hrp.index)

                            st.success("HRP weights computed.")

                            # Display weights
                            st.subheader("HRP Portfolio Weights")
                            st.dataframe(weights_hrp.to_frame("Weight").style.format("{:.3f}"))
                        except Exception as e:
                            st.error(f"HRP failed: {e}")

                if rp_btn:
                    with st.spinner("Computing risk parity allocation..."):
                        try:
                            # reuse cov_use (already shrunk for stability)
                            w_rp = risk_parity_weights(
                                cov_use,
                                lb=lb,
                                ub=ub,
                            )

                            # clear any old Markowitz frontier display
                            st.session_state.frontier = None
                            st.session_state.weights = w_rp
                            st.session_state.weights_labels = list(rets.columns)

                            st.success("Risk parity weights computed.")

                            # display weights table
                            st.subheader("Risk parity portfolio weights")
                            w_series = pd.Series(w_rp, index=rets.columns)
                            st.dataframe(
                                w_series.to_frame("Weight").style.format("{:.3f}")
                            )
                        except Exception as e:
                            st.error(f"Risk parity optimization failed: {e}")                       

                if cvar_btn:
                    with st.spinner("Computing CVaR-minimizing portfolio..."):
                        try:
                            # use historical returns matrix for CVaR optimization
                            R = rets.to_numpy()
                            w_cvar = solve_cvar_min(
                                returns=R,
                                alpha=0.95,          # could expose as a control later
                                long_only=long_only,
                                lb=lb,
                                ub=ub,
                            )

                            # store in session
                            st.session_state.frontier = None
                            st.session_state.weights = w_cvar
                            st.session_state.weights_labels = list(rets.columns)

                            st.success("CVaR-minimizing weights computed.")

                            # display weights
                            st.subheader("CVaR-minimizing portfolio weights")
                            w_series = pd.Series(w_cvar, index=rets.columns)
                            st.dataframe(w_series.to_frame("Weight").style.format("{:.3f}"))

                            # compute and display resulting CVaR on historical returns
                            port_rets_cvar = (rets * w_cvar).sum(axis=1)
                            var_val, cvar_val = var_cvar_historical(port_rets_cvar, alpha=0.95)
                            st.write(
                                f"Historical VaR (alpha=0.95): {var_val:.4f}  "
                                f"CVaR: {cvar_val:.4f}"
                            )
                        except Exception as e:
                            st.error(f"CVaR optimization failed: {e}")

                if robust_btn:
                    with st.spinner("Computing robust portfolio (ellipsoidal μ uncertainty)..."):
                        try:
                            # use annualized mu and shrunken cov for robustness
                            w_rob = robust_return_weights(
                                mu=mu,
                                cov=cov_use,
                                gamma=float(robust_gamma),
                                long_only=long_only,
                                lb=lb,
                                ub=ub,
                            )

                            st.session_state.frontier = None
                            st.session_state.weights = w_rob
                            st.session_state.weights_labels = list(rets.columns)

                            st.success("Robust portfolio weights computed.")

                            # display weights
                            st.subheader("Robust portfolio weights (ellipsoidal μ uncertainty)")
                            w_series = pd.Series(w_rob, index=rets.columns)
                            st.dataframe(w_series.to_frame("Weight").style.format("{:.3f}"))

                            # show nominal vs penalized expected return for this w
                            port_nominal_ret = float(w_rob @ mu)
                            # compute ||Sigma^{1/2} w||
                            cov_mat = cov_use
                            eigvals, eigvecs = np.linalg.eigh(cov_mat)
                            eigvals_clipped = np.clip(eigvals, 0.0, None)
                            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T
                            unc = float(np.linalg.norm(sqrt_cov @ w_rob, 2))
                            worst_case_ret = port_nominal_ret - float(robust_gamma) * unc

                            st.write(
                                f"Nominal expected return: {port_nominal_ret:.3f}  "
                                f"| Uncertainty term: {unc:.3f}  "
                                f"| Worst-case expected return (γ·||Σ^½w|| penalty): {worst_case_ret:.3f}"
                            )
                        except Exception as e:
                            st.error(f"Robust optimization failed: {e}")


                # 2) Always render any saved result, even after rerun
                f = st.session_state.frontier
                if f is not None:
                    colA, colB = st.columns([2, 1])

                    with colA:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=f["risks"], y=f["rets"],
                                                 mode="lines+markers", name="Frontier"))
                        if f["best_idx"] is not None:
                            i = f["best_idx"]
                            fig.add_trace(go.Scatter(
                                x=[f["risks"][i]], y=[f["rets"][i]],
                                mode="markers", marker=dict(size=12, symbol="star"),
                                name=f"Max Sharpe ~ {f['best_sharpe']:.2f}"
                            ))
                        fig.update_layout(title="Efficient Frontier",
                                          xaxis_title="Volatility", yaxis_title="Return")
                        st.plotly_chart(fig, use_container_width=True)

                        if st.session_state.weights is not None:
                            tickers_display = st.session_state.frontier["tickers"] if f is not None else st.session_state.weights_labels
                            w = pd.Series(st.session_state.weights, index=tickers_display)

                            st.subheader("Max Sharpe Weights")
                            st.dataframe(w.to_frame("Weight").style.format("{:.3f}"))

                            mets = portfolio_metrics(st.session_state.weights, f["mu"], f["cov"], rf=f["rf"])
                            st.write(f"Return: {mets['return']:.3f}   Vol: {mets['vol']:.3f}   Sharpe: {mets['sharpe']:.2f}")

                    with colB:
                        st.caption("Tips")
                        st.caption("• Use Daily data and a 5 year window for better overlap")
                        st.caption("• Raise shrinkage if the frontier looks unstable")
                        st.caption("• Widen upper bounds to reduce corner solutions")
                else:
                    st.info("Set options in the sidebar and click Solve frontier")

# Backtest
with tabs[4]:
    if st.session_state.rets is None or st.session_state.weights is None:
        st.info("Solve the frontier or compute HRP weights first to select a portfolio.")
    else:
        prices = st.session_state.prices
        rets = st.session_state.rets
        weights = st.session_state.weights

        st.subheader("Backtest")

        mode = st.radio("Backtest mode", ["Static weights", "Walk forward"], horizontal=True)

        ppy = periods_per_year(freq)

        if mode == "Static weights":
            bt = backtest_static(prices, rets, weights, include_equal_weight=True, market_proxy="SPY")
        else:
            st.markdown("Configure walk-forward settings:")
            col_w1, col_w2, col_w3, col_w4 = st.columns(4)
            with col_w1:
                window = st.slider("Training window (periods)", 60, 756, 252, 21)
            with col_w2:
                step = st.slider("Rebalance every N periods", 5, 63, 21, 1)
            with col_w3:
                tc_bps = st.slider("Transaction cost (bps)", 0.0, 50.0, 5.0, 0.5)
            with col_w4:
                slip_bps = st.slider("Slippage (bps)", 0.0, 50.0, 2.0, 0.5,
                                    help="Extra execution cost per 100% turnover at each rebalance.")

            try:
                bt = backtest_walkforward(
                    prices=prices,
                    returns=rets,
                    rf=rf,
                    ppy=ppy,
                    window=window,
                    step=step,
                    tc_bps=tc_bps / 10000.0,
                    slippage_bps=slip_bps / 10000.0,
                    long_only=long_only,
                    lb=lb,
                    ub=ub,
                    alpha=alpha,
                    n_pts=n_pts,
                    include_equal_weight=True,
                    market_proxy="SPY",
                )
                st.success("Walk-forward backtest completed.")
            except Exception as e:
                st.error(f"Walk-forward backtest failed: {e}")
                bt = None

        if bt is not None:
            # === Additional benchmarks: 60/40 and All Weather ===

            # 60/40: 60% SPY, 40% TLT (if both available)
            try:
                if {"SPY", "TLT"}.issubset(prices.columns):
                    r_spy = prices["SPY"].pct_change().dropna()
                    r_tlt = prices["TLT"].pct_change().dropna()
                    idx_60 = r_spy.index.intersection(r_tlt.index)
                    if len(idx_60) > 1:
                        r_60_40 = 0.6 * r_spy.loc[idx_60] + 0.4 * r_tlt.loc[idx_60]
                        eq_60_40 = equity_curve(r_60_40)
                        dd_60_40 = drawdown_series(eq_60_40)
                        bt["B60_40"] = eq_60_40
                        bt["B60_40DD"] = dd_60_40
            except Exception:
                pass

            # All Weather (simplified): SPY, TLT, IEF, GLD, DBC
            # Target weights: 30% stocks, 40% LT bonds, 15% int bonds, 7.5% gold, 7.5% commodities
            try:
                aw_spec = [
                    ("SPY", 0.30),
                    ("TLT", 0.40),
                    ("IEF", 0.15),
                    ("GLD", 0.075),
                    ("DBC", 0.075),
                ]
                available = [(t, w) for (t, w) in aw_spec if t in prices.columns]
                if len(available) >= 2:
                    # normalize weights over available assets
                    total_w = sum(w for _, w in available)
                    available = [(t, w / total_w) for (t, w) in available]

                    # build common index
                    ret_series = []
                    for t, w in available:
                        r_t = prices[t].pct_change().dropna()
                        ret_series.append(r_t)
                    idx_aw = ret_series[0].index
                    for s in ret_series[1:]:
                        idx_aw = idx_aw.intersection(s.index)

                    if len(idx_aw) > 1:
                        r_aw = 0.0
                        for t, w in available:
                            r_t = prices[t].pct_change().dropna().loc[idx_aw]
                            r_aw = r_aw + w * r_t
                        eq_aw = equity_curve(r_aw)
                        dd_aw = drawdown_series(eq_aw)
                        bt["AllWeather"] = eq_aw
                        bt["AllWeatherDD"] = dd_aw
            except Exception:
                pass

            # === In-sample / Out-of-sample split control ===
            split_idx = None
            ...

            # === In-sample / Out-of-sample split control ===
            split_idx = None
            if "Portfolio" in bt and isinstance(bt["Portfolio"], pd.Series) and len(bt["Portfolio"]) > 1:
                st.markdown("### In-sample / Out-of-sample split")

                split_frac = st.slider(
                    "OOS starts at fraction of backtest length",
                    0.2,
                    0.9,
                    0.7,
                    0.05,
                    help="Define the point where you treat performance as out-of-sample.",
                )
                eq_index = bt["Portfolio"].index
                pos = int(len(eq_index) * split_frac)
                pos = min(max(pos, 1), len(eq_index) - 1)
                split_idx = eq_index[pos]
                st.caption(
                    f"OOS starts at **{split_idx.date()}** "
                    f"(≈ {int(split_frac * 100)}% into the backtest)."
                )
            else:
                st.info("Not enough portfolio history to define an IS/OOS split.")

            # === Equity curve chart ===
            def _collect_series(bt_dict: dict, keys_and_labels):

                out = []
                for key, label in keys_and_labels:
                    if key in bt_dict and isinstance(bt_dict[key], pd.Series):
                        s = pd.to_numeric(bt_dict[key], errors="coerce")
                        s.name = label
                        out.append(s)
                return out

            eq_parts = _collect_series(
                bt,
                [("Portfolio", "Portfolio"), ("EqualWeight", "EqualWeight")]
                + ([("Market", "Market")] if "Market" in bt else [])
                + ([("B60_40", "60/40")] if "B60_40" in bt else [])
                + ([("AllWeather", "AllWeather")] if "AllWeather" in bt else [])
            )
            if eq_parts:
                eq_df = pd.concat(eq_parts, axis=1).astype(float)
                eq_long = eq_df.reset_index().melt(
                    id_vars="Date", var_name="Series", value_name="Equity"
                )
                fig_eq = px.line(
                    eq_long,
                    x="Date",
                    y="Equity",
                    color="Series",
                    title="Equity Curve (normalized to 1.0)",
                )

                # vertical line at OOS start
                if split_idx is not None:
                    fig_eq.add_vline(
                        x=split_idx,
                        line_dash="dash",
                        line_width=1,
                        line_color="black",
                    )

                st.plotly_chart(fig_eq, use_container_width=True)

            # === Drawdown chart ===
            dd_parts = _collect_series(
                bt,
                [("PortfolioDD", "PortfolioDD"), ("EqualWeightDD", "EqualWeightDD")]
                + ([("MarketDD", "MarketDD")] if "MarketDD" in bt else [])
                + ([("B60_40DD", "60/40DD")] if "B60_40DD" in bt else [])
                + ([("AllWeatherDD", "AllWeatherDD")] if "AllWeatherDD" in bt else [])
            )
            if dd_parts:
                dd_df = pd.concat(dd_parts, axis=1).astype(float)
                dd_long = dd_df.reset_index().melt(
                    id_vars="Date", var_name="Series", value_name="Drawdown"
                )
                fig_dd = px.line(
                    dd_long,
                    x="Date",
                    y="Drawdown",
                    color="Series",
                    title="Drawdown",
                )

                if split_idx is not None:
                    fig_dd.add_vline(
                        x=split_idx,
                        line_dash="dash",
                        line_width=1,
                        line_color="black",
                    )

                st.plotly_chart(fig_dd, use_container_width=True)

            # === Backtest metrics ===
            from portopt.core.stats import ann_return_vol_from_equity, max_drawdown
            st.subheader("Backtest metrics (full period)")

            rows = []

            def row_from(eq_key, name):
                if eq_key in bt:
                    eq = bt[eq_key]
                    mu_ann, vol_ann = ann_return_vol_from_equity(eq, ppy)
                    dd = max_drawdown(eq)
                    sharpe = (mu_ann - rf) / vol_ann if vol_ann and vol_ann > 0 else np.nan
                    rows.append(
                        {
                            "Strategy": name,
                            "Ann Return": mu_ann,
                            "Ann Vol": vol_ann,
                            "Sharpe": sharpe,
                            "Max DD": dd,
                        }
                    )

            row_from("Portfolio", "Portfolio")
            if "EqualWeight" in bt:
                row_from("EqualWeight", "EqualWeight")
            if "Market" in bt:
                row_from("Market", "Market")
            if "B60_40" in bt:
                row_from("B60_40", "60/40")
            if "AllWeather" in bt:
                row_from("AllWeather", "AllWeather")

            if rows:
                metrics_df = pd.DataFrame(rows)
                st.dataframe(
                    metrics_df.style.format(
                        {
                            "Ann Return": "{:.3f}",
                            "Ann Vol": "{:.3f}",
                            "Sharpe": "{:.2f}",
                            "Max DD": "{:.2%}",
                        }
                    )
                )

            # === Backtest metrics by segment (IS vs OOS) ===
            if split_idx is not None:
                st.subheader("Backtest metrics by segment (IS / OOS)")

                rows_seg = []

                def row_seg(eq_key, name):
                    if eq_key in bt:
                        eq_full = bt[eq_key]
                        eq_is = eq_full[eq_full.index <= split_idx]
                        eq_oos = eq_full[eq_full.index > split_idx]

                        for seg_label, eq_seg in [("IS", eq_is), ("OOS", eq_oos)]:
                            if eq_seg is None or len(eq_seg) < 2:
                                continue
                            mu_ann, vol_ann = ann_return_vol_from_equity(eq_seg, ppy)
                            dd = max_drawdown(eq_seg)
                            sharpe = (mu_ann - rf) / vol_ann if vol_ann and vol_ann > 0 else np.nan
                            rows_seg.append(
                                {
                                    "Strategy": name,
                                    "Segment": seg_label,
                                    "Ann Return": mu_ann,
                                    "Ann Vol": vol_ann,
                                    "Sharpe": sharpe,
                                    "Max DD": dd,
                                }
                            )

                row_seg("Portfolio", "Portfolio")
                if "EqualWeight" in bt:
                    row_seg("EqualWeight", "EqualWeight")
                if "Market" in bt:
                    row_seg("Market", "Market")
                if "B60_40" in bt:
                    row_seg("B60_40", "60/40")
                if "AllWeather" in bt:
                    row_seg("AllWeather", "AllWeather")

                if rows_seg:
                    metrics_seg_df = pd.DataFrame(rows_seg)
                    st.dataframe(
                        metrics_seg_df.style.format(
                            {
                                "Ann Return": "{:.3f}",
                                "Ann Vol": "{:.3f}",
                                "Sharpe": "{:.2f}",
                                "Max DD": "{:.2%}",
                            }
                        )
                    )

            # === Sharpe ratio significance (bootstrap) ===
            st.subheader("Sharpe ratio significance (bootstrap)")

            # build list of available strategies
            strat_options: list[tuple[str, str]] = []
            if ("Portfolio" in bt) or ("PortfolioR" in bt):
                strat_options.append(("Portfolio", "Portfolio"))
            if ("EqualWeight" in bt) or ("EqualWeightR" in bt):
                strat_options.append(("EqualWeight", "EqualWeight"))
            if ("Market" in bt) or ("MarketR" in bt):
                strat_options.append(("Market", "Market"))
            if "B60_40" in bt:
                strat_options.append(("60/40", "B60_40"))
            if "AllWeather" in bt:
                strat_options.append(("AllWeather", "AllWeather"))

            if strat_options:
                labels = [lab for (lab, _) in strat_options]
                chosen_label = st.selectbox("Strategy for bootstrap Sharpe analysis", labels)

                # find key_prefix for chosen label
                key_prefix = None
                for lab, key in strat_options:
                    if lab == chosen_label:
                        key_prefix = key
                        break

                # get return series for that strategy
                ret_series = None
                if key_prefix is not None:
                    r_key = key_prefix + "R"
                    if r_key in bt:
                        ret_series = bt[r_key]
                    elif key_prefix in bt:
                        # derive returns from equity curve
                        ret_series = bt[key_prefix].pct_change().dropna()

                if ret_series is None or len(ret_series) < 30:
                    st.info("Not enough data to run bootstrap for the selected strategy.")
                else:
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        n_boot = st.slider("Bootstrap samples", 500, 5000, 2000, 500)
                    with col_b2:
                        block_size = st.slider("Block size (periods)", 5, 63, 21, 2)

                    rf_per_period = rf / ppy
                    res_bs = bootstrap_sharpe_ci(
                        ret_series,
                        rf_per_period=rf_per_period,
                        n_boot=n_boot,
                        block_size=block_size,
                        ci=0.95,
                        random_state=42,
                    )

                    if res_bs["samples"] is None or len(res_bs["samples"]) == 0:
                        st.info("Bootstrap failed to produce valid Sharpe samples.")
                    else:
                        st.write(
                            f"Estimated Sharpe (per-period, annualization-invariant): "
                            f"mean = {res_bs['mean']:.2f}, "
                            f"95% CI = [{res_bs['ci_lower']:.2f}, {res_bs['ci_upper']:.2f}]"
                        )

                        samples_df = pd.DataFrame({"Sharpe": res_bs["samples"]})
                        fig_boot = px.histogram(
                            samples_df,
                            x="Sharpe",
                            nbins=40,
                            title=f"Bootstrap Sharpe distribution — {chosen_label}",
                        )
                        fig_boot.add_vline(x=res_bs["mean"], line_dash="dash", line_width=2)
                        fig_boot.add_vline(x=res_bs["ci_lower"], line_dash="dot", line_width=1)
                        fig_boot.add_vline(x=res_bs["ci_upper"], line_dash="dot", line_width=1)
                        st.plotly_chart(fig_boot, use_container_width=True)
            else:
                st.info("No strategies available for Sharpe bootstrap analysis.")

            # === Final equity snapshot ===
            if "Portfolio" in bt:
                series_list = [
                    bt.get("Portfolio"),
                    bt.get("EqualWeight"),
                    bt.get("Market"),
                    bt.get("B60_40"),
                    bt.get("AllWeather"),
                ]
                series_list = [s for s in series_list if s is not None]
                if series_list:
                    eq_df_final = pd.concat(series_list, axis=1)
                    eq_df_final.columns = [
                        name
                        for name, s in zip(
                            ["Portfolio", "EqualWeight", "Market", "60/40", "AllWeather"],
                            [bt.get("Portfolio"), bt.get("EqualWeight"), bt.get("Market"), bt.get("B60_40"), bt.get("AllWeather")],
                        )
                        if s is not None
                    ]
                    final_vals = eq_df_final.iloc[-1].rename("Final Equity")
                    st.subheader("Final equity by strategy")
                    st.dataframe(final_vals.to_frame())


# Risk analysis
with tabs[5]:
    if st.session_state.rets is None or st.session_state.weights is None:
        st.info("Load data and solve the frontier to enable risk analysis.")
    else:
        prices = st.session_state.prices
        rets = st.session_state.rets
        weights = st.session_state.weights

        st.subheader("Risk analysis for current portfolio")

        col_ctrl, col_empty = st.columns([2, 1])
        with col_ctrl:
            alpha = st.slider("Confidence level for VaR / CVaR", 0.90, 0.99, 0.95, 0.01)
            horizon_years = st.slider("Monte Carlo horizon (years)", 0.25, 5.0, 1.0, 0.25)
            n_paths = st.slider("Monte Carlo paths", 100, 2000, 500, 100)
            window = st.slider("Rolling window (periods)", 30, 252, 126, 10)
        st.markdown("---")

        # Portfolio historical returns
        port_rets = portfolio_returns(rets, weights)

        # 1) VaR and CVaR
        var_val, cvar_val = var_cvar_historical(port_rets, alpha=alpha)
        st.markdown("### Historical VaR and CVaR")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("VaR", f"{var_val:.4f}")
        with col2:
            st.metric("CVaR", f"{cvar_val:.4f}")
        with col3:
            st.metric("Mean return (per period)", f"{port_rets.mean():.4f}")

        # 2) Monte Carlo simulation
        st.markdown("### Monte Carlo simulation")

        # annualized mean and vol from historical portfolio series
        ppy = periods_per_year(freq)
        mu_annual = float(port_rets.mean() * ppy)
        vol_annual = float(port_rets.std(ddof=1) * np.sqrt(ppy))

        if np.isnan(mu_annual) or np.isnan(vol_annual) or vol_annual <= 0:
            st.warning("Not enough data to estimate annualized mean and volatility for Monte Carlo.")
        else:
            paths = simulate_gbm_portfolio(
                mu_annual=mu_annual,
                vol_annual=vol_annual,
                start_value=1.0,
                years=float(horizon_years),
                periods_per_year=ppy,
                n_paths=int(n_paths),
                random_state=42,
            )
            summary = summarize_terminal_distribution(paths)

            # fan chart
            quantiles = paths.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T
            quantiles.columns = ["p5", "p25", "p50", "p75", "p95"]
            quantiles = quantiles.reset_index().rename(columns={"index": "step"})

            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(
                x=quantiles["step"], y=quantiles["p50"],
                mode="lines", name="Median"
            ))
            fig_mc.add_trace(go.Scatter(
                x=quantiles["step"], y=quantiles["p75"],
                mode="lines", name="75%", line=dict(width=0.5)
            ))
            fig_mc.add_trace(go.Scatter(
                x=quantiles["step"], y=quantiles["p25"],
                mode="lines", name="25%", line=dict(width=0.5),
                fill="tonexty", fillcolor="rgba(0,0,0,0.1)"
            ))
            fig_mc.add_trace(go.Scatter(
                x=quantiles["step"], y=quantiles["p95"],
                mode="lines", name="95%", line=dict(width=0.5)
            ))
            fig_mc.add_trace(go.Scatter(
                x=quantiles["step"], y=quantiles["p5"],
                mode="lines", name="5%", line=dict(width=0.5),
                fill="tonexty", fillcolor="rgba(0,0,0,0.05)"
            ))
            fig_mc.update_layout(
                title="Monte Carlo simulated portfolio equity",
                xaxis_title="Step",
                yaxis_title="Equity"
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Terminal mean", f"{summary['mean']:.3f}")
            col_s2.metric("Terminal median", f"{summary['median']:.3f}")
            col_s3.metric("Terminal 5%", f"{summary['p5']:.3f}")
            col_s4.metric("Terminal 95%", f"{summary['p95']:.3f}")

        st.markdown("---")

        # 3) Correlation heatmap
        st.markdown("### Correlation heatmap")
        corr = corr_matrix(rets)
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                             title="Asset return correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

        # 4) Rolling Sharpe and Sortino
        st.markdown("### Rolling Sharpe and Sortino")

        rf_per_period = 0.0  # can be extended later
        roll_sharpe = rolling_sharpe(port_rets, rf_per_period=rf_per_period, window=window)
        roll_sortino = rolling_sortino(port_rets, rf_per_period=rf_per_period, window=window)

        roll_df = pd.DataFrame({
            "Sharpe": roll_sharpe,
            "Sortino": roll_sortino,
        }).dropna()

        if roll_df.empty:
            st.info("Not enough data for rolling metrics with the current window.")
        else:
            roll_df = roll_df.reset_index().rename(columns={"index": "Date"})
            fig_roll = px.line(roll_df, x="Date", y=["Sharpe", "Sortino"], title="Rolling Sharpe and Sortino")
            st.plotly_chart(fig_roll, use_container_width=True)

                # ----- Historical stress periods -----
        st.markdown("---")
        st.markdown("### Historical stress periods")

        # helper to compute cumulative return and max drawdown for a series
        def cumret_and_dd(ret_series: pd.Series) -> tuple[float, float]:
            r = ret_series.dropna()
            if r.empty:
                return np.nan, np.nan
            eq = equity_curve(r)
            cum_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
            dd = max_drawdown(eq)
            return cum_ret, dd

        # define stress windows (approximate, can be refined)
        stress_periods = [
            {
                "name": "Global Financial Crisis",
                "start": pd.Timestamp("2007-10-01"),
                "end": pd.Timestamp("2009-03-31"),
            },
            {
                "name": "COVID Crash",
                "start": pd.Timestamp("2020-02-01"),
                "end": pd.Timestamp("2020-04-30"),
            },
            {
                "name": "Rate Hike / Inflation Shock",
                "start": pd.Timestamp("2022-01-01"),
                "end": pd.Timestamp("2022-12-31"),
            },
        ]

        rows = []
        for sp in stress_periods:
            name = sp["name"]
            start_sp = sp["start"]
            end_sp = sp["end"]

            # align to available data range
            pr = port_rets.loc[start_sp:end_sp]
            if pr.empty:
                rows.append(
                    {
                        "Period": name,
                        "Strategy": "Portfolio",
                        "Cum Return": np.nan,
                        "Max DD": np.nan,
                    }
                )
                continue

            # portfolio
            cum_p, dd_p = cumret_and_dd(pr)
            rows.append(
                {
                    "Period": name,
                    "Strategy": "Portfolio",
                    "Cum Return": cum_p,
                    "Max DD": dd_p,
                }
            )

            # equal weight
            ew_w = np.ones(rets.shape[1]) / rets.shape[1]
            ew_rets = rets.mul(ew_w, axis=1).sum(axis=1).loc[start_sp:end_sp]
            cum_ew, dd_ew = cumret_and_dd(ew_rets)
            rows.append(
                {
                    "Period": name,
                    "Strategy": "EqualWeight",
                    "Cum Return": cum_ew,
                    "Max DD": dd_ew,
                }
            )

            # market proxy (SPY) if available
            if "SPY" in prices.columns:
                mkt_rets = prices["SPY"].pct_change().loc[start_sp:end_sp]
                cum_m, dd_m = cumret_and_dd(mkt_rets)
                rows.append(
                    {
                        "Period": name,
                        "Strategy": "Market(SPY)",
                        "Cum Return": cum_m,
                        "Max DD": dd_m,
                    }
                )

        if rows:
            stress_df = pd.DataFrame(rows)
            # nice formatting
            st.write("Performance during key crisis periods:")
            st.dataframe(
                stress_df.style.format(
                    {
                        "Cum Return": "{:.2%}",
                        "Max DD": "{:.2%}",
                    }
                )
            )

            # optional: bar chart of cumulative returns
            stress_plot = stress_df.dropna(subset=["Cum Return"]).copy()
            fig_stress = px.bar(
                stress_plot,
                x="Period",
                y="Cum Return",
                color="Strategy",
                barmode="group",
                title="Cumulative return during stress periods",
            )
            st.plotly_chart(fig_stress, use_container_width=True)
        else:
            st.info("Not enough data to evaluate historical stress periods for this portfolio.")

# Downloads
with tabs[6]:
    if st.session_state.weights is None:
        st.info("Solve the frontier to enable downloads")
    else:
        weights = pd.Series(st.session_state.weights, index=st.session_state.rets.columns, name="Weight")
        csv_bytes = weights.to_csv().encode("utf-8")
        st.download_button("Download weights CSV", data=csv_bytes, file_name="weights.csv", mime="text/csv")

        # also allow downloading the frontier points
        if st.session_state.frontier:
            f = st.session_state.frontier
            frontier_df = pd.DataFrame({"Volatility": f["risks"], "Return": f["rets"]})
            buf = io.StringIO()
            frontier_df.to_csv(buf, index=False)
            st.download_button("Download frontier CSV", data=buf.getvalue().encode("utf-8"),
                               file_name="frontier.csv", mime="text/csv")

        if "weights_labels" in st.session_state:
            weights_df = pd.DataFrame({
                "Ticker": st.session_state.weights_labels,
                "Weight": st.session_state.weights
            })
            st.download_button("Download current weights CSV", 
                            data=weights_df.to_csv(index=False).encode("utf-8"),
                            file_name="weights_hrp_or_mvo.csv",
                            mime="text/csv")
