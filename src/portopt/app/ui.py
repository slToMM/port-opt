import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
import io

# ensure src is importable if PYTHONPATH is not set by the shell
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from portopt.core.data import fetch_prices, align_and_clean
from portopt.core.stats import annualize_mean_cov, shrink_cov_to_diag, portfolio_metrics, equity_curve, drawdown_series
from portopt.core.opt import trace_efficient_frontier, pick_max_sharpe
from portopt.core.backtest import backtest_static

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

tabs = st.tabs(["Prices", "Returns", "Summary", "Optimization", "Backtest", "Downloads"])

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
                            w = pd.Series(st.session_state.weights, index=f["tickers"])
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
        st.info("Solve the frontier first to select a portfolio")
    else:
        prices = st.session_state.prices
        rets = st.session_state.rets
        weights = st.session_state.weights
        bt = backtest_static(prices, rets, weights, include_equal_weight=True, market_proxy="SPY")

        def _collect_series(bt: dict, keys_and_labels):
            out = []
            for key, label in keys_and_labels:
                if key in bt and isinstance(bt[key], pd.Series):
                    s = pd.to_numeric(bt[key], errors="coerce")
                    s.name = label
                    out.append(s)
            return out

        # Equity curve chart (normalize already done inside backtest)
        eq_parts = _collect_series(
            bt,
            [("Portfolio", "Portfolio"), ("EqualWeight", "EqualWeight")] + ([("Market", "Market")] if "Market" in bt else [])
        )
        eq_df = pd.concat(eq_parts, axis=1).astype(float)
        eq_long = eq_df.reset_index().melt(id_vars="Date", var_name="Series", value_name="Equity")
        fig_eq = px.line(eq_long, x="Date", y="Equity", color="Series", title="Equity Curve (normalized to 1.0)")
        st.plotly_chart(fig_eq, use_container_width=True)

        # Drawdown chart
        dd_parts = _collect_series(
            bt,
            [("PortfolioDD", "PortfolioDD"), ("EqualWeightDD", "EqualWeightDD")] + ([("MarketDD", "MarketDD")] if "MarketDD" in bt else [])
        )
        dd_df = pd.concat(dd_parts, axis=1).astype(float)
        dd_long = dd_df.reset_index().melt(id_vars="Date", var_name="Series", value_name="Drawdown")
        fig_dd = px.line(dd_long, x="Date", y="Drawdown", color="Series", title="Drawdown")
        st.plotly_chart(fig_dd, use_container_width=True)

        # Quick stats
        final_vals = eq_df.iloc[-1].rename("Final Equity")
        st.write("Final equity by strategy:")
        st.dataframe(final_vals.to_frame())

# Downloads
with tabs[5]:
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
