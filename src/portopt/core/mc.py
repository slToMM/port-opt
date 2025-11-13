from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_gbm_portfolio(
    mu_annual: float,
    vol_annual: float,
    start_value: float = 1.0,
    years: float = 1.0,
    periods_per_year: int = 252,
    n_paths: int = 500,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Simulate a geometric Brownian motion for a single portfolio.
    Returns a DataFrame of shape (n_steps + 1, n_paths) with equity paths.
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    dt = 1.0 / periods_per_year
    n_steps = int(round(years * periods_per_year))

    # drift and diffusion terms
    mu_dt = (mu_annual - 0.5 * vol_annual ** 2) * dt
    sigma_dt = vol_annual * np.sqrt(dt)

    # normal shocks
    shocks = rng.standard_normal(size=(n_steps, n_paths))
    increments = mu_dt + sigma_dt * shocks

    # log space
    log_s = np.empty_like(increments)
    log_s[0, :] = np.log(start_value) + increments[0, :]
    for t in range(1, n_steps):
        log_s[t, :] = log_s[t - 1, :] + increments[t, :]

    paths = np.exp(log_s)
    # prepend start value
    paths = np.vstack([np.full((1, n_paths), start_value), paths])

    index = np.arange(paths.shape[0])
    cols = [f"path_{i}" for i in range(n_paths)]
    return pd.DataFrame(paths, index=index, columns=cols)


def summarize_terminal_distribution(paths: pd.DataFrame) -> dict:
    """
    Summaries of terminal equity distribution from simulated paths.
    """
    if paths.empty:
        return {"mean": np.nan, "median": np.nan, "p5": np.nan, "p95": np.nan}
    terminal = paths.iloc[-1]
    return {
        "mean": float(terminal.mean()),
        "median": float(terminal.median()),
        "p5": float(terminal.quantile(0.05)),
        "p95": float(terminal.quantile(0.95)),
    }
