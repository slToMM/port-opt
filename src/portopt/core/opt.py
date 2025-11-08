from __future__ import annotations
import numpy as np
import cvxpy as cp

def _box_bounds(n: int, long_only: bool, lb: float, ub: float):
    lb_arr = np.full(n, max(0.0, lb)) if long_only else np.full(n, lb)
    ub_arr = np.full(n, ub)
    return lb_arr, ub_arr

def _solve(prob: cp.Problem) -> None:
    """Try a few solvers in sequence, quiet unless all fail."""
    for solver in (cp.ECOS, cp.OSQP, cp.SCS):
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate"):
                return
        except Exception:
            continue
    # last attempt with default
    prob.solve(verbose=False)

def solve_min_variance(mu: np.ndarray, cov: np.ndarray, long_only: bool = True, lb: float = 0.0, ub: float = 1.0):
    n = len(mu)
    w = cp.Variable(n)
    lb_arr, ub_arr = _box_bounds(n, long_only, lb, ub)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov)),
        [cp.sum(w) == 1, w >= lb_arr, w <= ub_arr],
    )
    _solve(prob)
    return None if w.value is None else np.asarray(w.value).ravel()

def solve_target_return(mu: np.ndarray, cov: np.ndarray, target: float,
                        long_only: bool = True, lb: float = 0.0, ub: float = 1.0):
    n = len(mu)
    w = cp.Variable(n)
    lb_arr, ub_arr = _box_bounds(n, long_only, lb, ub)
    constraints = [cp.sum(w) == 1, w >= lb_arr, w <= ub_arr, mu @ w >= target]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
    _solve(prob)
    return None if w.value is None else np.asarray(w.value).ravel()

def trace_efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_pts: int = 25,
                             long_only: bool = True, lb: float = 0.0, ub: float = 1.0):
    """
    Sweep target returns between min(mu) and max(mu). Skip infeasible targets.
    Return (weights_matrix, targets_used). Raises if no feasible point.
    """
    mu_min = float(np.min(mu))
    mu_max = float(np.max(mu))
    if abs(mu_max - mu_min) < 1e-12:
        mu_max = mu_min + 1e-6

    targets = np.linspace(mu_min, mu_max, n_pts)
    good_w = []
    good_t = []
    for t in targets:
        w = solve_target_return(mu, cov, target=t, long_only=long_only, lb=lb, ub=ub)
        if w is None or np.isnan(w).any():
            continue
        good_w.append(w)
        good_t.append(t)

    if len(good_w) == 0:
        raise ValueError("No feasible portfolios found for the chosen bounds. Try widening upper bound or lowering target range.")
    return np.vstack(good_w), np.array(good_t)

def pick_max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float, frontier_weights: np.ndarray):
    best_idx, best_sharpe = None, -1e18
    for i, w in enumerate(frontier_weights):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(0.0, w @ cov @ w)))
        if vol <= 0:
            continue
        s = (ret - rf) / vol
        if s > best_sharpe:
            best_sharpe, best_idx = s, i
    return best_idx, best_sharpe
