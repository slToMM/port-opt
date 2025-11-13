from __future__ import annotations
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

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

def risk_parity_weights(
    cov: np.ndarray,
    lb: float = 0.0,
    ub: float = 1.0,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute long-only Equal Risk Contribution (risk-parity) weights.

    Parameters
    ----------
    cov : (n, n) covariance matrix
    lb  : lower bound per weight (>= 0 for long-only)
    ub  : upper bound per weight
    initial : optional initial guess for weights

    Returns
    -------
    w : (n,) numpy array of portfolio weights
    """
    cov = np.asarray(cov)
    n = cov.shape[0]
    if cov.shape[1] != n:
        raise ValueError("cov must be square")

    if initial is None:
        x0 = np.full(n, 1.0 / n)
    else:
        x0 = np.asarray(initial, dtype=float)
        if x0.shape[0] != n:
            raise ValueError("initial length must match number of assets")

    # objective: sum_i (RC_i - target)^2 where RC_i = w_i * (Cw)_i
    def objective(w):
        w = np.asarray(w)
        # small penalty if negative (should be prevented by bounds anyway)
        if np.any(w < -1e-8):
            return 1e6
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            return 1e6
        mrc = cov @ w              # marginal risk contributions
        rc = w * mrc               # absolute risk contributions
        target = port_var / n      # desired equal contribution
        return float(((rc - target) ** 2).sum())

    # constraint: sum of weights = 1
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )

    # bounds: lb <= w_i <= ub (but no shorts if lb < 0 is not desired)
    lo = max(0.0, lb)   # enforce long-only floor
    bounds = [(lo, ub) for _ in range(n)]

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000, "ftol": 1e-9},
    )

    if not res.success:
        raise RuntimeError(f"Risk parity optimization failed: {res.message}")

    w = np.asarray(res.x, dtype=float)
    # normalize again just in case of tiny numerical drift
    w = np.maximum(w, 0.0)
    w /= w.sum()
    return w

def solve_cvar_min(
    returns: np.ndarray,
    alpha: float = 0.95,
    long_only: bool = True,
    lb: float = 0.0,
    ub: float = 1.0,
) -> np.ndarray:
    """
    Minimize historical CVaR at level alpha using linear programming.

    Parameters
    ----------
    returns : array (T, n)
        Scenario returns (per period) for n assets over T periods.
    alpha : float
        Confidence level (e.g. 0.95).
    long_only : bool
        If True, weights are constrained to be nonnegative.
    lb, ub : float
        Box bounds for weights.

    Returns
    -------
    w : array (n,)
        Portfolio weights that minimize CVaR.
    """
    R = np.asarray(returns)
    if R.ndim != 2:
        raise ValueError("returns must be 2D (T, n).")
    T, n = R.shape
    if T == 0 or n == 0:
        raise ValueError("returns array is empty.")

    # variables
    w = cp.Variable(n)
    zeta = cp.Variable()         # VaR threshold
    u = cp.Variable(T)           # slack variables

    # bounds
    if long_only:
        lb_eff = max(0.0, lb)
    else:
        lb_eff = lb
    lb_arr = np.full(n, lb_eff)
    ub_arr = np.full(n, ub)

    # portfolio returns per scenario: R w
    scen_rets = R @ w

    constraints = []
    # sum of weights = 1
    constraints.append(cp.sum(w) == 1.0)
    # box constraints
    constraints.append(w >= lb_arr)
    constraints.append(w <= ub_arr)
    # CVaR constraints: u_t >= -r_t^T w - zeta, u_t >= 0
    constraints.append(u >= -scen_rets - zeta)
    constraints.append(u >= 0)

    cvar_term = (1.0 / ((1.0 - alpha) * T)) * cp.sum(u)
    objective = cp.Minimize(zeta + cvar_term)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"CVaR optimization failed with status {prob.status}")

    w_val = w.value
    if w_val is None:
        raise RuntimeError("CVaR optimization returned no weights.")
    w_val = np.asarray(w_val, dtype=float).ravel()
    # clean and renormalize
    w_val = np.maximum(w_val, 0.0) if long_only else w_val
    if w_val.sum() != 0:
        w_val = w_val / w_val.sum()
    return w_val

def robust_return_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    gamma: float = 1.0,
    long_only: bool = True,
    lb: float = 0.0,
    ub: float = 1.0,
) -> np.ndarray:
    """
    Robust portfolio optimization with ellipsoidal uncertainty in mu.

    We consider mu in an ellipsoid around the estimate mu_hat:

        U = { mu : || Sigma^{-1/2} (mu - mu_hat) ||_2 <= gamma }

    The worst-case expected return for a weight vector w is:

        min_{mu in U} mu^T w = mu_hat^T w - gamma * || Sigma^{1/2} w ||_2

    We choose w by minimizing the *penalized* objective:

        minimize   gamma * ||Sigma^{1/2} w||_2 - mu_hat^T w

    subject to:
        sum(w) = 1
        lb <= w_i <= ub
        (and w_i >= 0 if long_only)

    This yields a portfolio that trades off nominal expected return vs. uncertainty
    implied by the covariance structure.

    Parameters
    ----------
    mu : (n,) array
        Estimated annualized expected returns.
    cov : (n, n) array
        Estimated annualized covariance matrix.
    gamma : float
        Robustness level; higher gamma => more conservative (more penalty on uncertainty).
    long_only : bool
        If True, enforce w_i >= 0 and lb clipped to 0.
    lb, ub : float
        Box bounds for weights.

    Returns
    -------
    w : (n,) numpy array of robust portfolio weights.
    """
    mu = np.asarray(mu).reshape(-1)
    cov = np.asarray(cov)
    n = mu.shape[0]

    if cov.shape != (n, n):
        raise ValueError("cov must be (n, n) with n = len(mu)")

    # eigen-based square root of covariance (handle semi-definiteness)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals_clipped = np.clip(eigvals, 0.0, None)
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T

    w = cp.Variable(n)

    if long_only:
        lb_eff = max(0.0, lb)
    else:
        lb_eff = lb

    lb_arr = np.full(n, lb_eff)
    ub_arr = np.full(n, ub)

    # objective: minimize gamma * ||Sigma^{1/2} w||_2 - mu^T w
    # (equivalently, maximize worst-case expected return)
    uncertainty_term = gamma * cp.norm(sqrt_cov @ w, 2)
    nominal_term = mu @ w
    objective = cp.Minimize(uncertainty_term - nominal_term)

    constraints = [
        cp.sum(w) == 1.0,
        w >= lb_arr,
        w <= ub_arr,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Robust optimization failed with status {prob.status}")

    w_val = w.value
    if w_val is None:
        raise RuntimeError("Robust optimization returned no weights.")
    w_val = np.asarray(w_val, dtype=float).ravel()

    # clean and renormalize
    if long_only:
        w_val = np.maximum(w_val, 0.0)
    if w_val.sum() != 0:
        w_val = w_val / w_val.sum()

    return w_val
