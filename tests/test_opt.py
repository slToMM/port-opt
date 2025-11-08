import numpy as np
from portopt.core.opt import solve_min_variance, trace_efficient_frontier, pick_max_sharpe

def test_min_variance_sum_to_one():
    mu = np.array([0.1, 0.12, 0.06])
    cov = np.array([[0.04,0.01,0.0],[0.01,0.05,0.0],[0.0,0.0,0.02]])
    w = solve_min_variance(mu, cov, long_only=True, lb=0.0, ub=1.0)
    assert w is not None
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    assert (w >= -1e-8).all()

def test_frontier_feasible():
    mu = np.array([0.08, 0.12, 0.06])
    cov = np.array([[0.04,0.01,0.0],[0.01,0.05,0.0],[0.0,0.0,0.02]])
    W, targets = trace_efficient_frontier(mu, cov, n_pts=5, long_only=True, lb=0.0, ub=1.0)
    assert W.shape[0] == targets.shape[0]
    idx, s = pick_max_sharpe(mu, cov, rf=0.02, frontier_weights=W)
    assert idx is not None
