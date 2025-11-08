import numpy as np
import pandas as pd
from portopt.core.stats import annualize_mean_cov, portfolio_metrics

def test_annualize_shapes():
    # 3 assets, 200 periods
    rets = pd.DataFrame(np.random.randn(200, 3) * 0.01, columns=["A","B","C"])
    mu, cov = annualize_mean_cov(rets, periods_per_year=252)
    assert mu.shape == (3,)
    assert cov.shape == (3,3)

def test_portfolio_metrics_basic():
    mu = np.array([0.10, 0.12, 0.06])
    cov = np.array([[0.04,0.01,0.0],[0.01,0.05,0.0],[0.0,0.0,0.02]])
    w = np.array([0.4,0.4,0.2])
    mets = portfolio_metrics(w, mu, cov, rf=0.0)
    assert 0.0 < mets["vol"] < 1.0
    assert 0.05 < mets["return"] < 0.2