from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


def correl_distance(corr: pd.DataFrame) -> np.ndarray:
    """
    López de Prado distance measure: d_ij = sqrt(0.5 * (1 - corr_ij))
    Converts correlation matrix to distance matrix.
    """
    corr = corr.copy()
    corr = corr.clip(-1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    return dist


def hierarchical_clustering(dist: pd.DataFrame) -> np.ndarray:
    """
    Perform hierarchical clustering (linkage) using distance matrix.
    Returns linkage matrix.
    """
    # squareform: convert to condensed form
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="single")
    return Z


def get_quasi_diag(link: np.ndarray) -> list[int]:
    """
    Compute the quasi-diagonalization order from the linkage matrix.
    This returns the order in which assets appear in the hierarchical tree.
    """
    link = link.astype(int)
    num_items = link.shape[0] + 1
    cluster = [link[-1, 0], link[-1, 1]]  # last merge

    def unpack(node):
        if node < num_items:
            return [node]
        else:
            idx = node - num_items
            return unpack(link[idx, 0]) + unpack(link[idx, 1])

    return unpack(cluster[0]) + unpack(cluster[1])


def get_recursive_bisection(cov: pd.DataFrame, sort_ix: list[int]) -> pd.Series:
    """
    Compute HRP weights by recursive bisection.
    """
    weights = pd.Series(1.0, index=sort_ix)

    def cluster_var(indices: list[int]) -> float:
        subcov = cov.iloc[indices, indices]
        w = np.ones(len(indices)) / len(indices)
        return float(w @ subcov.values @ w)

    def split(indices: list[int]):
        if len(indices) == 1:
            return
        split_point = len(indices) // 2
        left = indices[:split_point]
        right = indices[split_point:]

        var_left = cluster_var(left)
        var_right = cluster_var(right)
        alpha = 1 - var_left / (var_left + var_right)

        weights[left] *= alpha
        weights[right] *= 1 - alpha

        split(left)
        split(right)

    split(sort_ix)
    return weights / weights.sum()


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Full HRP pipeline: correlation -> distance -> clustering -> quasi diag -> recursive bisection.
    Returns HRP weights indexed by asset names.
    """
    corr = returns.corr()
    cov = returns.cov()

    dist = correl_distance(corr)
    Z = hierarchical_clustering(dist)
    sort_ix = get_quasi_diag(Z)
    hrp = get_recursive_bisection(cov, sort_ix)

    # map index numbers back to ticker names
    assets = returns.columns
    hrp.index = [assets[i] for i in hrp.index]

    return hrp
