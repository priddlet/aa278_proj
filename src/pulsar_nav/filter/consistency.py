"""Innovation consistency (NIS) for stacked EKF updates."""

from __future__ import annotations

import numpy as np


def normalized_innovation_squared(
    innovation: np.ndarray,
    innovation_covariance: np.ndarray,
) -> float:
    """
    NIS = y^T S^{-1} y for innovation vector ``y`` and covariance ``S``.

    Under a consistent linear-Gaussian filter, NIS ~ chi-squared(df=len(y)).
    Values >> chi2_95(df) suggest divergence or mis-modeled noise/geometry.
    """
    y = np.asarray(innovation, float).ravel()
    s = np.asarray(innovation_covariance, float)
    if y.size == 0:
        return float("nan")
    if s.ndim == 1:
        s = np.diag(s)
    try:
        return float(y @ np.linalg.solve(s, y))
    except np.linalg.LinAlgError:
        return float("inf")


def chi2_95_threshold(dof: int) -> float:
    """95th percentile of chi-squared(df) — quick NIS alarm threshold."""
    from scipy.stats import chi2

    return float(chi2.ppf(0.95, dof))
