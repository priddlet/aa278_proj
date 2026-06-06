"""Dilution-of-precision helpers for visibility validation."""

from __future__ import annotations

import numpy as np


def position_dop_from_los(los_unit_rows: np.ndarray) -> float:
    """
    Position PDOP (3D, no clock) from unit line-of-sight rows (N x 3).

    Uses ``PDOP = sqrt(tr((G^T G)^{-1}))`` via singular values of ``G``.
    Returns ``inf`` when fewer than four LOS or rank < 3.
    """
    g = np.asarray(los_unit_rows, float)
    if g.ndim != 2 or g.shape[1] != 3 or g.shape[0] < 4:
        return float("inf")

    _, s, _ = np.linalg.svd(g, full_matrices=False)
    s = np.sort(s[s > 1e-12])
    if s.size < 3:
        return float("inf")

    # Three largest singular values define 3D position observability.
    s3 = s[-3:]
    pdop = float(np.sqrt(np.sum(1.0 / (s3 * s3))))
    return pdop if np.isfinite(pdop) else float("inf")
