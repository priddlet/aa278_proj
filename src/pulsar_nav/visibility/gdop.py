"""Dilution-of-precision helpers for visibility validation."""

from __future__ import annotations

import numpy as np


def position_dop_from_los(los_unit_rows: np.ndarray) -> float:
    """
    Position DOP (3D, no clock) from unit line-of-sight rows.

    Returns ``inf`` when fewer than four independent LOS are available.
    """
    g = np.asarray(los_unit_rows, float)
    if g.ndim != 2 or g.shape[1] != 3 or g.shape[0] < 4:
        return float("inf")
    try:
        q = np.linalg.inv(g @ g.T)
    except np.linalg.LinAlgError:
        return float("inf")
    pdop = float(np.sqrt(np.trace(q)))
    return pdop if np.isfinite(pdop) else float("inf")
