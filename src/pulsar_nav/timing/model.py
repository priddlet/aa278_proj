"""Pulsar phase / TOA timing model (Sheikh Eq. 7, simplified)."""

from __future__ import annotations

import numpy as np

from pulsar_nav.catalog.pulsar import Pulsar
from pulsar_nav.constants import C_LIGHT, SECS_PER_DAY


def mjd_to_seconds_since_pepoch(mjd: float, pepoch_mjd: float) -> float:
    return (mjd - pepoch_mjd) * SECS_PER_DAY


def phase_cycles(pulsar: Pulsar, mjd: float) -> float:
    return pulsar.phase_at_mjd(mjd)


def range_residual_from_toa_s(delta_t_s: float) -> float:
    """Convert TOA residual (s) to range residual (m)."""
    return C_LIGHT * delta_t_s


def predicted_toa_offset_s(
    position_error_m: np.ndarray,
    pulsar: Pulsar,
) -> float:
    """
    First-order TOA offset from position error (Sheikh linearized).

    delta_t ≈ (n_hat · delta_r) / c
    """
    return float(np.dot(pulsar.unit_vector_icrs, position_error_m) / C_LIGHT)
