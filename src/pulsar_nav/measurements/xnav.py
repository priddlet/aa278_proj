"""XNAV scalar measurements (Sheikh & Pines delta-correction, linearized)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar_nav.catalog.pulsar import Pulsar
from pulsar_nav.constants import C_LIGHT
from pulsar_nav.filter.state import NAV_STATE_DIM, NavState, idx_pos


@dataclass
class XNAVMeasurement:
    """One pulsar range measurement (meters along LOS)."""

    pulsar: Pulsar
    range_m: float
    sigma_m: float

    @property
    def innovation(self) -> float:
        return self.range_m


def predicted_range(state: NavState, pulsar: Pulsar) -> float:
    """Predicted scalar observable h(x) = n_hat · r (m)."""
    return pulsar.line_of_sight_range_m(state.position_m)


def measurement_jacobian(pulsar: Pulsar) -> np.ndarray:
    """
    H row for EKF: dh/dx.

    For h = n_hat · r, only position states are nonzero (first 3).
    """
    H = np.zeros(NAV_STATE_DIM)
    H[idx_pos] = pulsar.unit_vector_icrs
    return H


def synthesize_measurement(
    pulsar: Pulsar,
    true_position_m: np.ndarray,
    rng: np.random.Generator,
    toa_sigma_s: float,
) -> XNAVMeasurement:
    """Generate noisy LOS range measurement from truth (simulation)."""
    sigma_m = pulsar.range_sigma_m(toa_sigma_s)
    truth = pulsar.line_of_sight_range_m(true_position_m)
    noise = rng.normal(0.0, sigma_m)
    return XNAVMeasurement(pulsar=pulsar, range_m=truth + noise, sigma_m=sigma_m)


def range_residual(
    measurement: XNAVMeasurement,
    state: NavState,
) -> float:
    """Innovation y = z - h(x_hat)."""
    return measurement.range_m - predicted_range(state, measurement.pulsar)


def batch_geometry_matrix(pulsars: list[Pulsar]) -> np.ndarray:
    """LOS matrix A (k x 3) for batch least-squares position fix (Sheikh Eq. 34)."""
    return np.vstack([p.unit_vector_icrs for p in pulsars])


def batch_position_fix(
    measurements: list[XNAVMeasurement],
    initial_position_m: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve A @ delta_r = b for position correction (single-epoch batch).

    Uses measured ranges as absolute constraints when initial is origin;
    for residual form, pass precomputed innovations instead.
    """
    A = batch_geometry_matrix([m.pulsar for m in measurements])
    z = np.array([m.range_m for m in measurements])
    if initial_position_m is None:
        initial_position_m = np.zeros(3)
    b = z - A @ initial_position_m
    delta_r, *_ = np.linalg.lstsq(A, b, rcond=None)
    return initial_position_m + delta_r


def toa_to_range_m(delta_t_s: float) -> float:
    """Convert TOA residual (s) to range residual (m)."""
    return C_LIGHT * delta_t_s
