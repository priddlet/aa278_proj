"""Pseudorange measurement model (AA278 HW2 P3, MCI frame)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pulsar_nav.constants import C_LIGHT
from pulsar_nav.filter.state import NAV_STATE_DIM, NavState, idx_clock_bias, idx_pos
from pulsar_nav.spice.ephemeris import moon_position_icrs_km

KM_TO_M = 1000.0
C_KM_S = C_LIGHT / KM_TO_M
LIGHT_TIME_ITERS = 3


@dataclass
class PseudorangeMeasurement:
    """One pseudorange observation (meters)."""

    tx_position_mci_km: np.ndarray
    tx_clock_bias_km: float
    range_m: float
    sigma_m: float
    sat_id: int | str = 0

    @property
    def tx_position_mci_m(self) -> np.ndarray:
        return np.asarray(self.tx_position_mci_km, float) * KM_TO_M


def icrs_position_to_mci_km(position_icrs_m: np.ndarray, et: float) -> np.ndarray:
    """Convert ICRS position (m) to Moon-centered J2000 (km)."""
    r_icrs_km = np.asarray(position_icrs_m, float) / KM_TO_M
    return r_icrs_km - moon_position_icrs_km(et)


def predicted_pseudorange_m(
    state: NavState,
    tx_position_mci_km: np.ndarray,
    tx_clock_bias_km: float,
    et_rx: float,
    *,
    et0: float | None = None,
    get_tx_position: Callable[[float], tuple[np.ndarray, float]] | None = None,
) -> float:
    """
    Predicted pseudorange (m) with light-time correction.

    rho = ||r_rx - r_tx(t_tx)|| + b_rx - b_tx,  all converted to meters.
    """
    r_rx_km = icrs_position_to_mci_km(state.position_m, et_rx)
    b_rx_km = state.clock_bias_m / KM_TO_M

    t_rx = (et_rx - et0) if et0 is not None else 0.0
    t_tx = t_rx
    r_tx = np.asarray(tx_position_mci_km, float)
    b_tx = float(tx_clock_bias_km)

    for _ in range(LIGHT_TIME_ITERS):
        if get_tx_position is not None:
            r_tx, b_tx = get_tx_position(et0 + t_tx if et0 is not None else et_rx)
        dr = r_rx_km - r_tx
        g_km = float(np.linalg.norm(dr))
        t_tx = t_rx - g_km / C_KM_S

    rho_km = g_km + b_rx_km - b_tx
    return rho_km * KM_TO_M


def pseudorange_jacobian_m(
    state: NavState,
    tx_position_mci_km: np.ndarray,
    et_rx: float,
) -> np.ndarray:
    """
    Measurement Jacobian d(rho)/d(x) for 10-state ICRS filter (meters).

    Only position and receiver clock bias are nonzero.
    """
    r_rx_km = icrs_position_to_mci_km(state.position_m, et_rx)
    r_tx_km = np.asarray(tx_position_mci_km, float)
    dr = r_rx_km - r_tx_km
    g_km = float(np.linalg.norm(dr))
    if g_km < 1e-12:
        los = np.zeros(3)
    else:
        los = dr / g_km

    H = np.zeros(NAV_STATE_DIM)
    H[idx_pos] = los  # d(rho_m)/d(r_icrs_m); MCI km chain cancels km<->m factors
    H[idx_clock_bias] = 1.0
    return H


def pseudorange_residual(
    measurement: PseudorangeMeasurement,
    state: NavState,
    et_rx: float,
    *,
    et0: float | None = None,
    get_tx_position: Callable[[float], tuple[np.ndarray, float]] | None = None,
) -> float:
    """Innovation y = z - h(x) in meters."""
    pred = predicted_pseudorange_m(
        state,
        measurement.tx_position_mci_km,
        measurement.tx_clock_bias_km,
        et_rx,
        et0=et0,
        get_tx_position=get_tx_position,
    )
    return measurement.range_m - pred


def synthesize_pseudorange(
    true_position_icrs_m: np.ndarray,
    tx_position_mci_km: np.ndarray,
    tx_clock_bias_km: float,
    et_rx: float,
    rng: np.random.Generator,
    sigma_m: float,
    *,
    et0: float | None = None,
    sat_id: int | str = 0,
) -> PseudorangeMeasurement:
    """Noisy pseudorange from truth state (simulation)."""
    truth = NavState.from_pv(true_position_icrs_m, np.zeros(3))
    rho = predicted_pseudorange_m(
        truth, tx_position_mci_km, tx_clock_bias_km, et_rx, et0=et0
    )
    noise = rng.normal(0.0, sigma_m)
    return PseudorangeMeasurement(
        tx_position_mci_km=np.asarray(tx_position_mci_km, float).copy(),
        tx_clock_bias_km=float(tx_clock_bias_km),
        range_m=rho + noise,
        sigma_m=sigma_m,
        sat_id=sat_id,
    )
