"""HW2 P3 discrete process noise (CWNA + clock), adapted to 10-state ICRS nav filter."""

from __future__ import annotations

import numpy as np

from pulsar_nav.constants import C_LIGHT
from pulsar_nav.filter.state import NAV_STATE_DIM, idx_clock_bias, idx_clock_drift

KM_TO_M = 1000.0
C_KM_S = C_LIGHT / KM_TO_M

# RAFS clock PSDs from AA278 HW2 P3 (km²/s and km²/s³), scaled by c².
HW2_CLK_Q1_KM2_S = 3.70e-24 * C_KM_S**2
HW2_CLK_Q2_KM2_S3 = 1.87e-33 * C_KM_S**2

# Typical filter acceleration tuning (km/s² / √s).
DEFAULT_DYNAMICS_SIGMA_ACC_KM = 1e-6


def process_noise_hw2(
    dt_s: float,
    *,
    sigma_acc_km: float = DEFAULT_DYNAMICS_SIGMA_ACC_KM,
    clk_q1_km2_s: float = HW2_CLK_Q1_KM2_S,
    clk_q2_km2_s3: float = HW2_CLK_Q2_KM2_S3,
    nav_state_dim: int = NAV_STATE_DIM,
) -> np.ndarray:
    """
    Discrete Q matching HW2 ``get_process_noise`` for [r, v, clkb, clkdr].

    State units: meters and m/s for position/velocity; clock bias/drift in meters
  and m/s. Spare states (indices 8–9) receive zero process noise.
    """
    dt = float(dt_s)
    s2 = float(sigma_acc_km) ** 2
    q_cwna = np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]]) * s2

    q = np.zeros((nav_state_dim, nav_state_dim))
    km2_to_m2 = KM_TO_M**2
    for i in range(3):
        q[i, i] = q_cwna[0, 0] * km2_to_m2
        q[i, i + 3] = q_cwna[0, 1] * km2_to_m2
        q[i + 3, i] = q_cwna[1, 0] * km2_to_m2
        q[i + 3, i + 3] = q_cwna[1, 1] * km2_to_m2

    cb, cd = idx_clock_bias, idx_clock_drift
    q[cb, cb] = (clk_q1_km2_s * dt + clk_q2_km2_s3 * dt**3 / 3.0) * km2_to_m2
    q[cb, cd] = (clk_q2_km2_s3 * dt**2 / 2.0) * km2_to_m2
    q[cd, cb] = q[cb, cd]
    q[cd, cd] = clk_q2_km2_s3 * dt * km2_to_m2
    return q
