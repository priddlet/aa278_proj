"""Dynamics-based EKF predict: MCI force model with ICRS state (Tier A / HW2-style)."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from pulsar_nav.measurements.pseudorange import icrs_position_to_mci_km
from pulsar_nav.propagation.dynamics import (
    DynamicsConfig,
    acceleration_mci,
    dynamics_jacobian_mci,
)
from pulsar_nav.spice.ephemeris import (
    icrs_position_from_mci_km,
    icrs_velocity_from_mci_km_s,
    mci_velocity_from_icrs_m_s,
)

KM_TO_M = 1000.0
PV_DIM = 6
RK45_RTOL = 1e-9
RK45_ATOL = 1e-12


def velocity_verlet_mci_step(
    r_mci_km: np.ndarray,
    v_mci_km_s: np.ndarray,
    et: float,
    dt_s: float,
    config: DynamicsConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """One velocity-Verlet step in MCI (km, km/s). Legacy / tests."""
    dt = float(dt_s)
    r = np.asarray(r_mci_km, float)
    v = np.asarray(v_mci_km_s, float)
    a0 = acceleration_mci(r, et, config)
    r_new = r + v * dt + 0.5 * a0 * dt**2
    a1 = acceleration_mci(r_new, et + dt, config)
    v_new = v + 0.5 * (a0 + a1) * dt
    return r_new, v_new


def _ode_pv_stm(
    t_rel_s: float,
    y: np.ndarray,
    et0: float,
    config: DynamicsConfig,
) -> np.ndarray:
    """Augmented ODE: MCI [r, v] and STM Φ (HW2 ``dynamics_ode_with_stm``)."""
    rv = y[:PV_DIM]
    phi = y[PV_DIM:].reshape(PV_DIM, PV_DIM)
    r, v = rv[:3], rv[3:]
    et = et0 + t_rel_s
    a = acceleration_mci(r, et, config)
    drv = np.concatenate((v, a))
    j = dynamics_jacobian_mci(r, v, et, config)
    dphi = (j @ phi).ravel()
    return np.concatenate((drv, dphi))


def propagate_mci_pv_with_stm(
    r_mci_km: np.ndarray,
    v_mci_km_s: np.ndarray,
    et0: float,
    dt_s: float,
    config: DynamicsConfig,
    *,
    rtol: float = RK45_RTOL,
    atol: float = RK45_ATOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate MCI position/velocity and STM over ``dt_s`` with ``solve_ivp`` (RK45).

    Returns ``(r_km, v_km_s, Phi_6x6)``.
    """
    dt = float(dt_s)
    y0 = np.concatenate(
        (np.asarray(r_mci_km, float), np.asarray(v_mci_km_s, float), np.eye(PV_DIM).ravel())
    )
    sol = solve_ivp(
        fun=lambda t, y: _ode_pv_stm(t, y, et0, config),
        t_span=(0.0, dt),
        y0=y0,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"MCI STM propagation failed: {sol.message}")
    yf = sol.y[:, -1]
    return yf[:3], yf[3:6], yf[6:].reshape(PV_DIM, PV_DIM)


def propagate_icrs_pv(
    position_icrs_m: np.ndarray,
    velocity_icrs_m_s: np.ndarray,
    et: float,
    dt_s: float,
    config: DynamicsConfig,
    *,
    rtol: float = RK45_RTOL,
    atol: float = RK45_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate ICRS position/velocity (m, m/s) with RK45 + STM path in MCI."""
    r_mci = icrs_position_to_mci_km(position_icrs_m, et)
    v_mci = mci_velocity_from_icrs_m_s(velocity_icrs_m_s, et)
    r_new, v_new, _ = propagate_mci_pv_with_stm(
        r_mci, v_mci, et, dt_s, config, rtol=rtol, atol=atol
    )
    et_next = et + dt_s
    pos_icrs_m = icrs_position_from_mci_km(r_new, et_next) * KM_TO_M
    vel_icrs_m_s = icrs_velocity_from_mci_km_s(v_new, et_next) * KM_TO_M
    return pos_icrs_m, vel_icrs_m_s


def pv_transition_stm(
    position_icrs_m: np.ndarray,
    velocity_icrs_m_s: np.ndarray,
    et: float,
    dt_s: float,
    config: DynamicsConfig,
    *,
    rtol: float = RK45_RTOL,
    atol: float = RK45_ATOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    6×6 STM for ICRS [r, v] (m, m/s) over one predict step.

    Moon translation cancels in the linearized increment, so Φ_mci applies to meter states.
    """
    r_mci = icrs_position_to_mci_km(position_icrs_m, et)
    v_mci = mci_velocity_from_icrs_m_s(velocity_icrs_m_s, et)
    _, _, phi = propagate_mci_pv_with_stm(
        r_mci, v_mci, et, dt_s, config, rtol=rtol, atol=atol
    )
    return phi, r_mci, v_mci


def numeric_pv_transition(
    position_icrs_m: np.ndarray,
    velocity_icrs_m_s: np.ndarray,
    et: float,
    dt_s: float,
    config: DynamicsConfig,
    *,
    eps_pos_m: float = 1.0,
    eps_vel_m_s: float = 0.05,
) -> np.ndarray:
    """Legacy finite-difference Φ (tests / fallback)."""
    x0 = np.concatenate(
        [np.asarray(position_icrs_m, float), np.asarray(velocity_icrs_m_s, float)]
    )
    f0 = np.concatenate(propagate_icrs_pv(position_icrs_m, velocity_icrs_m_s, et, dt_s, config))
    f = np.zeros((6, 6))
    for i in range(6):
        step = eps_pos_m if i < 3 else eps_vel_m_s
        step = step * max(abs(x0[i]), 1.0)
        xpert = x0.copy()
        xpert[i] += step
        r_p, v_p = propagate_icrs_pv(xpert[0:3], xpert[3:6], et, dt_s, config)
        f1 = np.concatenate([r_p, v_p])
        f[:, i] = (f1 - f0) / step
    return f


def full_state_transition(
    position_icrs_m: np.ndarray,
    velocity_icrs_m_s: np.ndarray,
    clock_bias_m: float,
    clock_drift_m_s: float,
    et: float,
    dt_s: float,
    config: DynamicsConfig,
    nav_state_dim: int,
    *,
    rtol: float = RK45_RTOL,
    atol: float = RK45_ATOL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full ``Phi`` and propagated state: RK45/STM on PV, CV on clock states.
    """
    from pulsar_nav.filter.state import idx_clock_bias, idx_clock_drift

    phi = np.eye(nav_state_dim)
    r_mci = icrs_position_to_mci_km(position_icrs_m, et)
    v_mci = mci_velocity_from_icrs_m_s(velocity_icrs_m_s, et)
    r_new, v_new, phi_pv = propagate_mci_pv_with_stm(
        r_mci, v_mci, et, dt_s, config, rtol=rtol, atol=atol
    )
    phi[0:6, 0:6] = phi_pv
    phi[idx_clock_bias, idx_clock_drift] = dt_s
    et_next = et + dt_s
    vec = np.zeros(nav_state_dim)
    vec[0:3] = icrs_position_from_mci_km(r_new, et_next) * KM_TO_M
    vec[3:6] = icrs_velocity_from_mci_km_s(v_new, et_next) * KM_TO_M
    vec[idx_clock_bias] = clock_bias_m + clock_drift_m_s * dt_s
    vec[idx_clock_drift] = clock_drift_m_s
    return phi, vec
