"""Classical orbital element conversions (Moon-centered)."""

from __future__ import annotations

import numpy as np

GM_MOON_KM3_S2 = 4902.800118


def _wrap_2pi(angle: float) -> float:
    return angle % (2.0 * np.pi)


def eccentric_anomaly(mean_anomaly: float, eccentricity: float, tol: float = 1e-10) -> float:
    """Solve Kepler's equation M = E - e sin(E) by Newton iteration."""
    m = _wrap_2pi(mean_anomaly)
    e_cur = np.pi
    for _ in range(50):
        f = e_cur - eccentricity * np.sin(e_cur) - m
        fp = 1.0 - eccentricity * np.cos(e_cur)
        e_cur -= f / fp
        if abs(f) < tol:
            break
    return _wrap_2pi(e_cur)


def coe_to_cart(
    coe: tuple[float, ...] | np.ndarray,
    mu: float = GM_MOON_KM3_S2,
) -> np.ndarray:
    """
    Convert classical orbital elements to Cartesian state in inertial frame.

    coe = (a, e, inc, raan, argp, M) with angles in radians, a in km.
    Returns [x, y, z, vx, vy, vz] in km and km/s.
    """
    a, ecc, inc, raan, argp, m_anom = coe
    e_anom = eccentric_anomaly(m_anom, ecc)
    n = np.sqrt(mu / a**3)

    r_pqw = np.array(
        [a * (np.cos(e_anom) - ecc), a * np.sqrt(1.0 - ecc**2) * np.sin(e_anom), 0.0]
    )
    v_pqw = (a * n / (1.0 - ecc * np.cos(e_anom))) * np.array(
        [-np.sin(e_anom), np.sqrt(1.0 - ecc**2) * np.cos(e_anom), 0.0]
    )

    c_o, s_o = np.cos(argp), np.sin(argp)
    c_ra, s_ra = np.cos(raan), np.sin(raan)
    c_i, s_i = np.cos(inc), np.sin(inc)

    rot = np.array(
        [
            [c_ra * c_o - s_ra * s_o * c_i, -c_ra * s_o - s_ra * c_o * c_i, s_ra * s_i],
            [s_ra * c_o + c_ra * s_o * c_i, -s_ra * s_o + c_ra * c_o * c_i, -c_ra * s_i],
            [s_o * s_i, c_o * s_i, c_i],
        ]
    )
    r = rot @ r_pqw
    v = rot @ v_pqw
    return np.concatenate((r, v))
