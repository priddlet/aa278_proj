"""Gravity-scaled process noise for a constant-velocity (CV) lunar filter."""

from __future__ import annotations

import numpy as np

from pulsar_nav.propagation.dynamics import GM_MOON

KM_TO_M = 1000.0


def gravity_scaled_q_accel(
    r_moon_km: float,
    *,
    scale: float = 1.0,
    gm_moon_km3_s2: float = GM_MOON,
    floor_mps2: float = 1e-4,
) -> float:
    """
  CWNA acceleration intensity ``q_a`` (m^2/s^3) from lunar point-mass gravity at radius ``r``.

  ``q_a approx (scale  |  GM/r^2)^2`` with ``r`` the spacecraft-Moon center distance (km).
  Inflates automatically near periapsis (smaller ``r``).
    """
    r = float(max(r_moon_km, 1.0))
    a_km_s2 = gm_moon_km3_s2 / r**2
    a_mps2 = max(float(a_km_s2) * KM_TO_M, floor_mps2)
    return float((scale * a_mps2) ** 2)


def q_accel_at_radius_km(
    r_moon_km: float,
    *,
    scale: float = 1.0,
) -> float:
    """Alias: ``sqrt(q_a)`` in m/s^2 (effective acceleration sigma)."""
    return float(np.sqrt(gravity_scaled_q_accel(r_moon_km, scale=scale)))
