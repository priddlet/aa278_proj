"""LunaNet relay constellation visibility (Walker delta, Moon-centered)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar_nav.propagation.dynamics import GM_MOON
from pulsar_nav.propagation.elements import coe_to_cart
from pulsar_nav.visibility.geometry import elevation_angle


@dataclass(frozen=True)
class LunaNetConfig:
    """Default notional LunaNet Walker constellation (AA278 HW2 P1)."""

    sma_km: float = 8000.0
    eccentricity: float = 0.0
    inclination_rad: float = np.deg2rad(55.0)
    argp_rad: float = 0.0
    walker_f: int = 1
    n_sats: int = 16
    n_planes: int = 4
    min_elevation_deg: float = 5.0


def construct_walker_constellation(
    sma_km: float,
    ecc: float,
    inc_rad: float,
    argp_rad: float,
    f: int,
    n_sats: int,
    n_planes: int,
) -> np.ndarray:
    """Return COE array (n_sats, 6): a, e, i, RAAN, argp, M."""
    if n_sats % n_planes != 0:
        raise ValueError("n_sats must be divisible by n_planes")
    sat_per_plane = n_sats // n_planes
    coes = np.zeros((n_sats, 6))
    for i in range(n_sats):
        plane = i // sat_per_plane
        slot = i % sat_per_plane
        raan = 2.0 * np.pi * plane / n_planes
        m_anom = 2.0 * np.pi * slot / sat_per_plane + 2.0 * np.pi * f * plane / n_sats
        coes[i] = [sma_km, ecc, inc_rad, raan, argp_rad, np.mod(m_anom, 2.0 * np.pi)]
    return coes


def propagate_constellation_mci(
    coes: np.ndarray,
    t_rel_s: np.ndarray,
    *,
    mu_km3_s2: float = GM_MOON,
) -> np.ndarray:
    """
    Propagate circular/elliptic COEs in inertial Moon frame (no OP rotation).

    Returns positions (n_sats, n_times, 3) in km.
    """
    coes = np.asarray(coes, dtype=float)
    t_rel = np.asarray(t_rel_s, dtype=float)
    n_sats = coes.shape[0]
    n_t = t_rel.size
    positions = np.zeros((n_sats, n_t, 3))

    for s in range(n_sats):
        a, e, inc, raan, argp, m0 = coes[s]
        n_mean = np.sqrt(mu_km3_s2 / a**3)
        for k, t in enumerate(t_rel):
            m = m0 + n_mean * t
            coe_t = (a, e, inc, raan, argp, np.mod(m, 2.0 * np.pi))
            positions[s, k] = coe_to_cart(coe_t, mu=mu_km3_s2)[:3]
    return positions


def lonet_visibility(
    spacecraft_mci_km: np.ndarray,
    relay_positions_mci_km: np.ndarray,
    *,
    min_elevation_deg: float = 5.0,
) -> tuple[bool, int, float]:
    """
    Check LunaNet relay visibility from spacecraft.

    Returns (any_visible, count_visible, max_elevation_deg).
    """
    sc = np.asarray(spacecraft_mci_km, float)
    zenith = sc / np.linalg.norm(sc)
    min_el = np.deg2rad(min_elevation_deg)
    max_el = -np.pi / 2.0
    count = 0
    for r_sat in relay_positions_mci_km:
        el = elevation_angle(sc, r_sat, zenith_hat=zenith)
        max_el = max(max_el, el)
        if el >= min_el:
            count += 1
    return count > 0, count, float(np.rad2deg(max_el))
