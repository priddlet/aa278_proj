"""Moon-centered dynamics for lunar orbiter truth propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pulsar_nav.spice.ephemeris import body_position_mci

# AA278 / DE440 values (km, km^3/s^2)
GM_MOON = 4902.800118
GM_EARTH = 398600.435507
GM_SUN = 132712440041.279419

# SRP scaling (HW2 P3): gamma * SRP_SCALE * rho_s/|rho_s|^3 -> km/s^2
C_LIGHT_MS = 299_792_458.0
P0_N_M2 = 1367.0 / C_LIGHT_MS
AU_KM = 149_597_870.7
SRP_SCALE = P0_N_M2 * AU_KM**2 * 1e-3

# Moon gravity field (optional J2)
MOON_J2 = 2.033e-4
MOON_RADIUS_KM = 1737.4

# HW2 P3 SRP truth spacecraft (gamma = C_R * A / m, course units in dynamics)
HW2_SRP_CR = 1.8
HW2_SRP_AREA_M2 = 1.0
HW2_SRP_MASS_KG = 850.0
DEFAULT_GAMMA_SRP = HW2_SRP_CR * HW2_SRP_AREA_M2 / HW2_SRP_MASS_KG


def dynamics_config_for_sim(*, include_disturbances: bool = False) -> DynamicsConfig:
    """
    Shared force model for truth propagation and filter dynamics predict.

    Baseline (HW2 P2): point-mass Moon + Earth + Sun indirect terms.
    With disturbances (HW2 P3): add Moon J2 and SRP at ``DEFAULT_GAMMA_SRP``.
    """
    if not include_disturbances:
        return DynamicsConfig(include_earth=True, include_sun=True)
    return DynamicsConfig(
        include_earth=True,
        include_sun=True,
        include_moon_j2=True,
        include_srp=True,
        gamma_srp=DEFAULT_GAMMA_SRP,
    )


@dataclass
class DynamicsConfig:
    """Force model selection for truth vs. filter-grade propagation."""

    include_earth: bool = True
    include_sun: bool = True
    include_srp: bool = False
    gamma_srp: float = 0.0  # km^3/s^2 style coefficient (course convention)
    include_moon_j2: bool = False
    ephemeris_callback: Callable[[float], tuple[np.ndarray, np.ndarray]] | None = None


def _inv_cube(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm**3


def grad_inv_cube(vec: np.ndarray) -> np.ndarray:
    """Jacobian of vec / |vec|^3 with respect to vec (3 x 3)."""
    v = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros((3, 3))
    return np.eye(3) / norm**3 - 3.0 * np.outer(v, v) / norm**5


def moon_j2_acceleration(r_mci: np.ndarray) -> np.ndarray:
    """J2 perturbation in Moon-centered frame (km/s^2)."""
    x, y, z = r_mci
    r = np.linalg.norm(r_mci)
    factor = 1.5 * GM_MOON * MOON_J2 * MOON_RADIUS_KM**2 / r**5
    ax = factor * x * (5.0 * z**2 / r**2 - 1.0)
    ay = factor * y * (5.0 * z**2 / r**2 - 1.0)
    az = factor * z * (5.0 * z**2 / r**2 - 3.0)
    return np.array([ax, ay, az])


def acceleration_mci(
    r_mci: np.ndarray,
    et: float,
    config: DynamicsConfig,
) -> np.ndarray:
    """
    Specific force on spacecraft in MCI (J2000, origin at Moon center), km/s^2.

    Point-mass Moon + indirect third-body terms (Earth, Sun) per AA278 HW2.
    """
    r = np.asarray(r_mci, dtype=float)
    acc = -GM_MOON * _inv_cube(r)

    if config.include_earth or config.include_sun:
        if config.ephemeris_callback is not None:
            r_earth, r_sun = config.ephemeris_callback(et)
        else:
            r_earth = body_position_mci("EARTH", et) if config.include_earth else None
            r_sun = body_position_mci("SUN", et) if config.include_sun else None

        if config.include_earth and r_earth is not None:
            rho_e = r - r_earth
            acc += -GM_EARTH * (_inv_cube(rho_e) + _inv_cube(r_earth))

        if config.include_sun and r_sun is not None:
            rho_s = r - r_sun
            acc += -GM_SUN * (_inv_cube(rho_s) + _inv_cube(r_sun))
            if config.include_srp and config.gamma_srp != 0.0:
                acc += -SRP_SCALE * config.gamma_srp * _inv_cube(rho_s)

    if config.include_moon_j2:
        acc += moon_j2_acceleration(r)

    return acc


def acceleration_mci_jacobian_r(
    r_mci: np.ndarray,
    et: float,
    config: DynamicsConfig,
) -> np.ndarray:
    """
    Jacobian da/dr for MCI specific force (km/s^2 per km).

    Point-mass Moon, indirect Earth/Sun, optional SRP; J2 via local FD if enabled.
    """
    r = np.asarray(r_mci, dtype=float)
    j = -GM_MOON * grad_inv_cube(r)

    if config.include_earth or config.include_sun:
        if config.ephemeris_callback is not None:
            r_earth, r_sun = config.ephemeris_callback(et)
        else:
            r_earth = body_position_mci("EARTH", et) if config.include_earth else None
            r_sun = body_position_mci("SUN", et) if config.include_sun else None

        if config.include_earth and r_earth is not None:
            rho_e = r - r_earth
            j += -GM_EARTH * grad_inv_cube(rho_e)

        if config.include_sun and r_sun is not None:
            rho_s = r - r_sun
            j += -GM_SUN * grad_inv_cube(rho_s)
            if config.include_srp and config.gamma_srp != 0.0:
                j += -SRP_SCALE * config.gamma_srp * grad_inv_cube(rho_s)

    if config.include_moon_j2:
        eps = 1e-3
        a0 = moon_j2_acceleration(r)
        for i in range(3):
            rp = r.copy()
            rp[i] += eps
            j[:, i] += (moon_j2_acceleration(rp) - a0) / eps

    return j


def dynamics_jacobian_mci(
    r_mci: np.ndarray,
    v_mci: np.ndarray,
    et: float,
    config: DynamicsConfig,
) -> np.ndarray:
    """6 x 6 Jacobian of [r_dot, v_dot] = [v, a(r)] w.r.t. [r, v] in MCI."""
    _ = v_mci  # velocity does not enter acceleration
    j = np.zeros((6, 6))
    j[0:3, 3:6] = np.eye(3)
    j[3:6, 0:3] = acceleration_mci_jacobian_r(r_mci, et, config)
    return j


def dynamics_ode(
    t_rel_s: float,
    state_mci: np.ndarray,
    et0: float,
    config: DynamicsConfig,
) -> np.ndarray:
    """ODE RHS for [r, v] in MCI; t_rel_s is seconds since et0."""
    r = state_mci[:3]
    v = state_mci[3:6]
    et = et0 + t_rel_s
    a = acceleration_mci(r, et, config)
    return np.concatenate((v, a))
