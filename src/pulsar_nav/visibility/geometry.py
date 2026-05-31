"""Line-of-sight and elevation geometry for lunar orbiters."""

from __future__ import annotations

import numpy as np

from pulsar_nav.propagation.dynamics import MOON_RADIUS_KM


def unit_vector(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return v / n


def elevation_angle(
    observer_km: np.ndarray,
    target_km: np.ndarray,
    zenith_hat: np.ndarray | None = None,
) -> float:
    """
    Elevation of ``target`` as seen from ``observer`` (radians).

    Zenith is the outward normal from the Moon center through the observer
    (valid for orbiting spacecraft in Moon-centered frame).
    """
    obs = np.asarray(observer_km, float)
    tgt = np.asarray(target_km, float)
    if zenith_hat is None:
        zenith_hat = unit_vector(obs)
    else:
        zenith_hat = unit_vector(zenith_hat)

    los = tgt - obs
    los_hat = unit_vector(los)
    sin_el = float(np.clip(np.dot(los_hat, zenith_hat), -1.0, 1.0))
    return float(np.arcsin(sin_el))


def earth_elevation_mci(
    spacecraft_mci_km: np.ndarray,
    earth_mci_km: np.ndarray,
) -> float:
    """Elevation angle of Earth from a lunar orbiter in MCI (radians)."""
    return elevation_angle(spacecraft_mci_km, earth_mci_km)


def moon_blocks_los(
    observer_mci_km: np.ndarray,
    target_mci_km: np.ndarray,
    moon_radius_km: float = MOON_RADIUS_KM,
) -> bool:
    """
    True if the Moon body occults the target (no line of sight).

    Uses the sphere occultation test from poliastro-style line_of_sight.
    """
    r1 = np.asarray(observer_mci_km, float)
    r2 = np.asarray(target_mci_km, float)
    r1n = np.linalg.norm(r1)
    r2n = np.linalg.norm(r2)
    if r1n <= moon_radius_km or r2n <= moon_radius_km:
        return True

    cos_theta = np.clip(np.dot(r1, r2) / (r1n * r2n), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    theta1 = np.arccos(np.clip(moon_radius_km / r1n, -1.0, 1.0))
    theta2 = np.arccos(np.clip(moon_radius_km / r2n, -1.0, 1.0))
    return bool(theta > theta1 + theta2)
