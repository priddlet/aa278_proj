"""Simplified GPS constellation for GNSS sidelobe simulation (MCI)."""

from __future__ import annotations

import numpy as np

from pulsar_nav.visibility.geometry import elevation_angle, moon_blocks_los

GPS_ORBIT_RADIUS_KM = 26_560.0
DEFAULT_N_GPS = 6


def gps_satellite_offsets_km(n_sats: int = DEFAULT_N_GPS) -> np.ndarray:
    """Body-frame positions on a sphere around Earth (km)."""
    offsets = []
    for k in range(n_sats):
        nu = 2.0 * np.pi * k / n_sats
        # Mix inclinations for better geometry
        inc = np.deg2rad(55.0) if k % 2 == 0 else np.deg2rad(20.0)
        offsets.append(
            GPS_ORBIT_RADIUS_KM
            * np.array(
                [
                    np.cos(nu),
                    np.sin(nu) * np.cos(inc),
                    np.sin(nu) * np.sin(inc),
                ]
            )
        )
    return np.vstack(offsets)


def gps_positions_mci(
    earth_mci_km: np.ndarray,
    et: float,
    *,
    n_sats: int = DEFAULT_N_GPS,
) -> np.ndarray:
    """
    Notional GPS constellation co-rotating with Earth in MCI (km).

    Uses a slow Earth-spin phase from ``et`` for along-track motion.
    """
    earth = np.asarray(earth_mci_km, float)
    offsets = gps_satellite_offsets_km(n_sats)
    phase = (et % 86400.0) / 86400.0 * 2.0 * np.pi
    c, s = np.cos(phase), np.sin(phase)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return earth + offsets @ rot.T


def visible_gps_indices(
    spacecraft_mci_km: np.ndarray,
    gps_mci_km: np.ndarray,
    *,
    min_elevation_deg: float = 5.0,
) -> list[int]:
    """Return indices of GPS sats above elevation mask and not Moon-occulted."""
    sc = np.asarray(spacecraft_mci_km, float)
    zenith = sc / np.linalg.norm(sc)
    min_el = np.deg2rad(min_elevation_deg)
    visible: list[int] = []
    for i, r_sat in enumerate(gps_mci_km):
        if moon_blocks_los(sc, r_sat):
            continue
        if elevation_angle(sc, r_sat, zenith_hat=zenith) >= min_el:
            visible.append(i)
    return visible
