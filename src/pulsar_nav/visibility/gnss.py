"""GNSS sidelobe visibility via Earth elevation from the lunar orbiter."""

from __future__ import annotations

import numpy as np

from pulsar_nav.visibility.geometry import earth_elevation_mci, moon_blocks_los


def gnss_earth_visible(
    spacecraft_mci_km: np.ndarray,
    earth_mci_km: np.ndarray,
    *,
    min_elevation_deg: float = 5.0,
    check_moon_occultation: bool = True,
) -> bool:
    """
    GNSS sidelobe availability proxy: Earth above elevation mask and not occulted.

    Far-side orbits have negative Earth elevation; near-side passes have
    positive elevation when Earth is in view above the local lunar horizon.
    """
    if check_moon_occultation and moon_blocks_los(spacecraft_mci_km, earth_mci_km):
        return False
    elev_rad = earth_elevation_mci(spacecraft_mci_km, earth_mci_km)
    return elev_rad >= np.deg2rad(min_elevation_deg)


def earth_elevation_deg(
    spacecraft_mci_km: np.ndarray,
    earth_mci_km: np.ndarray,
) -> float:
    return float(np.rad2deg(earth_elevation_mci(spacecraft_mci_km, earth_mci_km)))
