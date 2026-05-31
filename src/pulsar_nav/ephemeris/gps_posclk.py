"""GPS position and clock in Moon-centered inertial frame (MCI)."""

from __future__ import annotations

import numpy as np
import spiceypy as spice

from pulsar_nav.constants import C_LIGHT
from pulsar_nav.ephemeris.broadcast import GpsBroadcastEphemeris, default_gps_ephemeris
from pulsar_nav.spice.ephemeris import body_position_mci

C_KM_S = C_LIGHT / 1000.0


def get_gps_posclk_mci(
    et: float,
    prn: int,
    ephem: GpsBroadcastEphemeris | None = None,
) -> tuple[np.ndarray, float]:
    """
    GPS satellite state for pseudorange (AA278 HW2 P3).

    Returns
    -------
    r_gps_mci_km : ndarray (3,)
        Satellite position in Moon-centered J2000 (km).
    clkb_gps_km : float
        Satellite clock bias converted to km (c * delta_t).
    """
    ephem = ephem or default_gps_ephemeris()
    posvel_ecef_m, clk_tx = ephem.get_posvelclock_ecef_m(prn, et)
    pos_ecef_km = posvel_ecef_m[0:3] * 1e-3
    rot = np.asarray(spice.pxform("ITRF93", "J2000", et), dtype=float)
    pos_eci_km = rot @ pos_ecef_km
    earth_mci_km = body_position_mci("EARTH", et)
    r_gps_mci = pos_eci_km + earth_mci_km
    return r_gps_mci, clk_tx * C_KM_S


def iter_gps_prns(ephem: GpsBroadcastEphemeris | None = None) -> list[int]:
    """PRNs available in the broadcast file."""
    ephem = ephem or default_gps_ephemeris()
    return sorted(ephem._ephem.nav_dict.get("G", {}).keys())
