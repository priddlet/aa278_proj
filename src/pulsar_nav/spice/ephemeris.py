"""SPICE ephemeris queries in Moon-centered inertial (MCI / J2000) frame."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import spiceypy as spice

from pulsar_nav.constants import SECS_PER_DAY

MJD_J2000_TT = 51544.5


def et_to_mjd(et: float) -> float:
    return et / SECS_PER_DAY + MJD_J2000_TT


def mjd_to_et(mjd: float) -> float:
    return (mjd - MJD_J2000_TT) * SECS_PER_DAY


def str_to_et(time_str: str) -> float:
    """Parse UTC/TDB calendar string to SPICE ET (seconds past J2000)."""
    s = time_str.strip()
    if s.upper().endswith((" UTC", " TDB", " TDT")):
        return spice.str2et(s)
    # SPICE prefers space separator over ISO 'T'
    if "T" in s and " " not in s[:20]:
        s = s.replace("T", " ", 1)
    return spice.str2et(s + " UTC")


def datetime_to_et(dt: datetime, timesys: str = "UTC") -> float:
    sec = dt.second + dt.microsecond / 1e6
    base = (
        f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d} "
        f"{dt.hour:02d}:{dt.minute:02d}:{sec:09.6f}"
    )
    timesys = timesys.upper()
    if timesys == "UTC":
        return spice.str2et(base + " UTC")
    if timesys in ("TT", "TDT"):
        return spice.str2et(base + " TDT")
    if timesys in ("TDB", "ET"):
        return spice.str2et(base + " TDB")
    raise ValueError(f"Unsupported time system: {timesys}")


def body_position_mci(body: str, et: float) -> np.ndarray:
    """
    Position of body relative to Moon center in J2000 (MCI), km.

    body: 'EARTH' or 'SUN' (SPICE names).
    """
    pos, _ = spice.spkpos(body, et, "J2000", "NONE", "MOON")
    return np.asarray(pos, dtype=float)


def moon_position_icrs_km(et: float) -> np.ndarray:
    """Moon center position in J2000 (ICRS) relative to solar-system barycenter, km."""
    pos, _ = spice.spkpos("MOON", et, "J2000", "NONE", "SOLAR SYSTEM BARYCENTER")
    return np.asarray(pos, dtype=float)


def mci_to_icrs_position(position_mci_km: np.ndarray, et: float) -> np.ndarray:
    """Spacecraft position in ICRS (km) from Moon-centered J2000 position."""
    return moon_position_icrs_km(et) + np.asarray(position_mci_km, dtype=float)


def mci_to_op_rotation(et: float) -> np.ndarray:
    """6x6 state rotation matrix from MCI (J2000) to Moon principal-axis frame."""
    return np.asarray(spice.sxform("J2000", "MOON_PA", et), dtype=float)
