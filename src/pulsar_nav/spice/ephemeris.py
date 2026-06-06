"""SPICE ephemeris queries in Moon-centered inertial (MCI / J2000) frame."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import spiceypy as spice

from pulsar_nav.constants import SECS_PER_DAY

KM_TO_M = 1000.0
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


def icrs_position_from_mci_km(position_mci_km: np.ndarray, et: float) -> np.ndarray:
    """Alias for MCI -> ICRS position (km)."""
    return mci_to_icrs_position(position_mci_km, et)


def moon_velocity_icrs_km_s(et: float, *, fd_dt_s: float = 1.0) -> np.ndarray:
    """Moon center velocity in ICRS (km/s) via centered difference."""
    dt = float(fd_dt_s)
    return (moon_position_icrs_km(et + dt) - moon_position_icrs_km(et - dt)) / (2.0 * dt)


def mci_velocity_from_icrs_m_s(velocity_icrs_m_s: np.ndarray, et: float) -> np.ndarray:
    """Spacecraft velocity in MCI (km/s) from ICRS (m/s)."""
    return np.asarray(velocity_icrs_m_s, float) / KM_TO_M - moon_velocity_icrs_km_s(et)


def icrs_velocity_from_mci_km_s(velocity_mci_km_s: np.ndarray, et: float) -> np.ndarray:
    """Spacecraft velocity in ICRS (km/s) from MCI (km/s)."""
    return moon_velocity_icrs_km_s(et) + np.asarray(velocity_mci_km_s, dtype=float)


def mci_to_pa_rotation(et: float) -> np.ndarray:
    """6x6 state rotation from MCI (J2000) to Moon principal-axis (body-fixed) frame."""
    return np.asarray(spice.sxform("J2000", "MOON_PA", et), dtype=float)


def mci_to_op_rotation(et: float) -> np.ndarray:
    """
    6x6 state rotation from MCI (J2000) to Earth orbital-plane (OP) frame.

    AA278 HW2 P2.3: ``z_OP`` aligns with Earth-Moon ``r  x  v``; ``x_OP`` is built
    from the lunar pole (MOON_PA z-axis in MCI) crossed with ``z_OP``. The result
    is block-diagonal (same 3 x 3 on position and velocity), so ``R.T`` is the
    inverse map from OP to MCI - unlike ``mci_to_pa_rotation`` via ``sxform``.
    """
    rv_em, _ = spice.spkezr("EARTH", et, "J2000", "NONE", "MOON")
    r_em = np.asarray(rv_em[:3], dtype=float)
    v_em = np.asarray(rv_em[3:6], dtype=float)

    z_op = np.cross(r_em, v_em)
    z_op /= np.linalg.norm(z_op)

    pole_mci = spice.pxform("MOON_PA", "J2000", et) @ np.array([0.0, 0.0, 1.0])
    pole_mci /= np.linalg.norm(pole_mci)

    x_op = np.cross(pole_mci, z_op)
    x_op /= np.linalg.norm(x_op)

    y_op = np.cross(z_op, x_op)
    y_op /= np.linalg.norm(y_op)

    c_op = np.vstack((x_op, y_op, z_op))
    rot = np.zeros((6, 6))
    rot[:3, :3] = c_op
    rot[3:, 3:] = c_op
    return rot
