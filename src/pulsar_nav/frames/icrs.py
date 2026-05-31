"""ICRS coordinate helpers (no astropy dependency)."""

from __future__ import annotations

import re

import numpy as np

_HMS = re.compile(
    r"^(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)$"
)
_DMS = re.compile(
    r"^(?P<sign>[+-]?)(?P<d>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)$"
)


def hms_to_rad(hms: str) -> float:
    m = _HMS.match(hms.strip())
    if not m:
        raise ValueError(f"Invalid RAJ string: {hms!r}")
    h = int(m["h"])
    mi = int(m["m"])
    s = float(m["s"])
    hours = h + mi / 60.0 + s / 3600.0
    return hours * (np.pi / 12.0)


def dms_to_rad(dms: str) -> float:
    m = _DMS.match(dms.strip())
    if not m:
        raise ValueError(f"Invalid DecJ string: {dms!r}")
    sign = -1.0 if m["sign"] == "-" else 1.0
    d = int(m["d"])
    mi = int(m["m"])
    s = float(m["s"])
    deg = sign * (d + mi / 60.0 + s / 3600.0)
    return np.deg2rad(deg)


def unit_vector_icrs(raj: str, decj: str) -> np.ndarray:
    """Unit vector from ICRS RA (h:m:s) and Dec (d:m:s) strings."""
    ra = hms_to_rad(raj)
    dec = dms_to_rad(decj)
    return np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
        dtype=float,
    )
