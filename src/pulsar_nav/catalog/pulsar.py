"""Pulsar ephemeris record and sky geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pulsar_nav.constants import C_LIGHT
from pulsar_nav.frames.icrs import unit_vector_icrs


@dataclass(frozen=True)
class Pulsar:
    """Millisecond pulsar entry for XNAV (timing + sky direction)."""

    name: str
    raj: str
    decj: str
    f0_hz: float
    f1_hz_s: float
    pepoch_mjd: float
    dm: float | None = None
    jname: str | None = None
    notes: str | None = None

    @property
    def unit_vector_icrs(self) -> np.ndarray:
        """Unit vector from Solar-System Barycenter toward pulsar (ICRS)."""
        return unit_vector_icrs(self.raj, self.decj)

    def line_of_sight_range_m(self, position_icrs_m: np.ndarray) -> float:
        """Scalar range observable: projection of position onto pulsar LOS (m)."""
        return float(np.dot(self.unit_vector_icrs, position_icrs_m))

    def range_sigma_m(self, toa_sigma_s: float) -> float:
        return C_LIGHT * toa_sigma_s

    def phase_at_mjd(self, mjd: float) -> float:
        """Rotation phase (cycles) from quadratic frequency model."""
        dt_s = (mjd - self.pepoch_mjd) * 86400.0
        return self.f0_hz * dt_s + 0.5 * self.f1_hz_s * dt_s**2

    def toa_model_mjd(self, pulse_number: int, mjd_ref: float | None = None) -> float:
        """Barycentric MJD of integer pulse number N (Sheikh Eq. 7, simplified)."""
        mjd_ref = mjd_ref if mjd_ref is not None else self.pepoch_mjd
        dt_s = pulse_number / self.f0_hz
        return mjd_ref + dt_s / 86400.0

    @classmethod
    def from_dict(cls, data: dict) -> Pulsar:
        return cls(
            name=data["name"],
            raj=data["raj"],
            decj=data["decj"],
            f0_hz=float(data["f0_hz"]),
            f1_hz_s=float(data["f1_hz_s"]),
            pepoch_mjd=float(data["pepoch_mjd"]),
            dm=float(data["dm"]) if data.get("dm") is not None else None,
            jname=data.get("jname"),
            notes=data.get("notes"),
        )


def pulsars_to_los_matrix(pulsars: Sequence[Pulsar]) -> np.ndarray:
    """Stack unit LOS vectors as rows (k x 3)."""
    return np.vstack([p.unit_vector_icrs for p in pulsars])
