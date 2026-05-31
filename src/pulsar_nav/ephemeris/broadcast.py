"""GPS broadcast ephemeris (AA278 HW2 ``Ephemeris`` / ``brdc_data.npz``)."""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from pulsar_nav.ephemeris.paths import resolve_brdc_path


def _hw2_supplemental_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    candidates = [
        root / "HW2" / "supplemental" / "AA278_HW2_ephemeris_utils.py",
        root.parent / "AA278" / "HW2" / "supplemental" / "AA278_HW2_ephemeris_utils.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "AA278_HW2_ephemeris_utils.py not found under HW2/supplemental/"
    )


@lru_cache(maxsize=1)
def _load_hw2_ephemeris_module():
    path = _hw2_supplemental_path()
    spec = importlib.util.spec_from_file_location("aa278_hw2_ephemeris_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ephemeris module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GpsBroadcastEphemeris:
    """Thin wrapper around HW2 broadcast navigation (RINEX-style Keplerian)."""

    def __init__(self, brdc_path: str | Path | None = None):
        mod = _load_hw2_ephemeris_module()
        path = Path(brdc_path) if brdc_path is not None else resolve_brdc_path()
        self._ephem = mod.Ephemeris(str(path))

    def get_posvelclock_ecef_m(self, prn: int, et: float) -> tuple[np.ndarray, float]:
        """ECEF position/velocity (m, m/s) and clock correction (s)."""
        rv, clk = self._ephem.get_posvelclock(prn, et)
        return np.asarray(rv, float), float(clk)

    def has_prn(self, prn: int) -> bool:
        return prn in self._ephem.nav_dict.get("G", {})


@lru_cache(maxsize=1)
def default_gps_ephemeris() -> GpsBroadcastEphemeris:
    return GpsBroadcastEphemeris()
