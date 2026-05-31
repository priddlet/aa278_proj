"""Load NAIF SPICE kernels for lunar simulation."""

from __future__ import annotations

import os
from pathlib import Path

import spiceypy as spice

KERNEL_FILES = (
    "naif0012.tls",
    "de440.bsp",
    "moon_pa_de440_200625.bpc",
    "moon_de440_250416.tf",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def default_kernel_dirs() -> list[Path]:
    """Search paths for SPICE kernels (first match wins)."""
    candidates = []
    env = os.environ.get("PULSAR_NAV_KERNEL_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            _PROJECT_ROOT / "data" / "kernels",
            _PROJECT_ROOT / "HW2" / "data",
            _PROJECT_ROOT.parent / "AA278" / "HW2" / "data",
            _PROJECT_ROOT.parent / "AA278" / "HW1" / "data",
        ]
    )
    return candidates


def resolve_kernel_dir(kernel_dir: str | Path | None = None) -> Path:
    if kernel_dir is not None:
        path = Path(kernel_dir)
        if _has_required_kernels(path):
            return path
        raise FileNotFoundError(f"SPICE kernels incomplete in {path}")

    for path in default_kernel_dirs():
        if _has_required_kernels(path):
            return path

    raise FileNotFoundError(
        "SPICE kernels not found. Place "
        f"{', '.join(KERNEL_FILES)} in data/kernels/ or set PULSAR_NAV_KERNEL_DIR. "
        "Download from https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
    )


def _has_required_kernels(path: Path) -> bool:
    return all((path / name).is_file() for name in KERNEL_FILES)


def load_kernels(
    kernel_dir: str | Path | None = None,
    *,
    clear: bool = True,
    load_gps_frames: bool = False,
) -> Path:
    """
    Furnish leap-second, planetary ephemeris, and Moon orientation kernels.

    Set ``load_gps_frames=True`` to also load Earth ITRF93/J2000 kernels
    required for GPS broadcast ephemeris (HW2 ``earth_*`` files).
    """
    path = resolve_kernel_dir(kernel_dir)
    if clear:
        spice.kclear()
    for name in KERNEL_FILES:
        spice.furnsh(str(path / name))
    if load_gps_frames:
        load_gps_orientation_kernels(path)
    return path


def load_gps_orientation_kernels(kernel_dir: str | Path | None = None) -> Path:
    """Furnish Earth frame kernels for ITRF93 <-> J2000 (GPS ECEF to MCI)."""
    from pulsar_nav.ephemeris.paths import resolve_gps_kernel_dir

    path = resolve_gps_kernel_dir(kernel_dir)
    for name in ("earth_assoc_itrf93.tf", "earth_latest_high_prec.bpc"):
        spice.furnsh(str(path / name))
    return path
