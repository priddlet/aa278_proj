"""Resolve AA278 HW2 GPS broadcast ephemeris and Earth orientation kernels."""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

BRDC_FILENAME = "brdc_data.npz"
GPS_KERNEL_FILES = (
    "earth_assoc_itrf93.tf",
    "earth_latest_high_prec.bpc",
)


def default_data_dirs() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("PULSAR_NAV_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            _PROJECT_ROOT / "data" / "gnss",
            _PROJECT_ROOT / "HW2" / "data",
            _PROJECT_ROOT / "data" / "kernels",
            _PROJECT_ROOT.parent / "AA278" / "HW2" / "data",
        ]
    )
    return candidates


def resolve_brdc_path(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        path = Path(data_dir) / BRDC_FILENAME
        if path.is_file():
            return path
        raise FileNotFoundError(f"Broadcast ephemeris not found: {path}")

    for directory in default_data_dirs():
        candidate = directory / BRDC_FILENAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{BRDC_FILENAME} not found. Copy from AA278 HW2/data into "
        f"{_PROJECT_ROOT / 'HW2' / 'data'} or set PULSAR_NAV_DATA_DIR."
    )


def resolve_gps_kernel_dir(kernel_dir: str | Path | None = None) -> Path:
    if kernel_dir is not None:
        path = Path(kernel_dir)
        if all((path / name).is_file() for name in GPS_KERNEL_FILES):
            return path
        raise FileNotFoundError(f"GPS orientation kernels incomplete in {path}")

    for directory in default_data_dirs():
        if all((directory / name).is_file() for name in GPS_KERNEL_FILES):
            return directory
    raise FileNotFoundError(
        "Earth orientation kernels (earth_assoc_itrf93.tf, "
        "earth_latest_high_prec.bpc) not found in HW2/data."
    )
