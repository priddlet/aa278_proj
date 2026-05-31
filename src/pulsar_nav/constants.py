"""Physical and timing constants."""

import math

C_LIGHT = 299_792_458.0  # m/s
MJD_J2000 = 51544.5
SECS_PER_DAY = 86400.0

# Moon GM (km^3/s^2) for orbit-period helpers — matches propagation/dynamics.py.
GM_MOON_KM3_S2 = 4902.800118
ELFO_SMA_KM = 6541.4

# Default TOA timing noise (1-sigma), seconds — navigation-grade MSP ~0.1–1 µs.
DEFAULT_TOA_SIGMA_S = 1.0e-6


def elfo_orbital_period_s(sma_km: float = ELFO_SMA_KM) -> float:
    """Two-body orbital period (seconds) for HW2 frozen ELFO semi-major axis."""
    return 2.0 * math.pi * math.sqrt(sma_km**3 / GM_MOON_KM3_S2)


# Recommended Monte Carlo arc: ≥2 full orbits for stable blackout statistics.
DEFAULT_MC_DURATION_S = 2.0 * elfo_orbital_period_s()
