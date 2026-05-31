"""Ground-truth trajectory generators for XNAV simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar_nav.spice.ephemeris import str_to_et


@dataclass
class TrajectorySample:
    t_s: float
    position_m: np.ndarray
    velocity_m_s: np.ndarray


def constant_velocity_trajectory(
    t_s: np.ndarray,
    position0_m: np.ndarray,
    velocity_m_s: np.ndarray,
) -> list[TrajectorySample]:
    samples = []
    for t in t_s:
        pos = position0_m + velocity_m_s * t
        samples.append(
            TrajectorySample(t_s=float(t), position_m=pos, velocity_m_s=velocity_m_s.copy())
        )
    return samples


def propagate_lunar_truth(
    preset: str = "elfo",
    *,
    epoch_utc: str = "2026-01-15T12:00:00",
    duration_s: float = 6 * 3600.0,
    step_s: float = 60.0,
    include_srp: bool = False,
    include_moon_j2: bool = False,
    kernel_dir: str | None = None,
):
    """
    Generate truth trajectory via LunarPropagator + SPICE.

    preset: 'elfo' (elliptical frozen) or 'llo' (low lunar orbit).
    """
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator

    et0 = str_to_et(epoch_utc)
    config = DynamicsConfig(
        include_earth=True,
        include_sun=True,
        include_srp=include_srp,
        include_moon_j2=include_moon_j2,
    )
    prop = LunarPropagator(et0, config=config, kernel_dir=kernel_dir)
    return prop.propagate_preset(preset, duration_s=duration_s, step_s=step_s)


def lunar_elo_like_state() -> tuple[np.ndarray, np.ndarray]:
    """
    Representative far-side lunar orbiter state in ICRS (m, m/s).

    Uses a short ELFO propagation when SPICE kernels are available;
    otherwise returns the previous placeholder.
    """
    try:
        traj = propagate_lunar_truth(
            preset="elfo", duration_s=0.0, step_s=60.0
        )
        idx = 0
        return traj.position_icrs_m[idx].copy(), traj.velocity_icrs_m_s[idx].copy()
    except FileNotFoundError:
        position_m = np.array([-1.8e8, 0.4e8, -0.2e8])
        velocity_m_s = np.array([120.0, -80.0, 15.0])
        return position_m, velocity_m_s
