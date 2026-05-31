"""Label GNSS blackout and navigation mode along a propagated trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.spice.ephemeris import body_position_mci
from pulsar_nav.visibility.gnss import earth_elevation_deg, gnss_earth_visible
from pulsar_nav.visibility.lonet import (
    LunaNetConfig,
    construct_walker_constellation,
    lonet_visibility,
    propagate_constellation_mci,
)


class NavMode(str, Enum):
    """
    Which radio sources are geometrically available at one epoch.

    This is **not** a Monte Carlo ``NavPolicy`` (``hybrid``, ``gnss_xnav``, etc.).
    It labels visibility for timelines and orbit plots only.
    """

    GNSS = "gnss"  # GNSS sidelobe OK, no relay in view
    HYBRID = "hybrid"  # GNSS + at least one LunaNet relay
    LONET = "lonet"  # Relay visible, Earth below GNSS mask (uncommon on ELFO)
    XNAV = "xnav"  # Far-side blackout: no GNSS (pulsars always an option)
    NONE = "none"


# Human-readable orbit/timeline legend (avoid confusion with NavPolicy.GNSS_ONLY).
NAV_MODE_DISPLAY: dict[NavMode, str] = {
    NavMode.HYBRID: "GNSS + LunaNet visible",
    NavMode.GNSS: "GNSS visible",
    NavMode.LONET: "LunaNet only (no GNSS mask)",
    NavMode.XNAV: "GNSS blackout",
    NavMode.NONE: "none",
}


@dataclass
class VisibilitySample:
    t_s: float
    earth_elevation_deg: float
    gnss_visible: bool
    lonet_visible: bool
    n_lonet_visible: int
    max_lonet_elevation_deg: float
    nav_mode: NavMode
    in_blackout: bool


@dataclass
class BlackoutWindow:
    start_s: float
    end_s: float
    duration_s: float

    @property
    def duration_hr(self) -> float:
        return self.duration_s / 3600.0


@dataclass
class VisibilityTimeline:
    samples: list[VisibilitySample]
    windows: list[BlackoutWindow]

    @property
    def blackout_fraction(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean([s.in_blackout for s in self.samples]))

    @property
    def total_blackout_s(self) -> float:
        return float(sum(w.duration_s for w in self.windows))


def _nav_mode(gnss_visible: bool, lonet_visible: bool) -> NavMode:
    if gnss_visible and lonet_visible:
        return NavMode.HYBRID
    if gnss_visible:
        return NavMode.GNSS
    if lonet_visible:
        return NavMode.LONET
    return NavMode.XNAV


def compute_visibility_timeline(
    traj: PropagatedTrajectory,
    *,
    lonet_config: LunaNetConfig | None = None,
    min_gnss_elevation_deg: float = 5.0,
) -> VisibilityTimeline:
    """
    Evaluate GNSS (Earth elevation) and LunaNet visibility at each trajectory sample.
    """
    cfg = lonet_config or LunaNetConfig()
    coes = construct_walker_constellation(
        cfg.sma_km,
        cfg.eccentricity,
        cfg.inclination_rad,
        cfg.argp_rad,
        cfg.walker_f,
        cfg.n_sats,
        cfg.n_planes,
    )
    relay_pos = propagate_constellation_mci(coes, traj.t_rel_s)

    samples: list[VisibilitySample] = []
    for i, t in enumerate(traj.t_rel_s):
        r_sc = traj.position_mci_km[i]
        et = traj.et[i]
        r_earth = body_position_mci("EARTH", et)

        elev_deg = earth_elevation_deg(r_sc, r_earth)
        gnss_vis = gnss_earth_visible(
            r_sc, r_earth, min_elevation_deg=min_gnss_elevation_deg
        )
        lonet_vis, n_lon, max_lon_el = lonet_visibility(
            r_sc,
            relay_pos[:, i, :],
            min_elevation_deg=cfg.min_elevation_deg,
        )
        mode = _nav_mode(gnss_vis, lonet_vis)
        in_blackout = not gnss_vis  # pitch: far-side blackout = no GNSS sidelobes

        samples.append(
            VisibilitySample(
                t_s=float(t),
                earth_elevation_deg=elev_deg,
                gnss_visible=gnss_vis,
                lonet_visible=lonet_vis,
                n_lonet_visible=n_lon,
                max_lonet_elevation_deg=max_lon_el,
                nav_mode=mode,
                in_blackout=in_blackout,
            )
        )

    windows = find_blackout_windows(samples)
    return VisibilityTimeline(samples=samples, windows=windows)


def find_blackout_windows(
    samples: list[VisibilitySample],
    *,
    min_duration_s: float = 0.0,
) -> list[BlackoutWindow]:
    """Contiguous intervals where ``in_blackout`` is True."""
    windows: list[BlackoutWindow] = []
    if not samples:
        return windows

    start: float | None = None
    for i, s in enumerate(samples):
        if s.in_blackout and start is None:
            start = s.t_s
        end_blackout = s.in_blackout and (
            i == len(samples) - 1 or not samples[i + 1].in_blackout
        )
        if end_blackout and start is not None:
            end_t = s.t_s
            dur = end_t - start
            if dur >= min_duration_s:
                windows.append(BlackoutWindow(start_s=start, end_s=end_t, duration_s=dur))
            start = None
    return windows


def timeline_arrays(timeline: VisibilityTimeline) -> dict[str, np.ndarray]:
    """Convert timeline to numpy arrays for plotting."""
    n = len(timeline.samples)
    return {
        "t_s": np.array([s.t_s for s in timeline.samples]),
        "earth_elevation_deg": np.array([s.earth_elevation_deg for s in timeline.samples]),
        "gnss_visible": np.array([s.gnss_visible for s in timeline.samples], dtype=bool),
        "lonet_visible": np.array([s.lonet_visible for s in timeline.samples], dtype=bool),
        "in_blackout": np.array([s.in_blackout for s in timeline.samples], dtype=bool),
        "nav_mode": np.array([s.nav_mode.value for s in timeline.samples]),
    }
