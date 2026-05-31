"""Plots for GNSS blackout and LunaNet visibility along an orbit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.visibility.blackout import VisibilityTimeline, timeline_arrays

MODE_COLORS = {
    "gnss": "#22c55e",
    "hybrid": "#3b82f6",
    "lonet": "#a855f7",
    "xnav": "#ef4444",
    "none": "#6b7280",
}


def _require_matplotlib():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_visibility_timeline(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    *,
    title: str = "Visibility along lunar orbit",
):
    """Earth elevation, nav mode, and blackout windows vs time."""
    plt = _require_matplotlib()
    arr = timeline_arrays(timeline)
    t_hr = arr["t_s"] / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(title, fontsize=12)

    # Earth elevation + blackout shading
    ax0 = axes[0]
    ax0.plot(t_hr, arr["earth_elevation_deg"], color="#2563eb", lw=1.0)
    ax0.axhline(0.0, color="k", lw=0.6, ls=":")
    ax0.axhline(5.0, color="#f59e0b", lw=0.8, ls="--", label="GNSS mask (5°)")
    _shade_blackout(ax0, t_hr, arr["in_blackout"])
    ax0.set_ylabel("Earth elev. (deg)")
    ax0.set_title("Earth elevation from spacecraft (GNSS sidelobe proxy)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, ls=":", alpha=0.5)

    # LunaNet + GNSS flags
    ax1 = axes[1]
    ax1.fill_between(t_hr, 0, 1, where=arr["gnss_visible"], alpha=0.35, color="#22c55e", label="GNSS")
    ax1.fill_between(t_hr, 0, 0.5, where=arr["lonet_visible"], alpha=0.35, color="#a855f7", label="LunaNet")
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_ylabel("visible")
    ax1.set_title("Service availability flags")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, ls=":", alpha=0.5)

    # Nav mode (numeric for color strip)
    ax2 = axes[2]
    modes = arr["nav_mode"]
    unique = list(dict.fromkeys(modes))
    mode_to_y = {m: i for i, m in enumerate(unique)}
    y = np.array([mode_to_y[m] for m in modes])
    for m in unique:
        mask = modes == m
        ax2.scatter(
            t_hr[mask],
            y[mask],
            c=MODE_COLORS.get(m, "#333"),
            s=12,
            label=m,
        )
    _shade_blackout(ax2, t_hr, arr["in_blackout"], alpha=0.12)
    ax2.set_yticks(range(len(unique)))
    ax2.set_yticklabels(unique)
    ax2.set_xlabel("time since epoch (hr)")
    ax2.set_title("Navigation mode (pitch: XNAV during blackout)")
    ax2.legend(loc="upper right", fontsize=8, ncol=len(unique))
    ax2.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_orbit_colored_by_mode(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    *,
    title: str = "MCI orbit colored by nav mode",
):
    """3D Moon-centered orbit with color by navigation mode."""
    plt = _require_matplotlib()
    from pulsar_nav.visualization.orbit_plots import _draw_moon_sphere, _set_equal_3d

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    pos = traj.position_mci_km
    modes = np.array([s.nav_mode.value for s in timeline.samples])

    for mode in dict.fromkeys(modes):
        mask = modes == mode
        ax.plot(
            pos[mask, 0],
            pos[mask, 1],
            pos[mask, 2],
            ".",
            ms=4,
            color=MODE_COLORS.get(mode, "#333"),
            label=mode,
        )
    _draw_moon_sphere(ax, alpha=0.2)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _set_equal_3d(ax, pos)
    fig.tight_layout()
    return fig


def _shade_blackout(ax, t_hr, in_blackout, alpha=0.2) -> None:
    if not np.any(in_blackout):
        return
    in_blk = np.asarray(in_blackout, bool)
    start = None
    for i, flag in enumerate(in_blk):
        if flag and start is None:
            start = t_hr[i]
        if (not flag or i == len(in_blk) - 1) and start is not None:
            end = t_hr[i] if not flag else t_hr[i]
            ax.axvspan(start, end, color="#ef4444", alpha=alpha)
            start = None


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
