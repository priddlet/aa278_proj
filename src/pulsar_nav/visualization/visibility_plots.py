"""Plots for GNSS blackout and LunaNet visibility along an orbit."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.simulation.policy import (
    SEGMENT_COLORS,
    NavPolicy,
    planned_segment,
)
from pulsar_nav.visibility.blackout import (
    NAV_MODE_DISPLAY,
    NavMode,
    VisibilityTimeline,
    timeline_arrays,
)
from pulsar_nav.visualization.presentation_style import (
    policy_display_name,
    segment_plot_label,
)

MODE_COLORS = {
    NavMode.GNSS.value: "#22c55e",
    NavMode.HYBRID.value: "#3b82f6",
    NavMode.LONET.value: "#a855f7",
    NavMode.XNAV.value: "#ef4444",
    NavMode.NONE.value: "#6b7280",
}


def _mode_label(mode_value: str) -> str:
    try:
        return NAV_MODE_DISPLAY[NavMode(mode_value)]
    except ValueError:
        return mode_value


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
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, ls=":", alpha=0.5)

    # LunaNet + GNSS flags
    ax1 = axes[1]
    ax1.fill_between(t_hr, 0, 1, where=arr["gnss_visible"], alpha=0.35, color="#22c55e", label="GNSS")
    ax1.fill_between(t_hr, 0, 0.5, where=arr["lonet_visible"], alpha=0.35, color="#a855f7", label="LunaNet")
    ax1.set_ylim(-0.05, 1.15)
    ax1.set_ylabel("visible")
    ax1.legend(loc="upper right", fontsize=9)
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
    ax2.set_yticklabels([_mode_label(m) for m in unique])
    ax2.set_xlabel("time since epoch (hr)")
    ax2.legend(loc="upper right", fontsize=9, ncol=min(len(unique), 3))
    ax2.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_orbit_colored_by_blackout(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    *,
    title: str = "ELFO orbit — GNSS blackout segments",
):
    """3D MCI orbit: red = far-side blackout, blue = GNSS-visible."""
    plt = _require_matplotlib()
    from pulsar_nav.visualization.orbit_plots import _draw_moon_sphere, _set_equal_3d

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    pos = traj.position_mci_km
    in_blk = np.array([s.in_blackout for s in timeline.samples])

    ax.plot(
        pos[~in_blk, 0],
        pos[~in_blk, 1],
        pos[~in_blk, 2],
        ".",
        ms=3,
        color="#22c55e",
        label="GNSS visible",
    )
    ax.plot(
        pos[in_blk, 0],
        pos[in_blk, 1],
        pos[in_blk, 2],
        ".",
        ms=3,
        color="#ef4444",
        label="GNSS blackout",
    )
    _draw_moon_sphere(ax, alpha=0.2)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    _set_equal_3d(ax, pos)
    fig.tight_layout()
    return fig


def plot_orbit_blackout_xy(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    *,
    title: str = "XY projection — blackout along track",
):
    """Moon-centered XY view with blackout-colored ground track."""
    plt = _require_matplotlib()
    from pulsar_nav.propagation.dynamics import MOON_RADIUS_KM

    fig, ax = plt.subplots(figsize=(8, 8))
    pos = traj.position_mci_km
    in_blk = np.array([s.in_blackout for s in timeline.samples])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        MOON_RADIUS_KM * np.cos(theta),
        MOON_RADIUS_KM * np.sin(theta),
        color="#888",
        lw=0.8,
        ls="--",
        label="lunar limb",
    )
    ax.scatter(
        pos[~in_blk, 0],
        pos[~in_blk, 1],
        c="#22c55e",
        s=8,
        label="GNSS visible",
    )
    ax.scatter(
        pos[in_blk, 0],
        pos[in_blk, 1],
        c="#ef4444",
        s=8,
        label="GNSS blackout",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_orbit_colored_by_policy(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    policy: NavPolicy,
    *,
    title: str | None = None,
):
    """3D orbit colored by which measurements the filter uses (per ``NavPolicy``)."""
    plt = _require_matplotlib()
    from pulsar_nav.visualization.orbit_plots import _draw_moon_sphere, _set_equal_3d

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    pos = traj.position_mci_km
    segments = [planned_segment(policy, s) for s in timeline.samples]

    for seg in dict.fromkeys(segments):
        mask = np.array([s == seg for s in segments])
        ax.plot(
            pos[mask, 0],
            pos[mask, 1],
            pos[mask, 2],
            ".",
            ms=4,
            color=SEGMENT_COLORS.get(seg.value, "#333"),
            label=segment_plot_label(seg),
        )
    _draw_moon_sphere(ax, alpha=0.2)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title(title or f"ELFO orbit — {policy_display_name(policy)}")
    ax.legend(fontsize=9, loc="upper left")
    _set_equal_3d(ax, pos)
    fig.tight_layout()
    return fig


def plot_policy_segment_timeline(
    timeline: VisibilityTimeline,
    policy: NavPolicy,
    *,
    title: str | None = None,
):
    """Which measurement segment is active vs time for one ``NavPolicy``."""
    plt = _require_matplotlib()
    t_hr = np.array([s.t_s for s in timeline.samples]) / 3600.0
    segments = [planned_segment(policy, s) for s in timeline.samples]

    fig, ax = plt.subplots(figsize=(11, 2.8))
    unique = list(dict.fromkeys(segments))
    seg_to_y = {s: i for i, s in enumerate(unique)}
    y = np.array([seg_to_y[s] for s in segments])
    for seg in unique:
        mask = np.array([s == seg for s in segments])
        ax.scatter(
            t_hr[mask],
            y[mask],
            c=SEGMENT_COLORS.get(seg.value, "#333"),
            s=14,
            label=segment_plot_label(seg),
        )
    _shade_blackout(ax, t_hr, [s.in_blackout for s in timeline.samples], alpha=0.08)
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels([segment_plot_label(s) for s in unique], fontsize=9)
    ax.set_xlabel("time since epoch (hr)")
    ax.set_title(title or f"Planned segments — {policy_display_name(policy)}")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_orbit_colored_by_mode(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    *,
    title: str = "ELFO orbit — geometric source availability",
):
    """3D Moon-centered orbit colored by visibility (GNSS / relay / blackout)."""
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
            label=_mode_label(mode),
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
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
