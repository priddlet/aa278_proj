"""Matplotlib plots for lunar orbit propagation verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsar_nav.propagation.dynamics import MOON_RADIUS_KM
from pulsar_nav.propagation.propagator import PropagatedTrajectory

MOON_COLOR = "#888888"
ORBIT_COLOR = "#2563eb"
START_COLOR = "#16a34a"
END_COLOR = "#dc2626"


def _require_matplotlib():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for orbit plots. "
            "Install with: pip install -e '.[viz]'"
        ) from exc
    return plt


def _draw_moon_sphere(ax, radius_km: float = MOON_RADIUS_KM, alpha: float = 0.25) -> None:
    """Wireframe Moon at origin (MCI plots)."""
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 18)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=MOON_COLOR, alpha=alpha, linewidth=0, shade=True)


def plot_propagated_trajectory(
    traj: PropagatedTrajectory,
    *,
    preset: str = "orbit",
    title: str | None = None,
    show_moon: bool = True,
):
    """
    Multi-panel verification figure for a propagated arc.

    Panels: 3D MCI orbit, XY/XZ projections, radius and altitude vs time.
    """
    plt = _require_matplotlib()

    pos = traj.position_mci_km
    t_hr = traj.t_rel_s / 3600.0
    radii = np.linalg.norm(pos, axis=1)
    alt = radii - MOON_RADIUS_KM

    title = title or f"Lunar {preset.upper()} propagation (MCI)"

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(title, fontsize=13)

    # --- 3D MCI ---
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=ORBIT_COLOR, lw=1.2, label="spacecraft")
    ax3d.scatter(*pos[0], color=START_COLOR, s=40, label="start")
    ax3d.scatter(*pos[-1], color=END_COLOR, s=40, label="end")
    if show_moon:
        _draw_moon_sphere(ax3d)
    ax3d.set_xlabel("x (km)")
    ax3d.set_ylabel("y (km)")
    ax3d.set_zlabel("z (km)")
    ax3d.set_title("Moon-centered inertial (J2000)")
    ax3d.legend(loc="upper right", fontsize=8)
    _set_equal_3d(ax3d, pos)

    # --- XY projection ---
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.plot(pos[:, 0], pos[:, 1], color=ORBIT_COLOR, lw=1.0)
    ax_xy.scatter(pos[0, 0], pos[0, 1], color=START_COLOR, s=30, zorder=5)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax_xy.plot(
        MOON_RADIUS_KM * np.cos(theta),
        MOON_RADIUS_KM * np.sin(theta),
        color=MOON_COLOR,
        lw=0.8,
        ls="--",
    )
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x (km)")
    ax_xy.set_ylabel("y (km)")
    ax_xy.set_title("XY projection + lunar limb")
    ax_xy.grid(True, ls=":", alpha=0.5)

    # --- XZ projection ---
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_xz.plot(pos[:, 0], pos[:, 2], color=ORBIT_COLOR, lw=1.0)
    ax_xz.scatter(pos[0, 0], pos[0, 2], color=START_COLOR, s=30, zorder=5)
    ax_xz.plot(
        MOON_RADIUS_KM * np.cos(theta),
        MOON_RADIUS_KM * np.sin(theta),
        color=MOON_COLOR,
        lw=0.8,
        ls="--",
    )
    ax_xz.set_aspect("equal")
    ax_xz.set_xlabel("x (km)")
    ax_xz.set_ylabel("z (km)")
    ax_xz.set_title("XZ projection")
    ax_xz.grid(True, ls=":", alpha=0.5)

    # --- Radius / altitude vs time ---
    ax_rt = fig.add_subplot(2, 2, 4)
    ax_rt.plot(t_hr, radii, color=ORBIT_COLOR, lw=1.0, label=r"$|r|$")
    ax_rt.plot(t_hr, alt, color="#f59e0b", lw=0.9, ls="--", label="altitude")
    ax_rt.axhline(MOON_RADIUS_KM, color=MOON_COLOR, ls=":", lw=0.8, label="Moon radius")
    ax_rt.set_xlabel("time since epoch (hr)")
    ax_rt.set_ylabel("km")
    ax_rt.set_title("Orbital radius check")
    ax_rt.legend(fontsize=8)
    ax_rt.grid(True, ls=":", alpha=0.5)

    fig.tight_layout()
    return fig


def plot_icrs_trajectory(
    traj: PropagatedTrajectory,
    *,
    title: str | None = None,
):
    """3D plot of spacecraft path in ICRS (km) with Moon center track."""
    plt = _require_matplotlib()

    pos_icrs = traj.position_icrs_m / 1000.0
    moon_track = np.array(
        [
            traj.position_icrs_m[i] / 1000.0 - traj.position_mci_km[i]
            for i in range(len(traj.t_rel_s))
        ]
    )

    title = title or "ICRS trajectory (spacecraft + Moon barycenter track)"
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos_icrs[:, 0], pos_icrs[:, 1], pos_icrs[:, 2], color=ORBIT_COLOR, lw=1.0, label="spacecraft")
    ax.plot(
        moon_track[:, 0],
        moon_track[:, 1],
        moon_track[:, 2],
        color=MOON_COLOR,
        lw=0.8,
        ls="--",
        alpha=0.7,
        label="Moon (SSB)",
    )
    ax.scatter(*pos_icrs[0], color=START_COLOR, s=35)
    ax.scatter(*pos_icrs[-1], color=END_COLOR, s=35)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _set_equal_3d(ax, pos_icrs)
    fig.tight_layout()
    return fig


def save_propagation_figure(
    fig,
    path: str | Path,
    *,
    dpi: int = 150,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _set_equal_3d(ax, points: np.ndarray) -> None:
    """Equal axis limits for 3D orbit plots."""
    max_range = np.ptp(points, axis=0).max() / 2.0
    if max_range <= 0:
        max_range = 1.0
    mid = points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
