"""Plots for hybrid navigation filter runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsar_nav.simulation.hybrid_run import HybridRunResult
from pulsar_nav.visibility.blackout import VisibilityTimeline


def _require_matplotlib():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
    import matplotlib.pyplot as plt

    return plt


MODE_COLORS = {
    "gnss": "#22c55e",
    "hybrid": "#3b82f6",
    "lonet": "#a855f7",
    "xnav": "#ef4444",
    "none": "#6b7280",
}


def plot_hybrid_comparison(
    hybrid: HybridRunResult,
    xnav_only: HybridRunResult,
    timeline: VisibilityTimeline,
    *,
    title: str = "Hybrid vs XNAV-only",
):
    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    fig.suptitle(title, fontsize=12)

    t_hr = hybrid.t_s / 3600.0
    axes[0].plot(t_hr, hybrid.position_error_m / 1e3, lw=1.2, label="hybrid", color="#3b82f6")
    if hasattr(xnav_only, "position_error_m"):
        t_x = getattr(xnav_only, "t_s", t_hr * 3600.0) / 3600.0
        axes[0].plot(
            t_x,
            xnav_only.position_error_m / 1e3,
            lw=1.0,
            ls="--",
            label="XNAV-only",
            color="#ef4444",
        )
    axes[0].set_ylabel("position error (km)")
    axes[0].legend()
    axes[0].grid(True, ls=":", alpha=0.5)

    # Nav mode strip
    modes = np.array([s.nav_mode.value for s in timeline.samples])
    for mode in np.unique(modes):
        mask = modes == mode
        axes[1].fill_between(
            t_hr,
            0,
            1,
            where=mask,
            alpha=0.45,
            color=MODE_COLORS.get(mode, "#999"),
            label=mode,
        )
    axes[1].set_ylim(0, 1.05)
    axes[1].set_yticks([])
    axes[1].set_xlabel("time since epoch (hr)")
    axes[1].set_ylabel("nav mode")
    axes[1].legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
