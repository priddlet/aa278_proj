"""Plots for hybrid navigation filter runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsar_nav.simulation.hybrid_run import HybridRunResult
from pulsar_nav.simulation.policy import SEGMENT_COLORS
from pulsar_nav.visibility.blackout import VisibilityTimeline


def _require_matplotlib():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_segment_strip(ax, t_hr: np.ndarray, segment_values: np.ndarray) -> None:
    """Bottom strip colored by filter segment actually applied (or planned)."""
    unique = list(dict.fromkeys(segment_values))
    seg_to_y = {s: i for i, s in enumerate(unique)}
    y = np.array([seg_to_y[s] for s in segment_values])
    for seg in unique:
        mask = segment_values == seg
        ax.scatter(
            t_hr[mask],
            y[mask],
            c=SEGMENT_COLORS.get(seg, "#999"),
            s=14,
            label=seg,
        )
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(unique, fontsize=7)
    ax.set_ylim(-0.5, len(unique) - 0.5)


def plot_hybrid_comparison(
    hybrid: HybridRunResult,
    xnav_only: HybridRunResult,
    timeline: VisibilityTimeline,
    *,
    title: str = "Hybrid vs XNAV-only",
    use_measured_segments: bool = True,
):
    """
    Error time series plus strip of **filter segments applied** (not geometric NavMode).

    Purple in the geometric timeline only meant relay geometry; it did not imply
    LunaNet measurements during blackout before the hybrid policy fix.
    """
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

    if use_measured_segments and len(hybrid.policy_segments) == len(t_hr):
        seg_vals = hybrid.policy_segments
        strip_label = "filter segment applied (hybrid run)"
    else:
        from pulsar_nav.simulation.policy import planned_segment

        seg_vals = np.array(
            [planned_segment(hybrid.policy, s).value for s in timeline.samples]
        )
        strip_label = "planned segment (hybrid policy)"

    _plot_segment_strip(axes[1], t_hr, seg_vals)
    axes[1].set_xlabel("time since epoch (hr)")
    axes[1].set_ylabel(strip_label)
    axes[1].legend(loc="upper right", fontsize=7, ncol=1)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
