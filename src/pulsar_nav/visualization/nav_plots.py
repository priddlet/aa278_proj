"""Plots for XNAV filter performance vs propagated truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsar_nav.simulation.xnav_run import XNAVRunResult


def _require_matplotlib():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_xnav_errors(
    results: dict[str, XNAVRunResult],
    *,
    title: str = "XNAV EKF vs propagated truth (ICRS)",
):
    """Overlay position error norms for one or more filter runs."""
    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=12)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))

    for (label, res), color in zip(results.items(), colors):
        t_hr = res.t_s / 3600.0
        axes[0].plot(t_hr, res.position_error_m / 1e3, lw=1.2, label=label, color=color)
        if res.los_error_m is not None:
            axes[1].plot(
                t_hr,
                res.los_error_m / 1e3,
                lw=1.0,
                ls="--",
                label=f"{label} (LOS only)",
                color=color,
            )

    axes[0].set_ylabel("position error (km)")
    axes[0].set_title("3D position error ||r_est - r_true||")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, ls=":", alpha=0.5)

    axes[1].set_ylabel("LOS error (km)")
    axes[1].set_xlabel("time since epoch (hr)")
    if any(r.los_error_m is not None for r in results.values()):
        axes[1].set_title("Along-LOS error (single-pulsar runs)")
        axes[1].legend(fontsize=9)
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    return fig


def plot_xyz_errors(result: XNAVRunResult, *, title: str | None = None):
    """Per-axis position errors for one run."""
    plt = _require_matplotlib()
    err = result.est_position_m - result.truth_position_m
    t_hr = result.t_s / 3600.0

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, axis in enumerate("XYZ"):
        ax.plot(t_hr, err[:, i] / 1e3, lw=1.0, label=axis)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("time (hr)")
    ax.set_ylabel("error (km)")
    ax.set_title(title or f"Position errors — {', '.join(result.pulsar_names)}")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
