"""Monte Carlo result plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsar_nav.simulation.monte_carlo import LUNANET_REQUIREMENT_M, MonteCarloResult
from pulsar_nav.simulation.policy import NavPolicy


def _require_matplotlib():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
    import matplotlib.pyplot as plt

    return plt


POLICY_COLORS = {
    NavPolicy.HYBRID: "#3b82f6",
    NavPolicy.XNAV_ONLY: "#ef4444",
    NavPolicy.GNSS_ONLY: "#22c55e",
    NavPolicy.LONET_ONLY: "#a855f7",
}


def plot_final_error_boxplot(result: MonteCarloResult, *, title: str | None = None):
    """Box plot of final position error by policy."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))

    policies = list(result.config.policies)
    data = []
    labels = []
    for pol in policies:
        errs = [
            t.final_error_m / 1e3
            for t in result.trials
            if t.policy == pol
        ]
        data.append(errs)
        labels.append(pol.value)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, pol in zip(bp["boxes"], policies):
        patch.set_facecolor(POLICY_COLORS.get(pol, "#999"))

    ax.axhline(
        LUNANET_REQUIREMENT_M / 1e3,
        color="#f59e0b",
        ls="--",
        lw=1.0,
        label=f"LunaNet ref ({LUNANET_REQUIREMENT_M:.2f} m)",
    )
    ax.set_ylabel("final position error (km)")
    ax.set_title(title or "Monte Carlo — final position error")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_pulsar_count_sweep(
    sweep: dict[int, MonteCarloResult],
    *,
    policy: NavPolicy = NavPolicy.HYBRID,
    title: str | None = None,
):
    """Mean final error vs pulsar count."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))

    counts = sorted(sweep.keys())
    means = [sweep[n].by_policy[policy].final_mean_m / 1e3 for n in counts]
    p95s = [sweep[n].by_policy[policy].final_p95_m / 1e3 for n in counts]

    ax.plot(counts, means, "o-", label="mean final error", color=POLICY_COLORS.get(policy, "#3b82f6"))
    ax.plot(counts, p95s, "s--", label="p95 final error", color="#64748b")
    ax.axhline(LUNANET_REQUIREMENT_M / 1e3, color="#f59e0b", ls=":", label="LunaNet 13.43 m")
    ax.set_xlabel("number of pulsars")
    ax.set_ylabel("error (km)")
    ax.set_title(title or f"Pulsar count sweep — {policy.value}")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
