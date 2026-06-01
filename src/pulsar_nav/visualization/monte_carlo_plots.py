"""Monte Carlo result plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pulsar_nav.simulation.hybrid_run import HybridRunResult
from pulsar_nav.simulation.monte_carlo import LUNANET_REQUIREMENT_M, MonteCarloResult
from pulsar_nav.simulation.policy import SEGMENT_COLORS, NavPolicy, planned_segment
from pulsar_nav.simulation.presentation_runs import PolicyErrorEnvelope
from pulsar_nav.visibility.blackout import VisibilityTimeline
from pulsar_nav.visualization.presentation_style import policy_display_name


def _require_matplotlib():
    import os

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"), force=True)
    import matplotlib.pyplot as plt

    return plt


POLICY_COLORS = {
    NavPolicy.XNAV_ONLY: "#ef4444",
    NavPolicy.GNSS_ONLY: "#22c55e",
    NavPolicy.HYBRID: "#3b82f6",
    NavPolicy.GNSS_COAST: "#6b7280",
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
        labels.append(policy_display_name(pol))

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, pol in zip(bp["boxes"], policies):
        patch.set_facecolor(POLICY_COLORS.get(pol, "#999"))

    ax.axhline(
        LUNANET_REQUIREMENT_M / 1e3,
        color="#f59e0b",
        ls="--",
        lw=1.0,
        label="LunaNet 13.43 m",
    )
    ax.set_ylabel("final position error (km)")
    ax.set_title(title or "Final position error")
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
    ax.set_title(title or f"Pulsar count — {policy_display_name(policy)}")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_policy_metrics_bars(result: MonteCarloResult, *, title: str | None = None):
    """Grouped bar chart: final mean, blackout mean, non-blackout mean by policy."""
    plt = _require_matplotlib()
    policies = list(result.config.policies)
    x = np.arange(len(policies))
    width = 0.25

    finals = [result.by_policy[p].final_mean_m / 1e3 for p in policies]
    blk = [result.by_policy[p].blackout_mean_m / 1e3 for p in policies]
    non_blk = [result.by_policy[p].non_blackout_mean_m / 1e3 for p in policies]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, finals, width, label="final mean", color="#3b82f6")
    ax.bar(x, blk, width, label="blackout segment μ", color="#ef4444")
    ax.bar(x + width, non_blk, width, label="non-blackout μ", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels([policy_display_name(p) for p in policies])
    ax.set_ylabel("position error (km)")
    ax.set_title(title or "Mean error by segment")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_toa_noise_sweep(
    sweep: dict[float, MonteCarloResult],
    *,
    policies: tuple[NavPolicy, ...] | None = None,
    title: str | None = None,
):
    """Final mean error vs TOA σ for each policy."""
    plt = _require_matplotlib()
    sigmas_us = sorted(sweep.keys())
    policies = policies or (
        NavPolicy.XNAV_ONLY,
        NavPolicy.GNSS_ONLY,
        NavPolicy.HYBRID,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for pol in policies:
        means = [sweep[s].by_policy[pol].final_mean_m / 1e3 for s in sigmas_us]
        ax.plot(
            sigmas_us,
            means,
            "o-",
            label=policy_display_name(pol),
            color=POLICY_COLORS.get(pol, None),
            lw=1.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("TOA 1σ (µs)")
    ax.set_ylabel("final position error mean (km)")
    ax.set_title(title or "TOA noise sweep")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_pulsar_sweep_comparison(
    sweep: dict[int, MonteCarloResult],
    *,
    title: str | None = None,
):
    """Hybrid vs XNAV-only final mean vs pulsar count."""
    plt = _require_matplotlib()
    counts = sorted(sweep.keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pol, marker in ((NavPolicy.HYBRID, "o-"), (NavPolicy.XNAV_ONLY, "s--")):
        if pol not in sweep[counts[0]].config.policies:
            continue
        means = [sweep[n].by_policy[pol].final_mean_m / 1e3 for n in counts]
        ax.plot(
            counts,
            means,
            marker,
            label=policy_display_name(pol),
            color=POLICY_COLORS.get(pol, "#333"),
            lw=1.5,
        )
    ax.set_xlabel("number of MSPs")
    ax.set_ylabel("final mean error (km)")
    ax.set_title(title or "Pulsar count sweep")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def _shade_blackout(ax, t_hr: np.ndarray, timeline: VisibilityTimeline, *, alpha: float = 0.15) -> None:
    from pulsar_nav.visualization.visibility_plots import _shade_blackout as shade

    in_blk = np.array([s.in_blackout for s in timeline.samples])
    shade(ax, t_hr, in_blk, alpha=alpha)


def _shade_policy_segments(
    ax,
    t_hr: np.ndarray,
    timeline: VisibilityTimeline,
    policy: NavPolicy,
    run: HybridRunResult | None = None,
) -> None:
    """Background tint by filter segment (measured if ``run`` provided)."""
    if run is not None and len(run.policy_segments) == len(timeline.samples):
        segments = list(run.policy_segments)
    else:
        segments = [planned_segment(policy, s).value for s in timeline.samples]
    start_i = 0
    for i in range(1, len(segments) + 1):
        if i == len(segments) or segments[i] != segments[start_i]:
            seg = segments[start_i]
            key = seg.value if hasattr(seg, "value") else str(seg)
            color = SEGMENT_COLORS.get(key, "#ccc")
            ax.axvspan(t_hr[start_i], t_hr[i - 1], color=color, alpha=0.12)
            start_i = i


def _blackout_legend_handle(plt):
    from matplotlib.patches import Patch

    return Patch(facecolor="#ef4444", alpha=0.25, label="GNSS blackout")


def plot_policy_error_propagation(
    run: HybridRunResult,
    timeline: VisibilityTimeline,
    *,
    policy: NavPolicy,
    offset_km: float | None = None,
    title: str | None = None,
):
    """Position error vs time for one policy with GNSS blackout shading."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t_hr = run.t_s / 3600.0
    color = POLICY_COLORS.get(policy, "#333")
    ax.plot(t_hr, run.position_error_m / 1e3, lw=1.3, color=color)
    _shade_policy_segments(ax, t_hr, timeline, policy, run=run)
    ax.set_xlabel("time since epoch (hr)")
    ax.set_ylabel("position error (km)")
    ax.set_title(title or f"{policy_display_name(policy)} — position error")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=color, lw=1.3, label=policy_display_name(policy)),
            _blackout_legend_handle(plt),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    return fig


def plot_all_policies_propagation(
    runs: dict[NavPolicy, HybridRunResult],
    timeline: VisibilityTimeline,
    *,
    title: str | None = "Policy comparison — position error",
    offset_km: float | None = None,
):
    """Overlay position error for hybrid, XNAV-only, and GNSS-only."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 5))
    t_hr = next(iter(runs.values())).t_s / 3600.0
    for policy, run in runs.items():
        t_pol = run.t_s / 3600.0
        ax.plot(
            t_pol,
            run.position_error_m / 1e3,
            lw=1.2,
            label=policy_display_name(policy),
            color=POLICY_COLORS.get(policy, "#333"),
        )
    _shade_policy_segments(ax, t_hr, timeline, NavPolicy.GNSS_ONLY)
    ax.set_xlabel("time since epoch (hr)")
    ax.set_ylabel("position error (km)")
    ax.set_title(title or "Policy comparison — position error")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [_blackout_legend_handle(plt)], loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_policy_error_envelope(
    envelope: PolicyErrorEnvelope,
    timeline: VisibilityTimeline,
    *,
    title: str | None = None,
    ymin_km: float | None = None,
):
    """
    Mean error with p5–p95 band across Monte Carlo trials at each epoch.

    Note: a wide band (e.g. 0–70 km) at one time means *trials disagree then*,
    not that a single run swings 0–70 km. After policy switch into blackout,
    mean often drops to ~0.2–0.7 km (XNAV segment), which looks like “zero” on
    GNSS-visible epochs; ``gnss_coast`` can reach 10–150 km in blackout (no updates).
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    t_hr = envelope.t_s / 3600.0
    color = POLICY_COLORS.get(envelope.policy, "#333")
    mean_km = envelope.mean_m / 1e3
    p5_km = envelope.p5_m / 1e3
    p95_km = envelope.p95_m / 1e3
    _shade_policy_segments(ax, t_hr, timeline, envelope.policy)
    ax.fill_between(t_hr, p5_km, p95_km, color=color, alpha=0.2)
    ax.plot(t_hr, mean_km, lw=1.4, color=color)
    ax.set_xlabel("time since epoch (hr)")
    ax.set_ylabel("position error (km)")
    ax.set_title(
        title or f"{policy_display_name(envelope.policy)} — Monte Carlo mean"
    )
    ymax = float(np.nanmax(p95_km) * 1.05)
    if ymin_km is None:
        # Avoid auto-scale 0–90 km when post-convergence mean is sub-km
        post = mean_km[int(len(mean_km) * 0.08) :]
        floor = max(0.0, float(np.nanpercentile(post, 5)) * 0.5) if post.size else 0.0
        ymin_km = min(floor, ymax * 0.05)
    ax.set_ylim(bottom=ymin_km, top=max(ymax, ymin_km + 0.5))
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=color, lw=1.4, label="mean"),
            plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.2, label="p5–p95"),
            _blackout_legend_handle(plt),
        ],
        loc="upper right",
    )
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_all_policies_envelope(
    envelopes: dict[NavPolicy, PolicyErrorEnvelope],
    timeline: VisibilityTimeline,
    *,
    title: str | None = "Monte Carlo mean — all policies",
):
    """Compare mean error envelopes on one axes."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 5))
    t_hr = next(iter(envelopes.values())).t_s / 3600.0
    n_trials = next(iter(envelopes.values())).n_trials
    for policy, env in envelopes.items():
        ax.plot(
            env.t_s / 3600.0,
            env.mean_m / 1e3,
            lw=1.3,
            label=policy_display_name(policy),
            color=POLICY_COLORS.get(policy, "#333"),
        )
    ax.set_xlabel("time since epoch (hr)")
    ax.set_ylabel("mean position error (km)")
    p95_all = [env.p95_m / 1e3 for env in envelopes.values()]
    ymax = max(float(np.nanmax(p95_all)) * 1.05, 1.0)
    ax.set_ylim(bottom=0.0, top=ymax)
    ax.set_title(title or "Monte Carlo mean — all policies")
    ax.legend(loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_final_error_cdf(result: MonteCarloResult, *, title: str | None = None):
    """Empirical CDF of final position error by policy."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    for pol in result.config.policies:
        errs = sorted(
            t.final_error_m / 1e3 for t in result.trials if t.policy == pol
        )
        if not errs:
            continue
        y = np.linspace(1.0 / len(errs), 1.0, len(errs))
        ax.plot(errs, y, lw=1.5, label=policy_display_name(pol), color=POLICY_COLORS.get(pol, "#333"))
    ax.set_xlabel("final position error (km)")
    ax.set_ylabel("empirical CDF")
    ax.set_title(title or "Final error CDF")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_trial_final_scatter(result: MonteCarloResult, *, title: str | None = None):
    """Per-trial final errors: hybrid vs XNAV vs GNSS."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 7))
    by_trial: dict[int, dict[NavPolicy, float]] = {}
    for t in result.trials:
        by_trial.setdefault(t.trial_id, {})[t.policy] = t.final_error_m / 1e3

    hybrid = NavPolicy.HYBRID
    xnav = NavPolicy.XNAV_ONLY
    gnss = NavPolicy.GNSS_ONLY
    if hybrid in result.config.policies and xnav in result.config.policies:
        hx = [by_trial[i][hybrid] for i in sorted(by_trial) if hybrid in by_trial[i] and xnav in by_trial[i]]
        xy = [by_trial[i][xnav] for i in sorted(by_trial) if hybrid in by_trial[i] and xnav in by_trial[i]]
        ax.scatter(
            xy,
            hx,
            c=POLICY_COLORS[hybrid],
            s=40,
            alpha=0.75,
            label=f"{policy_display_name(hybrid)} vs {policy_display_name(xnav)}",
        )
        lim = max(max(hx + xy, default=1), 1)
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
        ax.set_xlabel("XNAV-only final error (km)")
        ax.set_ylabel("hybrid final error (km)")
    ax.set_title(title or "Per-trial final errors")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig


def apply_presentation_style() -> None:
    """Larger fonts and line weights for slide export."""
    from pulsar_nav.visualization.presentation_style import apply_presentation_style as _apply

    _apply()


def save_figure(fig, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt = _require_matplotlib()
    plt.close(fig)
    return path
