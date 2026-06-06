"""Orbit and blackout specification helpers for presentation tables."""

from __future__ import annotations

from dataclasses import dataclass

from pulsar_nav.constants import (
    DEFAULT_MC_DURATION_S,
    elfo_orbit_summary,
    elfo_orbital_period_s,
)
from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.simulation.presentation_runs import propagate_truth_arc
from pulsar_nav.visibility.blackout import VisibilityTimeline
from pulsar_nav.visibility.gnss_coverage import gnss_sidelobe_coverage_stats


@dataclass(frozen=True)
class BlackoutSpecRow:
    """Blackout statistics for one simulation arc."""

    label: str
    duration_hr: float
    revolutions: float
    blackout_fraction: float
    n_blackout_windows: int
    mean_blackout_window_hr: float
    gnss_coverage_line: str


def _spec_from_arc(
    label: str,
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    *,
    period_s: float,
) -> BlackoutSpecRow:
    gnss = gnss_sidelobe_coverage_stats(traj, timeline)
    wins = timeline.windows
    mean_win_hr = sum(w.duration_hr for w in wins) / len(wins) if wins else 0.0
    dur_hr = float(traj.t_rel_s[-1] - traj.t_rel_s[0]) / 3600.0
    return BlackoutSpecRow(
        label=label,
        duration_hr=dur_hr,
        revolutions=dur_hr / (period_s / 3600.0),
        blackout_fraction=timeline.blackout_fraction,
        n_blackout_windows=len(wins),
        mean_blackout_window_hr=mean_win_hr,
        gnss_coverage_line=gnss.summary_line(),
    )


def elfo_blackout_specs(
    *,
    preset: str = "elfo",
    epoch_utc: str = "2026-01-15T12:00:00",
    step_s: float = 120.0,
    include_disturbances: bool = False,
) -> tuple[BlackoutSpecRow, BlackoutSpecRow, float]:
    """
    Return (one-orbit spec, MC default arc spec, orbital period in hours).

    HW2 ELFO case 1: T approx 13.2 h at a = 6541.4 km (not LCRNS 30 h orbit).
    """
    period_s = elfo_orbital_period_s()
    one_traj, one_tl = propagate_truth_arc(
        preset=preset,
        epoch_utc=epoch_utc,
        duration_s=period_s,
        step_s=step_s,
        include_disturbances=include_disturbances,
    )
    mc_traj, mc_tl = propagate_truth_arc(
        preset=preset,
        epoch_utc=epoch_utc,
        duration_s=DEFAULT_MC_DURATION_S,
        step_s=step_s,
        include_disturbances=include_disturbances,
    )
    period_hr = period_s / 3600.0
    one = _spec_from_arc(
        f"One orbit (T approx {period_hr:.1f} h)",
        one_traj,
        one_tl,
        period_s=period_s,
    )
    mc = _spec_from_arc(
        f"MC arc ({DEFAULT_MC_DURATION_S / 3600:.1f} h, 2x period)",
        mc_traj,
        mc_tl,
        period_s=period_s,
    )
    return one, mc, period_hr


def blackout_specs_markdown(
    *,
    preset: str = "elfo",
    include_disturbances: bool = False,
) -> str:
    """Markdown table for slides: single-orbit vs Monte Carlo arc blackout."""
    one, mc, _period_hr = elfo_blackout_specs(
        preset=preset,
        include_disturbances=include_disturbances,
    )
    orbit_line = elfo_orbit_summary() if preset == "elfo" else preset.upper()
    lines = [
        f"## Blackout specifications - {preset.upper()}",
        "",
        orbit_line,
        "",
        "_GNSS blackout: Earth below 5 deg elevation mask (geometric far-side bound)._",
        "",
        "| Arc | Duration (hr) | Revolutions | Blackout % | # windows | Mean window (hr) |",
        "|-----|---------------|-------------|------------|-----------|------------------|",
    ]
    for row in (one, mc):
        lines.append(
            f"| {row.label} | {row.duration_hr:.2f} | {row.revolutions:.2f} | "
            f"{100.0 * row.blackout_fraction:.1f} | {row.n_blackout_windows} | "
            f"{row.mean_blackout_window_hr:.2f} |"
        )
    lines.extend(
        [
            "",
            f"**One-orbit blackout:** {100.0 * one.blackout_fraction:.1f}% "
            "(use this for per-revolution geometry, not the 30 hr visibility figure).",
            "",
            f"**{one.gnss_coverage_line}** (one orbit, non-blackout epochs)",
            "",
            f"**{mc.gnss_coverage_line}** (MC arc, non-blackout epochs)",
            "",
        ]
    )
    return "\n".join(lines)
