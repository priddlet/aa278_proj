"""GNSS geometric window vs sidelobe trackability along a trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar_nav.measurements.gnss_meas import visible_gps_prns
from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.simulation.policy import MIN_TRACKABLE_GNSS_PRNS
from pulsar_nav.visibility.blackout import VisibilityTimeline


@dataclass
class GnssCoverageStats:
    """Non-blackout epochs: geometric GNSS window vs actual sidelobe PRN count."""

    non_blackout_epochs: int
    with_any_prn: int
    with_min_trackable_prns: int
    prn_counts: list[int]

    @property
    def fraction_any_prn(self) -> float:
        if self.non_blackout_epochs == 0:
            return float("nan")
        return self.with_any_prn / self.non_blackout_epochs

    @property
    def fraction_trackable(self) -> float:
        if self.non_blackout_epochs == 0:
            return float("nan")
        return self.with_min_trackable_prns / self.non_blackout_epochs

    def summary_line(self, *, min_prns: int = MIN_TRACKABLE_GNSS_PRNS) -> str:
        return (
            f"Non-blackout epochs: {self.non_blackout_epochs} · "
            f"≥1 PRN: {100.0 * self.fraction_any_prn:.1f}% · "
            f"≥{min_prns} PRNs: {100.0 * self.fraction_trackable:.1f}%"
        )


def gnss_sidelobe_coverage_stats(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    *,
    min_trackable_prns: int = MIN_TRACKABLE_GNSS_PRNS,
) -> GnssCoverageStats:
    """
    Compare loose blackout geometry (``not in_blackout``) to sidelobe PRN counts.

    Green "GNSS visible" bands on geometry plots can overstate epochs where
    ``gnss_pseudoranges`` returns an empty set.
    """
    counts: list[int] = []
    for i, sample in enumerate(timeline.samples):
        if sample.in_blackout:
            continue
        r_sc = traj.position_mci_km[i]
        n = len(
            visible_gps_prns(
                r_sc,
                traj.et[i],
                sidelobe_only=True,
            )
        )
        counts.append(n)

    n_epochs = len(counts)
    return GnssCoverageStats(
        non_blackout_epochs=n_epochs,
        with_any_prn=int(sum(c > 0 for c in counts)),
        with_min_trackable_prns=int(sum(c >= min_trackable_prns for c in counts)),
        prn_counts=counts,
    )
