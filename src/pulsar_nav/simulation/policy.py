"""Navigation measurement policies for hybrid / Monte Carlo runs."""

from __future__ import annotations

from enum import Enum

from pulsar_nav.visibility.blackout import NavMode, VisibilitySample

# Minimum sidelobe PRNs for "trackable" GNSS epoch reporting (LuGRE-style 0–4 regime).
MIN_TRACKABLE_GNSS_PRNS = 4


class NavPolicy(str, Enum):
    """
    Three primary full-arc scenarios (switching by GNSS blackout).

    LunaNet is **supplemental** on top of these phases when relays are visible —
    not a fourth standalone phase.
    """

    XNAV_ONLY = "xnav_only"
    """Pulsars every epoch; supplemental LunaNet in blackout when relay visible."""

    GNSS_ONLY = "gnss_only"
    """GNSS when trackable and not in blackout; pulsars + LunaNet in blackout when relay visible."""

    HYBRID = "hybrid"
    """
    Non-blackout: **fuse** GNSS + pulsars (+ LunaNet if relay visible).
    Blackout: pulsars + supplemental LunaNet when relay visible (GNSS gap fill).
    """

    GNSS_COAST = "gnss_coast"
    """Legacy stress test: GNSS when visible, no measurements in blackout."""


class PolicySegment(str, Enum):
    """What the filter actually applied this epoch (for plots and logs)."""

    XNAV_ONLY_ARC = "xnav_only (full arc)"
    XNAV_BLACKOUT = "xnav (blackout)"
    XNAV_LONET_SUPPLEMENT = "xnav + LunaNet supplemental (blackout)"
    GNSS_VISIBLE = "gnss (sidelobe PRNs)"
    GNSS_XNAV_FALLBACK = "gnss window, 0 PRN → xnav fallback"
    HYBRID_VISIBLE = "hybrid (GNSS + XNAV + LunaNet)"
    HYBRID_GNSS_ONLY = "hybrid (GNSS + XNAV)"
    COAST = "coast (no measurements)"


SEGMENT_COLORS: dict[str, str] = {
    PolicySegment.XNAV_ONLY_ARC.value: "#ef4444",
    PolicySegment.XNAV_BLACKOUT.value: "#ef4444",
    PolicySegment.XNAV_LONET_SUPPLEMENT.value: "#c084fc",
    PolicySegment.GNSS_VISIBLE.value: "#22c55e",
    PolicySegment.GNSS_XNAV_FALLBACK.value: "#f59e0b",
    PolicySegment.HYBRID_VISIBLE.value: "#3b82f6",
    PolicySegment.HYBRID_GNSS_ONLY.value: "#60a5fa",
    PolicySegment.COAST.value: "#6b7280",
}


def segment_from_measurements(
    policy: NavPolicy,
    sample: VisibilitySample,
    *,
    n_gnss: int,
    n_lonet: int,
    n_pulsar: int,
) -> PolicySegment:
    """Label epoch by measurements actually passed to the EKF."""
    if n_gnss == 0 and n_lonet == 0 and n_pulsar == 0:
        return PolicySegment.COAST

    if policy == NavPolicy.XNAV_ONLY:
        return PolicySegment.XNAV_ONLY_ARC

    if sample.in_blackout:
        if n_lonet > 0 and n_pulsar > 0:
            return PolicySegment.XNAV_LONET_SUPPLEMENT
        if n_pulsar > 0:
            return PolicySegment.XNAV_BLACKOUT
        return PolicySegment.COAST

    # Non-blackout (geometric GNSS window)
    if n_gnss == 0 and n_pulsar > 0:
        return PolicySegment.GNSS_XNAV_FALLBACK
    if policy == NavPolicy.GNSS_ONLY:
        return PolicySegment.GNSS_VISIBLE
    if policy == NavPolicy.HYBRID:
        if n_pulsar > 0 and n_gnss > 0 and n_lonet > 0:
            return PolicySegment.HYBRID_VISIBLE
        if n_pulsar > 0 and n_gnss > 0:
            return PolicySegment.HYBRID_GNSS_ONLY
        if n_pulsar > 0 and n_lonet > 0:
            return PolicySegment.HYBRID_VISIBLE
        if n_pulsar > 0:
            return PolicySegment.GNSS_XNAV_FALLBACK
    return PolicySegment.COAST


def planned_segment(policy: NavPolicy, sample: VisibilitySample) -> PolicySegment:
    """
    Segment expected from policy + geometry (for pre-filter orbit plots).

    Mirrors ``measurements_for_epoch`` logic without synthesizing noise.
    """
    if policy == NavPolicy.XNAV_ONLY:
        return PolicySegment.XNAV_ONLY_ARC

    if sample.in_blackout:
        if policy == NavPolicy.GNSS_COAST:
            return PolicySegment.COAST
        if sample.lonet_visible:
            return PolicySegment.XNAV_LONET_SUPPLEMENT
        return PolicySegment.XNAV_BLACKOUT

    if policy == NavPolicy.GNSS_ONLY:
        return PolicySegment.GNSS_VISIBLE

    if policy == NavPolicy.HYBRID:
        if sample.lonet_visible:
            return PolicySegment.HYBRID_VISIBLE
        return PolicySegment.HYBRID_GNSS_ONLY

    return PolicySegment.GNSS_VISIBLE


# Backward-compatible alias (planned, not measured)
def active_segment(policy: NavPolicy, sample: VisibilitySample) -> PolicySegment:
    return planned_segment(policy, sample)
