"""Navigation measurement policies for hybrid / Monte Carlo runs."""

from __future__ import annotations

from enum import Enum

from pulsar_nav.visibility.blackout import NavMode, VisibilitySample


class NavPolicy(str, Enum):
    """
    Three primary full-arc scenarios (pitch-style switching by GNSS blackout).

    Blackout = no GNSS sidelobe (``in_blackout``). Non-blackout = GNSS visible.
    """

    XNAV_ONLY = "xnav_only"
    """Pulsars every epoch for the whole orbit."""

    GNSS_ONLY = "gnss_only"
    """GNSS pseudoranges when not in blackout; pulsars only in blackout."""

    HYBRID = "hybrid"
    """GNSS + LunaNet (when relays visible) when not in blackout; pulsars in blackout."""

    GNSS_COAST = "gnss_coast"
    """Legacy stress test: GNSS when visible, no measurements in blackout."""


class PolicySegment(str, Enum):
    """Active measurement segment for plots (one label per epoch per policy)."""

    XNAV_ONLY_ARC = "xnav_only (full arc)"
    XNAV_BLACKOUT = "xnav (blackout)"
    GNSS_VISIBLE = "gnss (GNSS visible)"
    HYBRID_VISIBLE = "hybrid (GNSS + LunaNet)"
    HYBRID_GNSS_ONLY = "hybrid (GNSS, no relay)"
    COAST = "coast (no measurements)"


SEGMENT_COLORS: dict[str, str] = {
    PolicySegment.XNAV_ONLY_ARC.value: "#ef4444",
    PolicySegment.XNAV_BLACKOUT.value: "#ef4444",
    PolicySegment.GNSS_VISIBLE.value: "#22c55e",
    PolicySegment.HYBRID_VISIBLE.value: "#3b82f6",
    PolicySegment.HYBRID_GNSS_ONLY.value: "#60a5fa",
    PolicySegment.COAST.value: "#6b7280",
}


def active_segment(
    policy: NavPolicy,
    sample: VisibilitySample,
) -> PolicySegment:
    """Which measurement set the filter uses this epoch."""
    if policy == NavPolicy.XNAV_ONLY:
        return PolicySegment.XNAV_ONLY_ARC

    if sample.in_blackout:
        if policy == NavPolicy.GNSS_COAST:
            return PolicySegment.COAST
        return PolicySegment.XNAV_BLACKOUT

    if policy == NavPolicy.GNSS_ONLY:
        return PolicySegment.GNSS_VISIBLE

    if policy == NavPolicy.HYBRID:
        if sample.nav_mode == NavMode.HYBRID:
            return PolicySegment.HYBRID_VISIBLE
        return PolicySegment.HYBRID_GNSS_ONLY

    if policy == NavPolicy.GNSS_COAST:
        return PolicySegment.GNSS_VISIBLE

    return PolicySegment.XNAV_BLACKOUT
