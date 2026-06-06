"""Short labels and styling for slide/report figures."""

from __future__ import annotations

from pulsar_nav.simulation.policy import NavPolicy, PolicySegment

POLICY_DISPLAY: dict[NavPolicy, str] = {
    NavPolicy.XNAV_ONLY: "XNAV-only",
    NavPolicy.GNSS_ONLY: "GNSS-only",
    NavPolicy.HYBRID: "Hybrid",
    NavPolicy.GNSS_COAST: "GNSS coast",
}

SEGMENT_PLOT_LABEL: dict[str, str] = {
    PolicySegment.XNAV_ONLY_ARC.value: "MSP",
    PolicySegment.XNAV_BLACKOUT.value: "MSP (blackout)",
    PolicySegment.XNAV_LONET_SUPPLEMENT.value: "MSP + LunaNet (blackout)",
    PolicySegment.GNSS_VISIBLE.value: "GNSS",
    PolicySegment.GNSS_XNAV_FALLBACK.value: "GNSS -> MSP fallback",
    PolicySegment.HYBRID_VISIBLE.value: "GNSS + MSP + LunaNet",
    PolicySegment.HYBRID_GNSS_ONLY.value: "GNSS + MSP",
    PolicySegment.COAST.value: "Coast",
}


def policy_display_name(policy: NavPolicy) -> str:
    return POLICY_DISPLAY.get(policy, policy.value)


def segment_plot_label(segment: PolicySegment | str) -> str:
    key = segment.value if isinstance(segment, PolicySegment) else str(segment)
    return SEGMENT_PLOT_LABEL.get(key, key)


def apply_presentation_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 1.5,
            "figure.dpi": 150,
        }
    )
