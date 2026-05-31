"""Navigation measurement policies for hybrid / Monte Carlo runs."""

from __future__ import annotations

from enum import Enum


class NavPolicy(str, Enum):
    """
    Which measurements the EKF may use each epoch.

    Visibility (``NavMode``) still gates GNSS / LunaNet; policy sets the stack.
    """

    HYBRID = "hybrid"  # XNAV always + GNSS (non-blackout) + LunaNet when visible
    XNAV_ONLY = "xnav_only"  # Pulsars only every epoch
    GNSS_ONLY = "gnss_only"  # GNSS sidelobes when not in blackout (coast in blackout)
    LONET_ONLY = "lonet_only"  # LunaNet when visible (coast otherwise)
