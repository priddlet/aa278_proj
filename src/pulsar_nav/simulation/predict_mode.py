"""EKF time-update (predict) modes for hybrid / Monte Carlo runs."""

from __future__ import annotations

from enum import Enum


class PredictMode(str, Enum):
    """How the filter propagates between measurement epochs."""

    TRUTH_VELOCITY = "truth_velocity"
    """Oracle: advance position with truth velocity (simulation aid)."""

    CV = "cv"
    """Constant-velocity predict using estimated velocity."""

    DYNAMICS = "dynamics"
    """MCI force model (Moon + Earth + Sun) on the estimated state, numeric Phi."""


def resolve_predict_mode(
    *,
    predict_mode: PredictMode | str | None = None,
    use_truth_velocity_predict: bool = True,
    use_dynamics_predict: bool = False,
) -> PredictMode:
    """Resolve mode from explicit ``predict_mode`` or legacy boolean flags."""
    if predict_mode is not None:
        return predict_mode if isinstance(predict_mode, PredictMode) else PredictMode(predict_mode)
    if use_dynamics_predict:
        return PredictMode.DYNAMICS
    if use_truth_velocity_predict:
        return PredictMode.TRUTH_VELOCITY
    return PredictMode.CV
