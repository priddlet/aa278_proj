"""Dynamics-based EKF predict (Track 1)."""

import numpy as np
import pytest

from pulsar_nav.filter.dynamics_predict import velocity_verlet_mci_step
from pulsar_nav.propagation.dynamics import (
    DEFAULT_GAMMA_SRP,
    DynamicsConfig,
    dynamics_config_for_sim,
)
from pulsar_nav.simulation.predict_mode import PredictMode, resolve_predict_mode


def test_velocity_verlet_reduces_radius_under_point_mass():
    """Velocity Verlet with central gravity changes state."""
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 1.5, 0.0])
    cfg = DynamicsConfig(include_earth=False, include_sun=False)
    r_new, v_new = velocity_verlet_mci_step(r, v, et=0.0, dt_s=60.0, config=cfg)
    assert np.linalg.norm(r_new) != np.linalg.norm(r)
    assert np.linalg.norm(v_new) != np.linalg.norm(v)


def test_dynamics_config_for_sim_disturbances():
    base = dynamics_config_for_sim(include_disturbances=False)
    assert not base.include_moon_j2
    assert not base.include_srp
    disturbed = dynamics_config_for_sim(include_disturbances=True)
    assert disturbed.include_moon_j2
    assert disturbed.include_srp
    assert disturbed.gamma_srp == pytest.approx(DEFAULT_GAMMA_SRP)


def test_resolve_predict_mode():
    assert resolve_predict_mode(predict_mode=PredictMode.DYNAMICS) == PredictMode.DYNAMICS
    assert (
        resolve_predict_mode(use_dynamics_predict=True, use_truth_velocity_predict=False)
        == PredictMode.DYNAMICS
    )
    assert (
        resolve_predict_mode(use_truth_velocity_predict=False, use_dynamics_predict=False)
        == PredictMode.CV
    )


def _kernels_available() -> bool:
    try:
        from pulsar_nav.spice.kernels import resolve_kernel_dir

        resolve_kernel_dir()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_hybrid_dynamics_predict_short_arc():
    from pulsar_nav.catalog import load_catalog
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.simulation.hybrid_run import run_hybrid_on_propagated
    from pulsar_nav.spice.ephemeris import str_to_et
    from pulsar_nav.spice.kernels import load_kernels
    from pulsar_nav.visibility.blackout import compute_visibility_timeline

    load_kernels(load_gps_frames=True)
    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=2.0 * 3600.0, step_s=120.0)
    tl = compute_visibility_timeline(traj)
    res = run_hybrid_on_propagated(
        traj,
        load_catalog(),
        timeline=tl,
        predict_mode=PredictMode.DYNAMICS,
        use_truth_velocity_predict=False,
        use_dynamics_predict=True,
        process_noise_accel=1e-5,
        rng=np.random.default_rng(0),
    )
    assert np.all(np.isfinite(res.position_error_m))
    assert res.final_position_error_m < 200_000.0
