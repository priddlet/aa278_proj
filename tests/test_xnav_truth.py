"""XNAV EKF integrated with propagated truth."""

import numpy as np
import pytest

from pulsar_nav.catalog import load_catalog
from pulsar_nav.spice.kernels import resolve_kernel_dir

spiceypy = pytest.importorskip("spiceypy")


def _kernels_available() -> bool:
    try:
        resolve_kernel_dir()
        return True
    except FileNotFoundError:
        return False


@pytest.fixture(scope="module")
def spice_loaded():
    from pulsar_nav.spice.kernels import load_kernels

    load_kernels()


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_multi_pulsar_beats_single_on_propagated_truth(spice_loaded):
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.simulation.xnav_run import run_xnav_on_propagated
    from pulsar_nav.spice.ephemeris import str_to_et

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=3600.0, step_s=120.0)

    kw = dict(
        toa_sigma_s=1e-4,
        process_noise_accel=1e-4,
        use_truth_velocity_predict=True,
    )
    multi = run_xnav_on_propagated(
        traj, load_catalog(), position_offset_m=30_000.0, rng=np.random.default_rng(0), **kw
    )
    single = run_xnav_on_propagated(
        traj,
        load_catalog(["B1937+21"]),
        position_offset_m=30_000.0,
        unobservable_offset=True,
        rng=np.random.default_rng(1),
        **kw,
    )

    assert multi.final_position_rmse_m < single.final_position_rmse_m
    assert multi.final_position_rmse_m < 0.6 * multi.position_error_m[0]
    assert single.los_error_m[-1] < single.position_error_m[0]
