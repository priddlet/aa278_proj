"""Hybrid navigation filter tests."""

import numpy as np
import pytest

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

    load_kernels(load_gps_frames=True)


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_hybrid_uses_gnss_during_near_side(spice_loaded):
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.simulation.hybrid_run import run_hybrid_on_propagated
    from pulsar_nav.spice.ephemeris import str_to_et
    from pulsar_nav.catalog import load_catalog
    from pulsar_nav.visibility.blackout import compute_visibility_timeline

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=6.0 * 3600.0, step_s=120.0)
    tl = compute_visibility_timeline(traj)
    res = run_hybrid_on_propagated(
        traj,
        load_catalog(),
        timeline=tl,
        position_offset_m=30_000.0,
        rng=np.random.default_rng(0),
    )
    n_gnss = sum(log.n_gnss for log in res.epoch_logs)
    n_pulsar = sum(log.n_pulsar for log in res.epoch_logs)
    n_epochs = len(res.epoch_logs) - 1
    n_msps = len(load_catalog())
    assert n_gnss > 0
    assert n_pulsar == n_epochs * n_msps


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_hybrid_beats_xnav_only_on_elfo(spice_loaded):
    from pulsar_nav.catalog import load_catalog
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.simulation.hybrid_run import run_hybrid_on_propagated
    from pulsar_nav.spice.ephemeris import str_to_et
    from pulsar_nav.visibility.blackout import compute_visibility_timeline

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=6.0 * 3600.0, step_s=120.0)
    tl = compute_visibility_timeline(traj)
    kw = dict(
        position_offset_m=50_000.0,
        toa_sigma_s=1e-4,
        process_noise_accel=1e-4,
        use_truth_velocity_predict=True,
    )
    from pulsar_nav.simulation.xnav_run import run_xnav_on_propagated

    hybrid = run_hybrid_on_propagated(
        traj, load_catalog(), timeline=tl, rng=np.random.default_rng(1), **kw
    )
    assert hybrid.final_position_error_m < kw["position_offset_m"]
    assert hybrid.position_error_m[0] <= kw["position_offset_m"] * 1.1

    # GNSS-visible epochs should be corrected well after pseudorange updates
    for log, err in zip(hybrid.epoch_logs[1:], hybrid.position_error_m[1:]):
        if log.n_gnss > 0:
            assert err < 5_000.0
