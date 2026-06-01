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
    n_msps = len(load_catalog())
    assert sum(log.n_gnss for log in res.epoch_logs) > 0
    for log in res.epoch_logs[1:]:
        if log.in_blackout:
            assert log.n_pulsar == n_msps
            assert log.n_gnss == 0
        else:
            assert log.n_gnss > 0
            assert log.n_pulsar == n_msps, "non-blackout hybrid fuses GNSS with all MSPs"


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_hybrid_lonet_supplement_in_blackout_when_relay_visible(spice_loaded):
    from pulsar_nav.catalog import load_catalog
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.simulation.hybrid_run import run_hybrid_on_propagated
    from pulsar_nav.simulation.policy import PolicySegment
    from pulsar_nav.spice.ephemeris import str_to_et
    from pulsar_nav.visibility.blackout import compute_visibility_timeline

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=26.4 * 3600.0, step_s=120.0)
    tl = compute_visibility_timeline(traj)
    res = run_hybrid_on_propagated(
        traj, load_catalog(), timeline=tl, rng=np.random.default_rng(1)
    )

    supplemental = [
        log
        for log in res.epoch_logs[1:]
        if log.in_blackout and log.n_lonet > 0 and log.n_pulsar > 0
    ]
    assert supplemental, "HYBRID should use supplemental LunaNet in some blackout epochs"
    assert all(
        log.policy_segment == PolicySegment.XNAV_LONET_SUPPLEMENT for log in supplemental
    )


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_gnss_only_lonet_supplement_in_blackout_when_relay_visible(spice_loaded):
    from pulsar_nav.catalog import load_catalog
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.simulation.hybrid_run import run_hybrid_on_propagated
    from pulsar_nav.simulation.policy import NavPolicy, PolicySegment
    from pulsar_nav.spice.ephemeris import str_to_et
    from pulsar_nav.visibility.blackout import compute_visibility_timeline

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=26.4 * 3600.0, step_s=120.0)
    tl = compute_visibility_timeline(traj)
    res = run_hybrid_on_propagated(
        traj,
        load_catalog(),
        policy=NavPolicy.GNSS_ONLY,
        timeline=tl,
        rng=np.random.default_rng(2),
    )
    supplemental = [
        log
        for log in res.epoch_logs[1:]
        if log.in_blackout and log.n_lonet > 0 and log.n_pulsar > 0
    ]
    assert supplemental, "GNSS-only should use LunaNet + pulsars in some blackout epochs"
    assert all(
        log.policy_segment == PolicySegment.XNAV_LONET_SUPPLEMENT for log in supplemental
    )


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

    hybrid = run_hybrid_on_propagated(
        traj, load_catalog(), timeline=tl, rng=np.random.default_rng(1), **kw
    )
    assert hybrid.final_position_error_m < kw["position_offset_m"]
    assert hybrid.position_error_m[0] <= kw["position_offset_m"] * 1.1

    gnss_errors = [
        err
        for log, err in zip(hybrid.epoch_logs[1:], hybrid.position_error_m[1:])
        if log.n_gnss > 0
    ]
    assert gnss_errors
    assert float(np.mean(gnss_errors)) < 8_000.0
