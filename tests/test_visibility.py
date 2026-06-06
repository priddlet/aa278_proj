"""Visibility and blackout geometry tests."""

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

    load_kernels()


def test_earth_elevation_sign():
    from pulsar_nav.visibility.geometry import earth_elevation_mci

    # Spacecraft on +X axis, Earth on -X -> Earth below horizon
    r_sc = np.array([7000.0, 0.0, 0.0])
    r_earth = np.array([-384000.0, 0.0, 0.0])
    el = earth_elevation_mci(r_sc, r_earth)
    assert el < 0.0

    # Earth in same direction as zenith -> positive elevation
    r_sc2 = np.array([7000.0, 0.0, 0.0])
    r_earth2 = np.array([400000.0, 0.0, 0.0])
    el2 = earth_elevation_mci(r_sc2, r_earth2)
    assert el2 > np.deg2rad(5.0)


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_elfo_has_blackout_windows(spice_loaded):
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.visibility.blackout import compute_visibility_timeline
    from pulsar_nav.spice.ephemeris import str_to_et

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=30.0 * 3600.0, step_s=300.0)
    timeline = compute_visibility_timeline(traj)

    assert timeline.blackout_fraction > 0.15
    assert len(timeline.windows) >= 1
    assert timeline.total_blackout_s > 3600.0


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_blackout_implies_xnav_mode(spice_loaded):
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.visibility.blackout import NavMode, compute_visibility_timeline
    from pulsar_nav.spice.ephemeris import str_to_et

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=6.0 * 3600.0, step_s=120.0)
    timeline = compute_visibility_timeline(traj)

    for s in timeline.samples:
        if s.in_blackout:
            assert s.nav_mode in (NavMode.XNAV, NavMode.LONET)
            assert not s.gnss_visible
