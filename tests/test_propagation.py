"""Propagation and orbital element tests."""

import numpy as np
import pytest

from pulsar_nav.propagation.dynamics import DynamicsConfig, acceleration_mci
from pulsar_nav.propagation.elements import coe_to_cart, GM_MOON_KM3_S2
from pulsar_nav.spice.kernels import resolve_kernel_dir

spiceypy = pytest.importorskip("spiceypy")


def _kernels_available() -> bool:
    try:
        resolve_kernel_dir()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_coe_to_cart_circular():
    a = 2000.0
    coe = (a, 0.0, np.pi / 2, 0.0, 0.0, 0.0)
    rv = coe_to_cart(coe)
    assert abs(np.linalg.norm(rv[:3]) - a) < 1e-3
    n = np.sqrt(GM_MOON_KM3_S2 / a**3)
    assert abs(np.linalg.norm(rv[3:]) - a * n) < 1e-2


@pytest.fixture(scope="module")
def spice_loaded():
    from pulsar_nav.spice.kernels import load_kernels

    load_kernels()


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_elfo_propagation_radii(spice_loaded):
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.spice.ephemeris import str_to_et

    et0 = str_to_et("2026-01-15T12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig())
    traj = prop.propagate_preset("elfo", duration_s=3600.0, step_s=300.0)
    radii = np.linalg.norm(traj.position_mci_km, axis=1)
    assert radii.min() > 1500.0
    assert radii.max() < 12000.0
    assert traj.position_icrs_m.shape == (len(traj.t_rel_s), 3)


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_acceleration_finite(spice_loaded):
    from pulsar_nav.spice.ephemeris import str_to_et

    et = str_to_et("2026-01-15T12:00:00")
    r = np.array([7000.0, 0.0, 0.0])
    a = acceleration_mci(r, et, DynamicsConfig())
    assert np.all(np.isfinite(a))
    assert np.linalg.norm(a) > 0.0
