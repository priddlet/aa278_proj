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
def test_icrs_mci_frame_roundtrip_and_velocity(spice_loaded):
    """ICRS <-> MCI conversions match truth propagator storage."""
    from pulsar_nav.measurements.pseudorange import icrs_position_to_mci_km
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.spice.ephemeris import (
        icrs_position_from_mci_km,
        icrs_velocity_from_mci_km_s,
        str_to_et,
    )

    et0 = str_to_et("2026-01-15T12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=6 * 3600.0, step_s=120.0)
    for i in range(len(traj.et)):
        r_mci = icrs_position_to_mci_km(traj.position_icrs_m[i], traj.et[i])
        pos_back = icrs_position_from_mci_km(r_mci, traj.et[i]) * 1000.0
        assert np.linalg.norm(pos_back - traj.position_icrs_m[i]) < 1e-6
        v_recon = icrs_velocity_from_mci_km_s(traj.velocity_mci_km_s[i], traj.et[i]) * 1000.0
        assert np.linalg.norm(v_recon - traj.velocity_icrs_m_s[i]) < 0.1


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_acceleration_finite(spice_loaded):
    from pulsar_nav.spice.ephemeris import str_to_et

    et = str_to_et("2026-01-15T12:00:00")
    r = np.array([7000.0, 0.0, 0.0])
    a = acceleration_mci(r, et, DynamicsConfig())
    assert np.all(np.isfinite(a))
    assert np.linalg.norm(a) > 0.0


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_mci_to_op_rotation_matches_hw_p23(spice_loaded):
    """Regression against AA278 HW2 P2.3 autograder at 2026-01-01T00:00:00."""
    from pulsar_nav.spice.ephemeris import mci_to_op_rotation, str_to_et

    et = str_to_et("2026-01-01T00:00:00")
    rot = mci_to_op_rotation(et)
    expected = np.array(
        [
            0.945, -0.3, -0.13, 0.0, 0.0, 0.0,
            0.326, 0.829, 0.455, 0.0, 0.0, 0.0,
            -0.029, -0.473, 0.881, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.945, -0.3, -0.13,
            0.0, 0.0, 0.0, 0.326, 0.829, 0.455,
            0.0, 0.0, 0.0, -0.029, -0.473, 0.881,
        ]
    ).reshape(6, 6)
    assert rot.shape == (6, 6)
    np.testing.assert_allclose(rot, expected, atol=1e-3)
    np.testing.assert_allclose(rot.T @ rot, np.eye(6), atol=1e-12)


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_initial_state_mci_from_coe_op_roundtrip(spice_loaded):
    from pulsar_nav.propagation.propagator import elfo_initial_coe_op, initial_state_mci_from_coe_op
    from pulsar_nav.spice.ephemeris import mci_to_op_rotation, str_to_et

    et0 = str_to_et("2026-01-15 12:00:00")
    coe = elfo_initial_coe_op()
    x_op = coe_to_cart(coe)
    x_mci = initial_state_mci_from_coe_op(coe, et0)
    rot = mci_to_op_rotation(et0)
    np.testing.assert_allclose(rot @ x_mci, x_op, atol=1e-9)
    np.testing.assert_allclose(rot.T @ x_op, x_mci, atol=1e-9)
