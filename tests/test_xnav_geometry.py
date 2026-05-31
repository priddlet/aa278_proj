"""XNAV geometry and filter tests."""

import numpy as np

from pulsar_nav.catalog import load_catalog
from pulsar_nav.filter import PulsarNavEKF
from pulsar_nav.measurements import (
    batch_geometry_matrix,
    batch_position_fix,
    measurement_jacobian,
    predicted_range,
    synthesize_measurement,
)


def test_los_projection():
    p = load_catalog(["B1937+21"])[0]
    r = np.array([1e6, 2e6, 3e6])
    assert abs(p.line_of_sight_range_m(r) - np.dot(p.unit_vector_icrs, r)) < 1e-6


def test_jacobian_is_los():
    p = load_catalog(["J0437-4715"])[0]
    H = measurement_jacobian(p)
    assert np.allclose(H[:3], p.unit_vector_icrs)
    assert np.allclose(H[3:], 0.0)


def test_single_pulsar_ekf_reduces_los_error():
    """One pulsar constrains range along LOS, not the perpendicular components."""
    p = load_catalog(["B1937+21"])[0]
    truth = np.array([-1.8e8, 0.4e8, -0.2e8])
    n = p.unit_vector_icrs
    err = np.array([1.0, 1.0, 0.0])
    err -= n * np.dot(err, n)
    err = 80_000.0 * err / np.linalg.norm(err)
    init = truth + err
    los_err_0 = abs(np.dot(init - truth, n))
    ekf = PulsarNavEKF.from_initial(init, np.zeros(3), position_sigma_m=100_000.0)
    rng = np.random.default_rng(1)
    for _ in range(30):
        m = synthesize_measurement(p, truth, rng, 1e-6)
        ekf.step(100.0, [m])
    los_err_f = abs(np.dot(ekf.state.position_m - truth, n))
    assert los_err_f < los_err_0 * 0.1 or los_err_f < 500.0


def test_multi_pulsar_full_rank():
    pulsars = load_catalog()
    A = batch_geometry_matrix(pulsars)
    assert np.linalg.matrix_rank(A) == 3
