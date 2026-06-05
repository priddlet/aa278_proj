"""NIS / stacked-update consistency."""

import numpy as np

from pulsar_nav.filter.consistency import normalized_innovation_squared
from pulsar_nav.filter.ekf import PulsarNavEKF, joseph_covariance_update
from pulsar_nav.filter.state import NAV_STATE_DIM


def test_nis_zero_innovation_is_zero():
    s = np.eye(2)
    assert normalized_innovation_squared(np.zeros(2), s) == 0.0


def test_joseph_covariance_symmetric_positive_semidefinite():
    n = 4
    p = np.diag([1e4, 1e4, 1e2, 1e2])
    h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    r = np.diag([100.0, 200.0])
    s = h @ p @ h.T + r
    k = p @ h.T @ np.linalg.inv(s)
    p_j = joseph_covariance_update(p, k, h, r)
    assert np.allclose(p_j, p_j.T, rtol=0, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(p_j) >= -1e-6)


def test_ekf_stacked_update_covariance_symmetric():
    ekf = PulsarNavEKF.from_initial(
        np.array([1e6, 2e5, -1e5]),
        np.array([100.0, -50.0, 20.0]),
        position_sigma_m=50_000.0,
    )
    h = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    y = np.array([1000.0])
    r = np.diag([1e4])
    ekf._stacked_measurement_update(h, y, r)
    p = ekf.covariance
    assert p.shape == (NAV_STATE_DIM, NAV_STATE_DIM)
    assert np.allclose(p, p.T, rtol=0, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(p) >= -1e-3)


def test_ekf_navigation_epoch_records_nis():
    ekf = PulsarNavEKF.from_initial(
        np.array([1e6, 0.0, 0.0]),
        np.zeros(3),
        position_sigma_m=1e5,
    )
    assert ekf.update_navigation_epoch([], [], 0.0) == []
    assert np.isnan(ekf.last_nis)
    assert ekf.last_dof == 0
