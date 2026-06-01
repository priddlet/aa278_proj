"""NIS / stacked-update consistency."""

import numpy as np

from pulsar_nav.filter.consistency import normalized_innovation_squared
from pulsar_nav.filter.ekf import PulsarNavEKF
from pulsar_nav.filter.state import NAV_STATE_DIM


def test_nis_zero_innovation_is_zero():
    s = np.eye(2)
    assert normalized_innovation_squared(np.zeros(2), s) == 0.0


def test_ekf_navigation_epoch_records_nis():
    ekf = PulsarNavEKF.from_initial(
        np.array([1e6, 0.0, 0.0]),
        np.zeros(3),
        position_sigma_m=1e5,
    )
    assert ekf.update_navigation_epoch([], [], 0.0) == []
    assert np.isnan(ekf.last_nis)
    assert ekf.last_dof == 0
