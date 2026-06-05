"""Tier A: RK45 + analytic STM dynamics predict."""

import numpy as np
import pytest

from pulsar_nav.filter.dynamics_predict import (
    full_state_transition,
    numeric_pv_transition,
    propagate_mci_pv_with_stm,
    pv_transition_stm,
    velocity_verlet_mci_step,
)
from pulsar_nav.filter.hw2_process_noise import process_noise_hw2
from pulsar_nav.propagation.dynamics import DynamicsConfig


def _kernels_available() -> bool:
    try:
        from pulsar_nav.spice.kernels import resolve_kernel_dir

        resolve_kernel_dir()
        return True
    except FileNotFoundError:
        return False


@pytest.fixture(scope="module")
def spice_loaded():
    from pulsar_nav.spice.kernels import load_kernels

    load_kernels()


def test_process_noise_hw2_positive_definite():
    q = process_noise_hw2(120.0, sigma_acc_km=1e-6)
    assert q.shape == (10, 10)
    evals = np.linalg.eigvalsh(q)
    assert np.all(evals >= -1e-20)


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_stm_matches_numeric_phi_loosely(spice_loaded):
    cfg = DynamicsConfig(include_earth=False, include_sun=False)
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 1.5, 0.0])
    et = 0.0
    dt = 60.0
    _, _, phi_stm = propagate_mci_pv_with_stm(r, v, et, dt, cfg)
    pos_m = r * 1000.0
    vel_m_s = v * 1000.0
    phi_num = numeric_pv_transition(pos_m, vel_m_s, et, dt, cfg)
    assert np.allclose(phi_stm, phi_num, rtol=0.05, atol=0.02)


def test_stm_not_identity_over_120s():
    cfg = DynamicsConfig(include_earth=False, include_sun=False)
    r = np.array([7000.0, 100.0, 0.0])
    v = np.array([0.0, 1.4, 0.1])
    _, _, phi = propagate_mci_pv_with_stm(r, v, 0.0, 120.0, cfg)
    assert not np.allclose(phi, np.eye(6), atol=0.01)


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_full_state_transition_finite(spice_loaded):
    from pulsar_nav.filter.state import NAV_STATE_DIM
    from pulsar_nav.propagation.dynamics import dynamics_config_for_sim

    cfg = dynamics_config_for_sim(include_disturbances=False)
    pos = np.array([-1.8e8, 0.4e8, -0.2e8])
    vel = np.array([120.0, -80.0, 15.0])
    phi, vec = full_state_transition(
        pos, vel, 0.0, 0.0, 0.0, 120.0, cfg, NAV_STATE_DIM
    )
    assert np.all(np.isfinite(phi))
    assert np.all(np.isfinite(vec))
    assert vec.shape == (NAV_STATE_DIM,)
