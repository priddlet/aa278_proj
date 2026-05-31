"""Pseudorange measurement model tests."""

import numpy as np
import pytest

from pulsar_nav.filter.state import NavState
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
from pulsar_nav.measurements.pseudorange import (
    predicted_pseudorange_m,
    pseudorange_jacobian_m,
    pseudorange_residual,
    synthesize_pseudorange,
)


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_pseudorange_zero_innovation_at_truth(spice_loaded):
    from pulsar_nav.spice.ephemeris import str_to_et

    rng = np.random.default_rng(0)
    r_true = np.array([-1.8e8, 0.4e8, -0.2e8])
    r_tx = np.array([7000.0, 2000.0, 1500.0])
    et = str_to_et("2026-01-15 12:00:00")
    m = synthesize_pseudorange(r_true, r_tx, 0.0, et, rng, sigma_m=1.0)
    state = NavState.from_pv(r_true, np.zeros(3), clock_bias_m=0.0)
    # Re-synthesize without noise for exact check
    m.range_m = predicted_pseudorange_m(state, r_tx, 0.0, et)
    y = pseudorange_residual(m, state, et)
    assert abs(y) < 1.0


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_pseudorange_jacobian_position(spice_loaded):
    from pulsar_nav.spice.ephemeris import str_to_et

    r_true = np.array([-1.8e8, 0.4e8, -0.2e8])
    r_tx = np.array([8000.0, 1000.0, 500.0])
    state = NavState.from_pv(r_true, np.zeros(3), clock_bias_m=10.0)
    et = str_to_et("2026-01-15 12:00:00")

    H = pseudorange_jacobian_m(state, r_tx, et)
    eps = 1.0
    numeric = np.zeros(3)
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = eps
        s_plus = NavState.from_pv(r_true + dp, np.zeros(3), clock_bias_m=10.0)
        s_minus = NavState.from_pv(r_true - dp, np.zeros(3), clock_bias_m=10.0)
        numeric[k] = (
            predicted_pseudorange_m(s_plus, r_tx, 0.0, et)
            - predicted_pseudorange_m(s_minus, r_tx, 0.0, et)
        ) / (2 * eps)
    assert np.allclose(H[:3], numeric, rtol=1e-5, atol=1e-3)


def test_gps_constellation_offsets():
    from pulsar_nav.measurements.gnss_sim import gps_positions_mci, gps_satellite_offsets_km

    offs = gps_satellite_offsets_km(6)
    assert offs.shape == (6, 3)
    earth = np.array([380000.0, 0.0, 0.0])
    gps = gps_positions_mci(earth, 0.0)
    assert gps.shape == (6, 3)
    assert np.all(np.linalg.norm(gps - earth, axis=1) > 20000.0)
