"""Gravity-scaled process noise."""

import numpy as np

from pulsar_nav.filter.process_noise import gravity_scaled_q_accel
from pulsar_nav.propagation.dynamics import GM_MOON


def test_gravity_q_grows_at_periapsis():
    q_apo = gravity_scaled_q_accel(8000.0, scale=1.0)
    q_peri = gravity_scaled_q_accel(4000.0, scale=1.0)
    assert q_peri > q_apo


def test_gravity_q_scales_with_factor():
    r = 5000.0
    q1 = gravity_scaled_q_accel(r, scale=1.0)
    q2 = gravity_scaled_q_accel(r, scale=2.0)
    assert abs(q2 / q1 - 4.0) < 1e-9


def test_matches_gm_over_r_squared():
    r = 6000.0
    a_mps2 = GM_MOON / r**2 * 1000.0
    expected = a_mps2**2
    assert abs(gravity_scaled_q_accel(r, scale=1.0) - expected) < 1e-6 * expected
