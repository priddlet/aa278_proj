#!/usr/bin/env python3
"""
Single-pulsar XNAV geometry validation (AA278 Week 7).

Demonstrates that LOS range measurements n_hat·r constrain position
along the pulsar direction and that the EKF reduces position error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.catalog import load_catalog
from pulsar_nav.constants import DEFAULT_TOA_SIGMA_S
from pulsar_nav.filter import PulsarNavEKF
from pulsar_nav.measurements import synthesize_measurement
from pulsar_nav.simulation import constant_velocity_trajectory, lunar_elo_like_state


def main() -> None:
    pulsar = load_catalog(["B1937+21"])[0]
    truth_pos, truth_vel = lunar_elo_like_state()

    # Large initial position error (100 km) orthogonal-ish to LOS.
    n_hat = pulsar.unit_vector_icrs
    err = np.array([1.0, -0.5, 0.3])
    err -= n_hat * np.dot(err, n_hat)  # component perpendicular to LOS
    err = 100_000.0 * err / np.linalg.norm(err)
    init_pos = truth_pos + err

    ekf = PulsarNavEKF.from_initial(
        init_pos,
        truth_vel,
        position_sigma_m=150_000.0,
        velocity_sigma_m_s=50.0,
    )

    rng = np.random.default_rng(42)
    dt_s = 300.0
    n_steps = 40
    times = np.arange(n_steps) * dt_s
    traj = constant_velocity_trajectory(times, truth_pos, truth_vel)

    los_errors = []
    for sample in traj:
        meas = synthesize_measurement(
            pulsar, sample.position_m, rng, DEFAULT_TOA_SIGMA_S
        )
        ekf.step(dt_s, [meas])
        los_errors.append(
            abs(np.dot(ekf.state.position_m - sample.position_m, n_hat))
        )

    print(f"Pulsar: {pulsar.name}  F0={pulsar.f0_hz:.6f} Hz")
    print(f"LOS unit vector (ICRS): {n_hat}")
    print(f"Initial LOS position error: {los_errors[0]:.1f} m")
    print(f"Final LOS position error:   {los_errors[-1]:.1f} m")
    print(f"LOS observable at truth: {pulsar.line_of_sight_range_m(truth_pos):.1f} m")
    print("(Perpendicular errors are unobservable with a single pulsar.)")


if __name__ == "__main__":
    main()
