#!/usr/bin/env python3
"""Multi-pulsar batch geometry fix + EKF (3+ pulsars)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.catalog import load_catalog
from pulsar_nav.constants import DEFAULT_TOA_SIGMA_S
from pulsar_nav.filter import PulsarNavEKF
from pulsar_nav.measurements import batch_position_fix, synthesize_measurement
from pulsar_nav.simulation import lunar_elo_like_state


def main() -> None:
    pulsars = load_catalog()
    truth_pos, truth_vel = lunar_elo_like_state()
    rng = np.random.default_rng(0)

    measurements = [
        synthesize_measurement(p, truth_pos, rng, DEFAULT_TOA_SIGMA_S)
        for p in pulsars
    ]

    init_pos = truth_pos + np.array([50_000.0, -30_000.0, 20_000.0])
    batch_fix = batch_position_fix(measurements, initial_position_m=np.zeros(3))
    # batch fix from origin uses absolute ranges; compare EKF from bad init
    ekf = PulsarNavEKF.from_initial(init_pos, truth_vel, position_sigma_m=80_000.0)
    for m in measurements:
        ekf.update(m)

    print(f"Truth position (km): {truth_pos / 1e3}")
    print(f"Initial error (m): {np.linalg.norm(init_pos - truth_pos):.1f}")
    print(f"Batch LSQ fix (m) from origin: {np.linalg.norm(batch_fix - truth_pos):.1f}")
    print(f"EKF after {len(pulsars)} updates (m): {ekf.position_rmse_vs_truth(truth_pos):.1f}")


if __name__ == "__main__":
    main()
