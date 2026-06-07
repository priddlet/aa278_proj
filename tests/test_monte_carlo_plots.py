"""Tests for Monte Carlo visualization helpers."""

import numpy as np

from pulsar_nav.visualization.monte_carlo_plots import _error_ylim_km


def test_error_ylim_km_skips_epoch_zero_spike():
    trace = np.array([70.0, 0.5, 0.6, 0.4, 0.55, 0.48, 0.52, 0.47])
    ymin, ymax = _error_ylim_km(trace)
    assert ymax < 5.0
    assert ymin >= 0.0


def test_error_ylim_km_includes_blackout_coast_errors():
    trace = np.concatenate(
        [
            np.array([80.0, 0.3, 0.4]),
            np.full(20, 45.0),
            np.full(10, 0.35),
        ]
    )
    ymin, ymax = _error_ylim_km(trace)
    assert ymax > 40.0
    assert ymin >= 0.0
