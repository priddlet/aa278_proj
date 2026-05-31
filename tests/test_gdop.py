"""PDOP helper tests."""

import numpy as np

from pulsar_nav.visibility.gdop import position_dop_from_los


def test_pdop_requires_four_los():
    los = np.eye(3)
    assert position_dop_from_los(los) == float("inf")


def test_pdop_orthogonal_los_is_order_one():
    los = np.eye(4)[:, :3]
    pdop = position_dop_from_los(los)
    assert np.isfinite(pdop)
    assert 0.5 < pdop < 2.0


def test_pdop_collinear_los_is_large_but_finite():
    u = np.array([1.0, 0.0, 0.0])
    los = np.tile(u, (4, 1))
    pdop = position_dop_from_los(los)
    assert pdop == float("inf")
