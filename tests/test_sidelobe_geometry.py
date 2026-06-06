"""Sidelobe visibility gate (synthetic Moon-Earth-GPS geometry)."""

import numpy as np

from pulsar_nav.measurements.gnss_meas import gps_sidelobe_limb_deg
from pulsar_nav.visibility.geometry import earth_limb_angle_deg


def _target_at_limb_deg(
    sc: np.ndarray,
    earth: np.ndarray,
    limb_deg: float,
    *,
    far_side: bool,
) -> np.ndarray:
    """Place a test GPS target at ``limb_deg`` from Earth center as seen from ``sc``."""
    e_hat = (earth - sc) / np.linalg.norm(earth - sc)
    perp = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(e_hat, perp)) > 0.9:
        perp = np.array([0.0, 0.0, 1.0])
    perp -= np.dot(perp, e_hat) * e_hat
    perp /= np.linalg.norm(perp)
    theta = np.deg2rad(limb_deg)
    d_hat = np.cos(theta) * e_hat + np.sin(theta) * perp
    d_sc_earth = np.linalg.norm(earth - sc)
    if far_side:
        dist = d_sc_earth + 30_000.0
    else:
        dist = d_sc_earth - 30_000.0
    return sc + dist * d_hat


def test_sidelobe_gate_rejects_occulted_and_keeps_near_limb():
    """Occulted in-disk targets rejected; clear near-limb far-side targets kept."""
    sc = np.array([7000.0, 0.0, 0.0])
    earth = np.array([384_000.0, 0.0, 0.0])

    occulted_0 = _target_at_limb_deg(sc, earth, 0.0, far_side=False)
    occulted_09 = _target_at_limb_deg(sc, earth, 0.9, far_side=False)
    receivable_18 = _target_at_limb_deg(sc, earth, 1.8, far_side=True)
    receivable_36 = _target_at_limb_deg(sc, earth, 3.6, far_side=True)

    assert gps_sidelobe_limb_deg(sc, occulted_0, earth) is None
    assert gps_sidelobe_limb_deg(sc, occulted_09, earth) is None
    assert gps_sidelobe_limb_deg(sc, receivable_18, earth) is not None
    assert gps_sidelobe_limb_deg(sc, receivable_36, earth) is not None

    assert abs(gps_sidelobe_limb_deg(sc, receivable_18, earth) - 1.8) < 0.2
    assert abs(gps_sidelobe_limb_deg(sc, receivable_36, earth) - 3.6) < 0.2


def test_sidelobe_gate_rejects_near_side_clear_los():
    """Clear LOS but GPS closer than Earth (beam away from Moon) is rejected."""
    sc = np.array([7000.0, 0.0, 0.0])
    earth = np.array([384_000.0, 0.0, 0.0])
    near_side = _target_at_limb_deg(sc, earth, 2.0, far_side=False)
    assert np.linalg.norm(near_side - sc) < np.linalg.norm(earth - sc)
    assert gps_sidelobe_limb_deg(sc, near_side, earth) is None


def test_limb_angle_matches_geometry_helper():
    sc = np.array([7000.0, 0.0, 0.0])
    earth = np.array([384_000.0, 0.0, 0.0])
    tgt = _target_at_limb_deg(sc, earth, 3.0, far_side=True)
    limb = gps_sidelobe_limb_deg(sc, tgt, earth)
    assert limb is not None
    assert abs(limb - earth_limb_angle_deg(sc, tgt, earth)) < 1e-9
