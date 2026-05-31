"""GPS broadcast ephemeris tests."""

import numpy as np
import pytest

from pulsar_nav.ephemeris.paths import resolve_brdc_path

spiceypy = pytest.importorskip("spiceypy")


def _brdc_available() -> bool:
    try:
        resolve_brdc_path()
        return True
    except FileNotFoundError:
        return False


@pytest.fixture(scope="module")
def gps_ready():
    from pulsar_nav.spice.kernels import load_kernels

    load_kernels(load_gps_frames=True)


@pytest.mark.skipif(not _brdc_available(), reason="brdc_data.npz not on disk")
def test_get_gps_posclk_mci(gps_ready):
    from pulsar_nav.ephemeris.gps_posclk import get_gps_posclk_mci
    from pulsar_nav.spice.ephemeris import str_to_et

    et = str_to_et("2026-01-15 12:00:00")
    r, b = get_gps_posclk_mci(et, 5)
    assert r.shape == (3,)
    assert np.linalg.norm(r) > 100_000.0
    assert abs(b) < 1000.0


@pytest.mark.skipif(not _brdc_available(), reason="brdc_data.npz not on disk")
def test_visible_gps_prns_on_lunar_orbit(gps_ready):
    from pulsar_nav.ephemeris.gps_posclk import get_gps_posclk_mci
    from pulsar_nav.measurements.gnss_meas import visible_gps_prns
    from pulsar_nav.spice.ephemeris import str_to_et

    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.propagation.dynamics import DynamicsConfig

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=60.0, step_s=60.0)
    r_sc = traj.position_mci_km[0]
    prns = visible_gps_prns(r_sc, et0)
    assert len(prns) >= 1
    for prn in prns[:3]:
        r, _ = get_gps_posclk_mci(et0, prn)
        assert np.linalg.norm(r - r_sc) > 1000.0
