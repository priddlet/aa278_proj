"""LunaNet relay pseudorange measurement synthesis."""

from __future__ import annotations

import numpy as np

from pulsar_nav.measurements.pseudorange import PseudorangeMeasurement, synthesize_pseudorange
from pulsar_nav.visibility.geometry import elevation_angle
from pulsar_nav.visibility.lonet import LunaNetConfig


def select_visible_relays(
    spacecraft_mci_km: np.ndarray,
    relay_positions_mci_km: np.ndarray,
    *,
    min_elevation_deg: float = 5.0,
    max_relays: int = 4,
) -> list[int]:
    """Indices of relays above mask, sorted by elevation (best first)."""
    sc = np.asarray(spacecraft_mci_km, float)
    zenith = sc / np.linalg.norm(sc)
    min_el = np.deg2rad(min_elevation_deg)
    candidates: list[tuple[float, int]] = []
    for i, r_sat in enumerate(relay_positions_mci_km):
        el = elevation_angle(sc, r_sat, zenith_hat=zenith)
        if el >= min_el:
            candidates.append((el, i))
    candidates.sort(reverse=True)
    return [idx for _, idx in candidates[:max_relays]]


def lonet_pseudoranges(
    true_position_icrs_m: np.ndarray,
    relay_positions_mci_km: np.ndarray,
    et_rx: float,
    rng: np.random.Generator,
    *,
    sigma_m: float = 15.0,
    relay_clock_sigma_km: float = 0.001,
    et0: float | None = None,
    lonet_config: LunaNetConfig | None = None,
    max_relays: int = 4,
) -> list[PseudorangeMeasurement]:
    """Synthesize pseudoranges from visible LunaNet relays."""
    cfg = lonet_config or LunaNetConfig()
    from pulsar_nav.measurements.pseudorange import icrs_position_to_mci_km

    r_sc_km = icrs_position_to_mci_km(true_position_icrs_m, et_rx)
    indices = select_visible_relays(
        r_sc_km,
        relay_positions_mci_km,
        min_elevation_deg=cfg.min_elevation_deg,
        max_relays=max_relays,
    )
    meas: list[PseudorangeMeasurement] = []
    for idx in indices:
        b_tx = rng.normal(0.0, relay_clock_sigma_km)
        meas.append(
            synthesize_pseudorange(
                true_position_icrs_m,
                relay_positions_mci_km[idx],
                b_tx,
                et_rx,
                rng,
                sigma_m,
                et0=et0,
                sat_id=f"LN{idx}",
            )
        )
    return meas
