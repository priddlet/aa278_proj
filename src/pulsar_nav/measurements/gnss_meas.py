"""GNSS sidelobe pseudorange synthesis (GPS broadcast ephemeris)."""

from __future__ import annotations

import numpy as np

from pulsar_nav.ephemeris.broadcast import GpsBroadcastEphemeris, default_gps_ephemeris
from pulsar_nav.ephemeris.gps_posclk import get_gps_posclk_mci, iter_gps_prns
from pulsar_nav.filter.state import NavState
from pulsar_nav.measurements.pseudorange import (
    PseudorangeMeasurement,
    icrs_position_to_mci_km,
    predicted_pseudorange_m,
)
from pulsar_nav.spice.ephemeris import body_position_mci
from pulsar_nav.visibility.geometry import earth_occults_los, moon_blocks_los
from pulsar_nav.visibility.gnss import gnss_earth_visible

DEFAULT_URE_M = 15.0  # HW2 typical URE (m)
DEFAULT_MAX_GPS_PRNS = 4


def visible_gps_prns(
    spacecraft_mci_km: np.ndarray,
    et: float,
    ephem: GpsBroadcastEphemeris | None = None,
    *,
    min_elevation_deg: float = 5.0,
    prn_candidates: list[int] | None = None,
    earth_mci_km: np.ndarray | None = None,
    sidelobe_only: bool = True,
    max_prns: int | None = DEFAULT_MAX_GPS_PRNS,
) -> list[int]:
    """
    PRNs usable for lunar sidelobe pseudorange synthesis at ``et``.

    When ``sidelobe_only`` (default), requires Earth in view (same gate as
    ``gnss_earth_visible``) and Earth occultation of the SC→GPS line — the
    direct path is blocked and only sidelobe energy is assumed reachable.
    Without that filter, every broadcast PRN above a local elevation mask can
    appear visible (~31 sats), which is far above the 0–4 trackable regime
    seen in LuGRE and Capuano-style studies.

    For course replay data with visibility baked in, use HW2 ``gnss_measurements.pkl``.
    """
    ephem = ephem or default_gps_ephemeris()
    sc = np.asarray(spacecraft_mci_km, float)
    r_earth = (
        np.asarray(earth_mci_km, float)
        if earth_mci_km is not None
        else body_position_mci("EARTH", et)
    )
    if sidelobe_only and not gnss_earth_visible(sc, r_earth, min_elevation_deg=min_elevation_deg):
        return []

    candidates = prn_candidates or iter_gps_prns(ephem)
    visible: list[tuple[float, int]] = []

    for prn in candidates:
        if not ephem.has_prn(prn):
            continue
        try:
            r_gps, _ = get_gps_posclk_mci(et, prn, ephem)
        except (ValueError, KeyError):
            continue
        if moon_blocks_los(sc, r_gps):
            continue
        if sidelobe_only and not earth_occults_los(sc, r_gps, r_earth):
            continue
        if not sidelobe_only:
            from pulsar_nav.visibility.geometry import elevation_angle

            zenith = sc / np.linalg.norm(sc)
            el = elevation_angle(sc, r_gps, zenith_hat=zenith)
            if el < np.deg2rad(min_elevation_deg):
                continue
            visible.append((el, prn))
        else:
            # Prefer PRNs whose LOS grazes the Earth limb (smaller limb angle first).
            from pulsar_nav.visibility.geometry import earth_limb_angle_deg

            limb = earth_limb_angle_deg(sc, r_gps, r_earth)
            visible.append((limb, prn))

    visible.sort()
    prns = [prn for _, prn in visible]
    if max_prns is not None:
        prns = prns[:max_prns]
    return prns


def gnss_pseudoranges(
    true_position_icrs_m: np.ndarray,
    earth_mci_km: np.ndarray,
    et_rx: float,
    rng: np.random.Generator,
    *,
    sigma_m: float = DEFAULT_URE_M,
    et0: float | None = None,
    min_elevation_deg: float = 5.0,
    ephem: GpsBroadcastEphemeris | None = None,
    max_sats: int | None = DEFAULT_MAX_GPS_PRNS,
    sidelobe_only: bool = True,
) -> list[PseudorangeMeasurement]:
    """
    Synthesize pseudoranges from visible GPS satellites (broadcast ephemeris).

    Uses HW2-style ECEF broadcast + ITRF93->J2000 + Earth-Moon chain to MCI.
    Default sidelobe geometry yields 0–4 PRNs per epoch when Earth is in view.
    """
    del earth_mci_km  # Earth position enters via ``get_gps_posclk_mci`` / SPICE
    ephem = ephem or default_gps_ephemeris()
    r_sc_km = icrs_position_to_mci_km(true_position_icrs_m, et_rx)
    prns = visible_gps_prns(
        r_sc_km,
        et_rx,
        ephem,
        min_elevation_deg=min_elevation_deg,
        sidelobe_only=sidelobe_only,
        max_prns=max_sats,
    )

    truth = NavState.from_pv(true_position_icrs_m, np.zeros(3))
    meas: list[PseudorangeMeasurement] = []
    for prn in prns:
        r_tx, b_tx = get_gps_posclk_mci(et_rx, prn, ephem)
        rho_true = predicted_pseudorange_m(
            truth,
            r_tx,
            b_tx,
            et_rx,
            et0=et0,
            get_tx_position=lambda et, p=prn: get_gps_posclk_mci(et, p, ephem),
        )
        meas.append(
            PseudorangeMeasurement(
                tx_position_mci_km=r_tx,
                tx_clock_bias_km=b_tx,
                range_m=rho_true + rng.normal(0.0, sigma_m),
                sigma_m=sigma_m,
                sat_id=f"G{prn:02d}",
            )
        )
    return meas
