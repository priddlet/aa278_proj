#!/usr/bin/env python3
"""
Compare simulated visibility against AA278 / LuGRE / LunaNet lecture anchors.

Run after SPICE + brdc kernels are installed:

    python scripts/validate_visibility_anchors.py --preset elfo --duration 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.measurements.gnss_meas import visible_gps_prns
from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.propagation.propagator import LunarPropagator
from pulsar_nav.spice.ephemeris import body_position_mci, str_to_et
from pulsar_nav.spice.kernels import load_kernels
from pulsar_nav.visibility.blackout import compute_visibility_timeline
from pulsar_nav.visibility.gdop import position_dop_from_los
from pulsar_nav.visibility.gnss import gnss_earth_visible
from pulsar_nav.visibility.lonet import (
    LunaNetConfig,
    construct_walker_constellation,
    lonet_visibility,
    propagate_constellation_mci,
)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze(
    preset: str,
    duration_h: float,
    step_s: float,
    epoch: str,
) -> None:
    load_kernels(load_gps_frames=True)
    et0 = str_to_et(epoch)
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset(preset, duration_s=duration_h * 3600.0, step_s=step_s)
    tl = compute_visibility_timeline(traj)

    radii = np.linalg.norm(traj.position_mci_km, axis=1)
    earth_el = np.array([s.earth_elevation_deg for s in tl.samples])
    geo_vis = np.array([s.gnss_visible for s in tl.samples])
    prn_counts = []
    gps_pdops: list[float] = []

    for i in range(len(traj.t_rel_s)):
        r_sc = traj.position_mci_km[i]
        et = traj.et[i]
        r_earth = body_position_mci("EARTH", et)
        prns = visible_gps_prns(r_sc, et, earth_mci_km=r_earth)
        prn_counts.append(len(prns))
        if len(prns) >= 4:
            from pulsar_nav.ephemeris.gps_posclk import get_gps_posclk_mci

            los = []
            for prn in prns[:4]:
                r_gps, _ = get_gps_posclk_mci(et, prn)
                dr = r_gps - r_sc
                los.append(dr / np.linalg.norm(dr))
            gps_pdops.append(position_dop_from_los(np.vstack(los)))

    prn_counts = np.array(prn_counts)
    lonet_cfg = LunaNetConfig(n_sats=5, n_planes=5)
    coes = construct_walker_constellation(
        lonet_cfg.sma_km,
        lonet_cfg.eccentricity,
        lonet_cfg.inclination_rad,
        lonet_cfg.argp_rad,
        lonet_cfg.walker_f,
        lonet_cfg.n_sats,
        lonet_cfg.n_planes,
    )
    relay = propagate_constellation_mci(coes, traj.t_rel_s)
    lonet_counts = []
    lonet_pdop_lt6 = 0
    lonet_visible_epochs = 0
    for i in range(len(traj.t_rel_s)):
        r_sc = traj.position_mci_km[i]
        vis, n, _ = lonet_visibility(r_sc, relay[:, i, :])
        lonet_counts.append(n)
        if n >= 4:
            lonet_visible_epochs += 1
            los = []
            for r_sat in relay[:, i, :]:
                dr = r_sat - r_sc
                los.append(dr / np.linalg.norm(dr))
            if position_dop_from_los(np.vstack(los[:4])) < 6.0:
                lonet_pdop_lt6 += 1

    lonet_counts = np.array(lonet_counts)

    print(f"\n{'=' * 64}")
    print(f"Visibility anchors — {preset}, {duration_h:.1f} hr, epoch {epoch}")
    print(f"{'=' * 64}")
    print(f"Perigee / apoapsis radius (km): {radii.min():.0f} / {radii.max():.0f}")
    print(f"corr(radius, Earth elev): {_corr(radii, earth_el):+.3f}")
    print("  (negative ⇒ higher altitude ≈ lower Earth elev / far-side apo dwell)")
    print()
    print("GNSS geometric (Earth elev mask, timeline blackout):")
    print(f"  Earth-visible fraction: {100 * geo_vis.mean():.1f}%")
    print(f"  Blackout fraction:      {100 * tl.blackout_fraction:.1f}%")
    print()
    print("GNSS sidelobe PRNs (far-side clear LOS, near-limb annulus, cap 4):")
    print(
        f"  Epochs with ≥1 PRN: {100 * np.mean(prn_counts > 0):.1f}%  "
        f"(lecture: sparse, often 0)"
    )
    print(
        f"  PRN count mean={prn_counts.mean():2f} max={prn_counts.max()} "
        f"p95={np.percentile(prn_counts, 95):.0f}  (lecture: 0–4)"
    )
    if gps_pdops:
        g = np.array(gps_pdops)
        print(f"  PDOP when ≥4 sidelobe PRNs: median={np.median(g):.0f} (expect large)")
    print()
    print("LunaNet 5-relay Walker (8000 km, lecture ~40–55% GDOP<6 anchor):")
    print(f"  Epochs with ≥1 relay: {100 * np.mean(lonet_counts > 0):.1f}%")
    print(f"  Relay count mean={lonet_counts.mean():1f} max={lonet_counts.max()} (lecture: 3–5)")
    if lonet_visible_epochs:
        print(
            f"  GDOP<6 when ≥4 relays: "
            f"{100 * lonet_pdop_lt6 / lonet_visible_epochs:.1f}% of those epochs"
        )
    print()
    print("Sanity checks:")
    ok_prn = prn_counts.max() <= 4
    ok_band = prn_counts.mean() <= 4.0
    print(f"  [{'PASS' if ok_prn else 'FAIL'}] max PRN count ≤ 4")
    print(f"  [{'PASS' if ok_band else 'FAIL'}] mean PRN count ≤ 4 (lecture 0–4 band)")


def main() -> None:
    p = argparse.ArgumentParser(description="Validate visibility vs lecture anchors")
    p.add_argument("--preset", default="elfo", choices=("elfo", "elfo_nav", "llo"))
    p.add_argument("--duration", type=float, default=30.0, help="Hours")
    p.add_argument("--step", type=float, default=120.0, help="Seconds")
    p.add_argument("--epoch", default="2026-01-15 12:00:00")
    args = p.parse_args()
    analyze(args.preset, args.duration, args.step, args.epoch)


if __name__ == "__main__":
    main()
