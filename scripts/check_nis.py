#!/usr/bin/env python3
"""
Filter-consistency (NIS/df) diagnostic for the hybrid lunar EKF.

Target: **median(NIS/dof) ~ 1** with ~5% exceedance per epoch (chi-squared df = # measurements).
Uses constant CWNA ``process_noise_accel`` (default 1e-4 m²/s³ in Monte Carlo).

Usage:
    python scripts/check_nis.py --filter-predict
    python scripts/check_nis.py --filter-predict --pacc 1e-3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.propagation.propagator import LunarPropagator
from pulsar_nav.simulation.hybrid_run import offset_initial_position, run_hybrid_ekf
from pulsar_nav.simulation.monte_carlo import select_pulsars
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.spice.kernels import load_kernels
from pulsar_nav.visibility.blackout import compute_visibility_timeline

POLICIES = (NavPolicy.XNAV_ONLY, NavPolicy.GNSS_ONLY, NavPolicy.HYBRID)


def chi2_ppf95(dof: int) -> float:
    if dof <= 0:
        return float("nan")
    try:
        from scipy.stats import chi2

        return float(chi2.ppf(0.95, dof))
    except Exception:
        z = 1.6448536269514722
        t = 1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))
        return float(dof * t**3)


def summarize(nis: np.ndarray, dof: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    sel = mask & (dof > 0) & np.isfinite(nis)
    if not np.any(sel):
        return float("nan"), float("nan"), 0
    nis_s = nis[sel]
    dof_s = dof[sel]
    ratio = nis_s / dof_s
    thr = np.array([chi2_ppf95(int(d)) for d in dof_s])
    pct_over = 100.0 * float(np.mean(nis_s > thr))
    return float(np.median(ratio)), pct_over, int(sel.sum())


def run_policy(
    traj,
    timeline,
    pulsars,
    policy,
    *,
    trials: int,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    master = np.random.default_rng(0)
    nis_all, dof_all, blk_all, finals, rms_all = [], [], [], [], []
    for _ in range(trials):
        offset = master.uniform(cfg["offset_min"], cfg["offset_max"])
        init_pos = offset_initial_position(traj.position_icrs_m[0], offset, master)
        init_vel = traj.velocity_icrs_m_s[0].copy()
        res = run_hybrid_ekf(
            traj,
            timeline,
            pulsars,
            initial_position_m=init_pos,
            initial_velocity_m_s=init_vel,
            toa_sigma_s=cfg["toa_sigma_s"],
            gnss_sigma_m=cfg["gnss_sigma_m"],
            lonet_sigma_m=cfg["lonet_sigma_m"],
            process_noise_accel=cfg["process_noise_accel"],
            use_truth_velocity_predict=cfg["use_truth_velocity_predict"],
            rng=np.random.default_rng(master.integers(0, 2**63 - 1)),
            policy=policy,
        )
        nis_all.append(res.nis)
        dof_all.append(res.nis_dof)
        blk_all.append(np.array([s.in_blackout for s in timeline.samples], dtype=bool))
        finals.append(res.final_position_error_m / 1e3)
        rms_all.append(res.rms_position_error_m / 1e3)
    return (
        np.concatenate(nis_all),
        np.concatenate(dof_all),
        np.concatenate(blk_all),
        float(np.mean(finals)),
        float(np.mean(rms_all)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="elfo")
    ap.add_argument("--duration", type=float, default=26.4)
    ap.add_argument("--step", type=float, default=120.0)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--toa-us", type=float, default=1.0)
    ap.add_argument("--gnss-sigma", type=float, default=15.0)
    ap.add_argument("--pacc", type=float, default=1e-4, help="constant process_noise_accel (m²/s³)")
    ap.add_argument("--filter-predict", action="store_true")
    args = ap.parse_args()

    load_kernels(load_gps_frames=True)
    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset(args.preset, duration_s=args.duration * 3600.0, step_s=args.step)
    timeline = compute_visibility_timeline(traj)
    pulsars = select_pulsars(None)

    cfg = dict(
        toa_sigma_s=args.toa_us * 1e-6,
        gnss_sigma_m=args.gnss_sigma,
        lonet_sigma_m=15.0,
        process_noise_accel=args.pacc,
        use_truth_velocity_predict=not args.filter_predict,
        offset_min=30_000.0,
        offset_max=100_000.0,
    )

    predict_mode = "filter CV" if args.filter_predict else "truth-velocity"
    bf = 100.0 * float(np.mean([s.in_blackout for s in timeline.samples]))

    print(
        f"\nNIS/df — {args.preset} {args.duration}h, predict={predict_mode}, "
        f"Q_accel={args.pacc:g} m²/s³, blackout={bf:.1f}%, trials={args.trials}\n"
        "Target: median NIS/df ~ 1.0, %>chi2_95 ~ 5%\n"
    )

    print(f"--- constant Q (process_noise_accel={args.pacc:g}) ---")
    print(
        f"{'policy':10s} {'reg':12s} {'med NIS/df':>11s} {'%>chi2':>8s} "
        f"{'n':>6s} {'final km':>9s} {'RMS km':>8s}"
    )
    for policy in POLICIES:
        nis, dof, blk, final_km, rms_km = run_policy(
            traj, timeline, pulsars, policy, trials=args.trials, cfg=cfg
        )
        for label, mask in (
            ("non-blackout", ~blk),
            ("blackout", blk),
            ("all", np.ones_like(blk, dtype=bool)),
        ):
            med, pct, npts = summarize(nis, dof, mask)
            fin = f"{final_km:9.2f}" if label == "all" else " " * 9
            rms = f"{rms_km:8.2f}" if label == "all" else " " * 8
            print(
                f"{policy.value:10s} {label:12s} {med:11.2f} {pct:7.1f}% "
                f"{npts:6d} {fin} {rms}"
            )
    print()


if __name__ == "__main__":
    main()
