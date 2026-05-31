#!/usr/bin/env python3
"""
Hybrid navigation demo: GNSS / LunaNet / XNAV by visibility mode on ELFO truth.

Compares mode-switching hybrid filter vs XNAV-only baseline on the same arc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.catalog import load_catalog
from pulsar_nav.constants import DEFAULT_TOA_SIGMA_S
from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.propagation.propagator import LunarPropagator
from pulsar_nav.simulation.hybrid_run import run_hybrid_on_propagated
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.spice.kernels import load_kernels
from pulsar_nav.visibility.blackout import compute_visibility_timeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid nav on propagated ELFO truth")
    p.add_argument("--preset", choices=("elfo", "llo"), default="elfo")
    p.add_argument("--epoch", default="2026-01-15T12:00:00")
    p.add_argument("--duration", type=float, default=6.0, help="Hours")
    p.add_argument("--step", type=float, default=120.0, help="Seconds")
    p.add_argument("--offset-km", type=float, default=50.0)
    p.add_argument("--toa-us", type=float, default=100.0)
    p.add_argument("--gnss-sigma-m", type=float, default=15.0)
    p.add_argument("--lonet-sigma-m", type=float, default=15.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--save", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_kernels(load_gps_frames=True)

    et0 = str_to_et(args.epoch)
    duration_s = args.duration * 3600.0
    toa_sigma_s = args.toa_us * 1e-6

    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset(args.preset, duration_s=duration_s, step_s=args.step)
    timeline = compute_visibility_timeline(traj)

    pulsars = load_catalog()
    offset_m = args.offset_km * 1e3
    kw = dict(
        position_offset_m=offset_m,
        toa_sigma_s=toa_sigma_s,
        gnss_sigma_m=args.gnss_sigma_m,
        lonet_sigma_m=args.lonet_sigma_m,
        process_noise_accel=1e-4,
        use_truth_velocity_predict=True,
        position_sigma_m=150_000.0,
        rng=np.random.default_rng(42),
    )

    res_hybrid = run_hybrid_on_propagated(traj, pulsars, timeline=timeline, **kw)
    from pulsar_nav.simulation.xnav_run import run_xnav_on_propagated

    xnav_kw = {
        k: v
        for k, v in kw.items()
        if k not in ("gnss_sigma_m", "lonet_sigma_m", "rng")
    }
    res_xnav = run_xnav_on_propagated(traj, pulsars, rng=np.random.default_rng(43), **xnav_kw)

    n_gnss = sum(log.n_gnss for log in res_hybrid.epoch_logs)
    n_lonet = sum(log.n_lonet for log in res_hybrid.epoch_logs)
    n_pulsar = sum(log.n_pulsar for log in res_hybrid.epoch_logs)

    print(f"Hybrid navigation — {args.preset.upper()}  {args.duration:.1f} hr")
    print(f"  blackout fraction: {timeline.blackout_fraction * 100:.1f}%")
    print(f"  initial offset: {args.offset_km:.1f} km")
    print()
    print("Hybrid (XNAV always + GNSS/LunaNet when visible):")
    print(f"  measurements applied: GNSS={n_gnss}  LunaNet={n_lonet}  pulsar={n_pulsar}")
    print(f"  initial error: {res_hybrid.position_error_m[0]/1e3:.2f} km")
    print(f"  final error:   {res_hybrid.final_position_error_m/1e3:.2f} km")
    print(f"  mean error:    {res_hybrid.mean_position_error_m/1e3:.2f} km")
    print()
    print(f"XNAV-only baseline ({len(pulsars)} pulsars every epoch):")
    print(f"  initial error: {res_xnav.position_error_m[0]/1e3:.2f} km")
    print(f"  final error:   {res_xnav.final_position_rmse_m/1e3:.2f} km")
    print(f"  mean error:    {res_xnav.mean_position_error_m/1e3:.2f} km")

    try:
        from pulsar_nav.visualization.hybrid_plots import plot_hybrid_comparison, save_figure

        fig = plot_hybrid_comparison(
            res_hybrid,
            res_xnav,
            timeline,
            title=f"Hybrid vs XNAV-only — {args.preset.upper()}",
        )
        out = Path(args.save) if args.save else ROOT / "figures" / f"hybrid_{args.preset}_errors.png"
        save_figure(fig, out)
        print(f"\n  saved: {out}")

        if not args.no_show:
            import matplotlib.pyplot as plt

            plt.show()
    except ImportError:
        print("\n  (pip install -e '.[viz]' for plots)")


if __name__ == "__main__":
    main()
