#!/usr/bin/env python3
"""
XNAV EKF on a propagated ELFO truth arc (AA278 Week 7 integration).

Compares single-pulsar vs multi-pulsar filtering on the same truth trajectory.

Uses kinematic predict with truth velocity (simulation aid until ODTS dynamics
are in the filter). Measurements are applied in batch per epoch (all pulsars).
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
from pulsar_nav.simulation.xnav_run import run_xnav_on_propagated
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.spice.kernels import load_kernels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XNAV EKF on propagated lunar truth")
    p.add_argument("--preset", choices=("elfo", "llo"), default="elfo")
    p.add_argument("--epoch", default="2026-01-15T12:00:00")
    p.add_argument("--duration", type=float, default=6.0, help="Hours")
    p.add_argument("--step", type=float, default=120.0, help="Seconds")
    p.add_argument("--offset-km", type=float, default=50.0, help="Initial position error")
    p.add_argument("--toa-us", type=float, default=100.0, help="TOA 1-sigma in microseconds")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--save", type=str, default=None, help="Save error plot path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_kernels()

    et0 = str_to_et(args.epoch)
    duration_s = args.duration * 3600.0
    toa_sigma_s = args.toa_us * 1e-6

    prop = LunarPropagator(et0, config=DynamicsConfig())
    traj = prop.propagate_preset(args.preset, duration_s=duration_s, step_s=args.step)
    samples = traj.samples()

    all_pulsars = load_catalog()
    single = load_catalog(["B1937+21"])

    offset_m = args.offset_km * 1e3
    ekf_kw = dict(
        toa_sigma_s=toa_sigma_s,
        process_noise_accel=1e-4,
        position_sigma_m=150_000.0,
        use_truth_velocity_predict=True,
    )

    res_multi = run_xnav_on_propagated(
        traj,
        all_pulsars,
        position_offset_m=offset_m,
        unobservable_offset=False,
        **ekf_kw,
    )
    res_single = run_xnav_on_propagated(
        traj,
        single,
        position_offset_m=offset_m,
        unobservable_offset=True,
        rng=np.random.default_rng(2),
        **ekf_kw,
    )

    print(f"Truth: {args.preset.upper()}  {args.duration:.1f} hr  step={args.step:.0f} s")
    print(f"  samples: {len(samples)}  TOA sigma: {args.toa_us:.1f} µs")
    print(f"  initial position offset: {args.offset_km:.1f} km")
    print()
    print("Single pulsar (B1937+21), offset ⊥ LOS:")
    print(f"  initial 3D error: {res_single.position_error_m[0]/1e3:.2f} km")
    print(f"  final 3D error:   {res_single.final_position_rmse_m/1e3:.2f} km")
    print(f"  final LOS error:  {res_single.los_error_m[-1]/1e3:.2f} km")
    print()
    print(f"Multi pulsar ({len(all_pulsars)} MSPs):")
    print(f"  initial 3D error: {res_multi.position_error_m[0]/1e3:.2f} km")
    print(f"  final 3D error:   {res_multi.final_position_rmse_m/1e3:.2f} km")
    print(f"  mean 3D error:    {res_multi.mean_position_error_m/1e3:.2f} km")

    try:
        from pulsar_nav.visualization.nav_plots import (
            plot_xnav_errors,
            plot_xyz_errors,
            save_figure,
        )

        fig = plot_xnav_errors(
            {
                f"1 pulsar ({single[0].name})": res_single,
                f"{len(all_pulsars)} pulsars": res_multi,
            },
            title=f"XNAV on propagated {args.preset.upper()} truth",
        )
        out = Path(args.save) if args.save else ROOT / "figures" / f"xnav_{args.preset}_errors.png"
        save_figure(fig, out)
        print(f"\n  saved: {out}")

        if not args.no_show:
            import matplotlib.pyplot as plt

            plt.show()
    except ImportError:
        print("\n  (pip install -e '.[viz]' for error plots)")


if __name__ == "__main__":
    main()
