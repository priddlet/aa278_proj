#!/usr/bin/env python3
"""
Propagate a lunar orbit and visualize for verification.

Examples:
  python scripts/demo_propagate_visualize.py --preset elfo
  python scripts/demo_propagate_visualize.py --preset llo --duration 2 --save figures/llo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.propagation.propagator import LunarPropagator
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.spice.kernels import load_kernels
from pulsar_nav.visualization.orbit_plots import (
    plot_icrs_trajectory,
    plot_propagated_trajectory,
    save_propagation_figure,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Propagate and visualize lunar orbit")
    p.add_argument("--preset", choices=("elfo", "llo"), default="elfo")
    p.add_argument("--epoch", default="2026-01-15T12:00:00", help="UTC epoch")
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in hours (default: 30 for elfo, 2 for llo)",
    )
    p.add_argument("--step", type=float, default=60.0, help="Sample step (seconds)")
    p.add_argument("--j2", action="store_true", help="Include Moon J2 in dynamics")
    p.add_argument("--save", type=str, default=None, help="Save MCI figure to path")
    p.add_argument("--save-icrs", type=str, default=None, help="Save ICRS figure to path")
    p.add_argument("--no-show", action="store_true", help="Do not open plot window")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_kernels()

    duration_hr = args.duration
    if duration_hr is None:
        duration_hr = 30.0 if args.preset == "elfo" else 2.0
    duration_s = duration_hr * 3600.0
    step_s = args.step

    et0 = str_to_et(args.epoch)
    config = DynamicsConfig(include_earth=True, include_sun=True, include_moon_j2=args.j2)
    prop = LunarPropagator(et0, config=config)
    traj = prop.propagate_preset(args.preset, duration_s=duration_s, step_s=step_s)

    radii = np.linalg.norm(traj.position_mci_km, axis=1)
    print(f"Preset: {args.preset.upper()}  epoch: {args.epoch}")
    print(f"  steps={len(traj.t_rel_s)}  dt={step_s:.0f}s  duration={duration_hr:.1f} hr")
    print(f"  |r| min/mean/max (km): {radii.min():.1f} / {radii.mean():.1f} / {radii.max():.1f}")
    print(f"  altitude min/max (km): {radii.min() - 1737.4:.1f} / {radii.max() - 1737.4:.1f}")

    fig_mci = plot_propagated_trajectory(
        traj,
        preset=args.preset,
        title=f"{args.preset.upper()} — Moon-centered propagation",
    )
    fig_icrs = plot_icrs_trajectory(traj)

    out_dir = ROOT / "figures"
    save_mci = Path(args.save) if args.save else out_dir / f"{args.preset}_mci.png"
    save_icrs = Path(args.save_icrs) if args.save_icrs else out_dir / f"{args.preset}_icrs.png"

    p1 = save_propagation_figure(fig_mci, save_mci)
    p2 = save_propagation_figure(fig_icrs, save_icrs)
    print(f"  saved: {p1}")
    print(f"  saved: {p2}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
