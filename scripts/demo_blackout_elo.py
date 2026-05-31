#!/usr/bin/env python3
"""
Propagate ELFO and label GNSS / LunaNet visibility + far-side blackout windows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.propagation.propagator import LunarPropagator
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.spice.kernels import load_kernels
from pulsar_nav.visibility import compute_visibility_timeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ELFO visibility and blackout analysis")
    p.add_argument("--preset", choices=("elfo", "llo"), default="elfo")
    p.add_argument("--epoch", default="2026-01-15T12:00:00")
    p.add_argument("--duration", type=float, default=30.0, help="Hours (one ELFO period)")
    p.add_argument("--step", type=float, default=120.0)
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--save-dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_kernels()

    et0 = str_to_et(args.epoch)
    prop = LunarPropagator(et0, config=DynamicsConfig())
    traj = prop.propagate_preset(
        args.preset, duration_s=args.duration * 3600.0, step_s=args.step
    )
    timeline = compute_visibility_timeline(traj)

    print(f"Visibility analysis — {args.preset.upper()}")
    print(f"  duration: {args.duration:.1f} hr   step: {args.step:.0f} s")
    print(f"  blackout fraction: {100.0 * timeline.blackout_fraction:.1f}%")
    print(f"  total blackout time: {timeline.total_blackout_s / 3600.0:.2f} hr")
    print(f"  number of blackout windows: {len(timeline.windows)}")
    for j, w in enumerate(timeline.windows[:8]):
        print(
            f"    window {j+1}: {w.start_s/3600:.2f}–{w.end_s/3600:.2f} hr "
            f"({w.duration_hr:.2f} hr)"
        )
    if len(timeline.windows) > 8:
        print(f"    ... and {len(timeline.windows) - 8} more")

    try:
        from pulsar_nav.visualization.visibility_plots import (
            plot_orbit_colored_by_mode,
            plot_visibility_timeline,
            save_figure,
        )

        fig_dir = Path(args.save_dir) if args.save_dir else ROOT / "figures"
        save_figure(
            plot_visibility_timeline(traj, timeline),
            fig_dir / f"{args.preset}_visibility.png",
        )
        save_figure(
            plot_orbit_colored_by_mode(traj, timeline),
            fig_dir / f"{args.preset}_orbit_nav_mode.png",
        )
        print(f"\n  saved figures to {fig_dir}/")

        if not args.no_show:
            import matplotlib.pyplot as plt

            plt.show()
    except ImportError:
        print("\n  (pip install -e '.[viz]' for plots)")


if __name__ == "__main__":
    main()
