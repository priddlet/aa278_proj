#!/usr/bin/env python3
"""Propagate an ELFO truth arc and report orbital geometry."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.propagation import LunarPropagator
from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.spice.kernels import load_kernels
from pulsar_nav.spice.ephemeris import str_to_et


def main() -> None:
    load_kernels()
    et0 = str_to_et("2026-01-15T12:00:00")
    # ~30 hr period ELFO: sample one orbit in 2 hr steps
    duration_s = 30.0 * 3600.0
    step_s = 120.0

    config = DynamicsConfig(include_earth=True, include_sun=True)
    prop = LunarPropagator(et0, config=config)
    traj = prop.propagate_preset("elfo", duration_s=duration_s, step_s=step_s)

    r = traj.position_mci_km
    radii = np.linalg.norm(r, axis=1)
    print("ELFO truth propagation (Moon-centered, km)")
    print(f"  Epoch UTC: 2026-01-15T12:00:00")
    print(f"  Steps: {len(traj.t_rel_s)}  dt={step_s:.0f} s  duration={duration_s/3600:.1f} hr")
    print(f"  Radius min/mean/max: {radii.min():.1f} / {radii.mean():.1f} / {radii.max():.1f} km")
    print(f"  Initial ICRS position (km): {traj.position_icrs_m[0] / 1e3}")
    print(f"  Final ICRS position (km):   {traj.position_icrs_m[-1] / 1e3}")

    try:
        from pulsar_nav.propagation.poliastro_backend import PoliastroLunarPropagator

        t_check = np.array([0.0, step_s])
        state0 = np.concatenate((traj.position_mci_km[0], traj.velocity_mci_km_s[0]))
        pa = PoliastroLunarPropagator(et0, config)
        traj_pa = pa.propagate(state0, t_check)
        diff = np.linalg.norm(traj_pa.position_mci_km[-1] - traj.position_mci_km[1])
        print(f"  poliastro vs scipy position diff @ {step_s}s: {diff:.6f} km")
    except ImportError:
        print("  (Install poliastro for cross-check: pip install -e ./poliastro)")

    try:
        from pulsar_nav.visualization.orbit_plots import (
            plot_icrs_trajectory,
            plot_propagated_trajectory,
            save_propagation_figure,
        )

        fig_dir = ROOT / "figures"
        fig_mci = plot_propagated_trajectory(traj, preset="elfo")
        fig_icrs = plot_icrs_trajectory(traj)
        save_propagation_figure(fig_mci, fig_dir / "elfo_mci.png")
        save_propagation_figure(fig_icrs, fig_dir / "elfo_icrs.png")
        print(f"  figures saved to {fig_dir}/")
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except Exception:
            pass
    except ImportError:
        print("  (Install viz extras for plots: pip install -e '.[viz]')")


if __name__ == "__main__":
    main()
