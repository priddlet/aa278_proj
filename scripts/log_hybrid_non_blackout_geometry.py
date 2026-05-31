#!/usr/bin/env python3
"""Log GNSS PRN count and sidelobe PDOP on hybrid non-blackout epochs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.catalog import load_catalog
from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.propagation.propagator import LunarPropagator
from pulsar_nav.simulation.hybrid_run import run_hybrid_on_propagated
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.spice.kernels import load_kernels
from pulsar_nav.visibility.blackout import compute_visibility_timeline


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=26.4, help="Hours")
    p.add_argument("--step", type=float, default=120.0, help="Seconds")
    args = p.parse_args()

    load_kernels(load_gps_frames=True)
    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=args.duration * 3600.0, step_s=args.step)
    tl = compute_visibility_timeline(traj)
    res = run_hybrid_on_propagated(
        traj,
        load_catalog(),
        timeline=tl,
        policy=NavPolicy.HYBRID,
        rng=np.random.default_rng(0),
    )

    clear = [log for log in res.epoch_logs[1:] if not log.in_blackout and log.n_gnss > 0]
    n_gnss = np.array([log.n_gnss for log in clear], float)
    pdop = np.array([log.gnss_pdop for log in clear], float)
    finite = np.isfinite(pdop)

    print(f"ELFO {args.duration:.1f} hr — non-blackout GNSS epochs: {len(clear)}")
    print(f"  n_gnss: min={n_gnss.min():.0f} max={n_gnss.max():.0f} median={np.median(n_gnss):.0f}")
    if finite.any():
        p = pdop[finite]
        print(
            f"  sidelobe PDOP (≥4 PRNs): median={np.median(p):.1f} "
            f"p95={np.percentile(p, 95):.1f} max={p.max():.1f}"
        )
        print(f"  PDOP > 100: {100 * np.mean(p > 100):.1f}% of finite epochs")
    else:
        print("  sidelobe PDOP: no epochs with ≥4 PRNs")


if __name__ == "__main__":
    main()
