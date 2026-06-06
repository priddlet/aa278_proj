#!/usr/bin/env python3
"""
Monte Carlo comparison: constant CWNA vs gravity-scaled Q (periapsis-aware).

Constant: fixed ``process_noise_accel`` (m^2/s^3).
Gravity-scaled: q_a(r) approx (scale  |  GM/r^2)^2 updated each predict from estimate radius.

Usage:
    python scripts/sweep_process_noise.py --filter-predict --trials 10
    python scripts/sweep_process_noise.py --dynamics-predict --trials 10
    python scripts/sweep_process_noise.py --filter-predict --quick
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.constants import DEFAULT_MC_DURATION_S
from pulsar_nav.filter.process_noise import gravity_scaled_q_accel
from pulsar_nav.propagation.dynamics import DynamicsConfig, GM_MOON, acceleration_mci
from pulsar_nav.propagation.propagator import LunarPropagator
from pulsar_nav.simulation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_dynamics_sigma_acc_sweep,
    run_gravity_q_scale_sweep,
    run_process_noise_sweep,
)
from pulsar_nav.simulation.predict_mode import PredictMode
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.spice.kernels import load_kernels


def gravity_reference_pacc(traj) -> float:
    """RMS norm(a_truth)^2 along arc (m^2/s^3) - scalar reference, not periapsis-varying."""
    cfg = DynamicsConfig(include_earth=True, include_sun=True)
    norms_m_s2: list[float] = []
    for i in range(len(traj.et)):
        r_km = traj.position_icrs_m[i] / 1000.0
        a_km_s2 = acceleration_mci(r_km, traj.et[i], cfg)
        norms_m_s2.append(float(np.linalg.norm(a_km_s2)) * 1000.0)
    a_rms = float(np.sqrt(np.mean(np.square(norms_m_s2))))
    return a_rms**2


def periapsis_q_range_km(traj, *, scale: float = 1.0) -> tuple[float, float]:
    """Min/max q_a (m^2/s^3) from Moon-centered truth radius over arc."""
    r_km = np.linalg.norm(traj.position_mci_km, axis=1)
    qa = [gravity_scaled_q_accel(float(r), scale=scale) for r in r_km]
    return float(min(qa)), float(max(qa))


def _format_sci(x: float) -> str:
    if x == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10**exp
    if abs(mant - 1.0) < 0.01:
        return f"1e{exp}"
    return f"{x:.3g}"


def _rows_from_result(
    result: MonteCarloResult,
    *,
    predict_label: str,
    q_mode: str,
    q_param: float,
    q_label: str,
    note: str,
    policies: tuple[NavPolicy, ...],
) -> list[dict]:
    rows: list[dict] = []
    for pol in policies:
        if pol not in result.by_policy:
            continue
        s = result.by_policy[pol]
        rows.append(
            {
                "predict_mode": predict_label,
                "q_mode": q_mode,
                "q_param": q_param,
                "q_label": q_label,
                "q_note": note,
                "policy": pol.value,
                "final_mean_km": round(s.final_mean_m / 1e3, 3),
                "final_p95_km": round(s.final_p95_m / 1e3, 3),
                "rms_km": round(s.rms_error_m / 1e3, 3),
                "blackout_mean_km": round(s.blackout_mean_m / 1e3, 3),
                "non_blackout_mean_km": round(s.non_blackout_mean_m / 1e3, 3),
                "nis_median_dof": round(s.nis_median_dof, 2)
                if math.isfinite(s.nis_median_dof)
                else "",
                "timing_mean_m": round(s.timing_mean_m, 2)
                if math.isfinite(s.timing_mean_m)
                else "",
                "n_trials": s.n_trials,
            }
        )
    return rows


def _markdown_table(
    rows: list[dict],
    *,
    title: str,
    predict_label: str,
    n_trials: int,
    gravity_ref: float,
    r_min_km: float,
    r_max_km: float,
    qa_peri: tuple[float, float],
    q_description: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"**EKF predict:** {predict_label}  |  **trials:** {n_trials}",
        "",
        "| Mode | Q setting | note | policy | final mu (km) | RMS (km) | "
        "blk mu (km) | non-blk mu (km) | med NIS/df | |b| mean (m) |",
        "|------|-----------|------|--------|--------------|----------|"
        "------------|----------------|------------|-------------|",
        "",
        q_description,
        f"Truth-radius range: **{r_min_km:.0f}-{r_max_km:.0f} km**. "
        f"Gravity-scaled q_a at truth radii (scale=1): **{_format_sci(qa_peri[0])}-{_format_sci(qa_peri[1])}** m^2/s^3. "
        f"Scalar RMS reference q_a approx **{_format_sci(gravity_ref)}** m^2/s^3.",
        "",
        "Target **med NIS/df approx 1** under filter CV / dynamics predict. "
        "XNAV-only: timing blank (MSP-only H). "
        "**|b| mean** = mean |b_rx-b_truth| (m) on GNSS/LunaNet pseudorange epochs.",
        "",
    ]
    for r in rows:
        nis = r["nis_median_dof"] if r["nis_median_dof"] != "" else "-"
        tmg = r["timing_mean_m"] if r["timing_mean_m"] != "" else "-"
        lines.append(
            f"| {r['q_mode']} | {r['q_label']} | {r['q_note']} | {r['policy']} | "
            f"{r['final_mean_km']:.2f} | {r['rms_km']:.2f} | {r['blackout_mean_km']:.2f} | "
            f"{r['non_blackout_mean_km']:.2f} | {nis} | {tmg} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Constant vs gravity-scaled Q sweep")
    p.add_argument("--preset", default="elfo")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--duration", type=float, default=None, help="hours")
    pred = p.add_mutually_exclusive_group()
    pred.add_argument("--filter-predict", action="store_true")
    pred.add_argument(
        "--dynamics-predict",
        action="store_true",
        help="Filter dynamics (RK45+STM) with HW2 sigma_acc_km sweep",
    )
    p.add_argument("--quick", action="store_true", help="3 trials; smaller sweep grids")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "presentation" / "tables",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_kernels(load_gps_frames=True)

    duration_s = (
        args.duration * 3600.0 if args.duration is not None else DEFAULT_MC_DURATION_S
    )
    if args.dynamics_predict:
        predict_label = "filter_dynamics"
        predict_title = "Filter dynamics predict (RK45+STM)"
    elif args.filter_predict:
        predict_label = "filter_predict"
        predict_title = "Filter CV predict"
    else:
        predict_label = "truth_velocity"
        predict_title = "Truth-velocity predict"
    policies = (
        (NavPolicy.HYBRID, NavPolicy.GNSS_ONLY)
        if args.quick
        else (NavPolicy.HYBRID, NavPolicy.GNSS_ONLY, NavPolicy.XNAV_ONLY)
    )
    n_trials = 3 if args.quick else args.trials

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset(args.preset, duration_s=duration_s, step_s=120.0)
    g_ref = gravity_reference_pacc(traj)
    r_km = np.linalg.norm(traj.position_mci_km, axis=1)
    r_min, r_max = float(np.min(r_km)), float(np.max(r_km))
    qa_peri = periapsis_q_range_km(traj, scale=1.0)

    if args.dynamics_predict:
        if args.quick:
            sigma_values = (1e-7, 1e-6, 1e-5)
        else:
            sigma_values = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
        pacc_values: tuple[float, ...] = ()
        grav_scales: tuple[float, ...] = ()
    elif args.quick:
        pacc_values = (1e-5, 1e-4, 1e-3)
        grav_scales = (0.5, 1.0, 2.0)
        sigma_values = ()
    else:
        pacc_values = (1e-6, 1e-5, 1e-4, 1e-3, g_ref / 10.0, g_ref, g_ref * 10.0)
        grav_scales = (0.25, 0.5, 0.65, 1.0, 1.5, 2.0)
        sigma_values = ()

    base = MonteCarloConfig(
        n_trials=n_trials,
        seed=42,
        preset=args.preset,
        duration_s=duration_s,
        step_s=120.0,
        predict_mode=PredictMode.DYNAMICS if args.dynamics_predict else None,
        use_truth_velocity_predict=not args.filter_predict and not args.dynamics_predict,
        use_dynamics_predict=args.dynamics_predict,
        policies=policies,
    )

    print(
        f"\nQ sweep - {args.preset}, {duration_s/3600:.1f} hr, {predict_label}, n={n_trials}"
    )
    print(f"Radius: {r_min:.0f}-{r_max:.0f} km  |  q_a(scale=1): {_format_sci(qa_peri[0])}-{_format_sci(qa_peri[1])} m^2/s^3")
    if args.dynamics_predict:
        print(
            "HW2 sigma_acc_km (km/s^2/sqrt(s)): "
            + ", ".join(_format_sci(s) for s in sigma_values)
        )
    else:
        print(f"Constant pacc: {', '.join(_format_sci(p) for p in pacc_values)}")
        print(f"Gravity scales: {', '.join(str(s) for s in grav_scales)}")
    print()

    rows: list[dict] = []

    if args.dynamics_predict:
        dyn_sweep = run_dynamics_sigma_acc_sweep(sigma_values, base_config=base)
        for sigma_km, result in sorted(dyn_sweep.items(), key=lambda kv: kv[0]):
            note = "MC default" if abs(sigma_km - 1e-6) / 1e-6 < 0.02 else ""
            rows.extend(
                _rows_from_result(
                    result,
                    predict_label=predict_label,
                    q_mode="hw2_sigma_acc",
                    q_param=sigma_km,
                    q_label=_format_sci(sigma_km),
                    note=note,
                    policies=policies,
                )
            )
    else:
        const_sweep = run_process_noise_sweep(pacc_values, base_config=base)
        for pacc, result in sorted(const_sweep.items(), key=lambda kv: kv[0]):
            note = "gravity-RMS ref" if abs(pacc - g_ref) / max(g_ref, 1e-30) < 0.02 else ""
            rows.extend(
                _rows_from_result(
                    result,
                    predict_label=predict_label,
                    q_mode="constant",
                    q_param=pacc,
                    q_label=_format_sci(pacc),
                    note=note,
                    policies=policies,
                )
            )

        grav_sweep = run_gravity_q_scale_sweep(grav_scales, base_config=base)
        for scale, result in sorted(grav_sweep.items(), key=lambda kv: kv[0]):
            q_lo, q_hi = periapsis_q_range_km(traj, scale=scale)
            note = f"q_a in [{_format_sci(q_lo)},{_format_sci(q_hi)}]"
            rows.extend(
                _rows_from_result(
                    result,
                    predict_label=predict_label,
                    q_mode="gravity_scaled",
                    q_param=scale,
                    q_label=f"scale={scale:g}",
                    note=note,
                    policies=policies,
                )
            )

    out_dir = args.out_dir / predict_label
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "q_sweep.csv"
    md_path = out_dir / "q_sweep.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if args.dynamics_predict:
        q_desc = (
            "**HW2 CWNA:** `dynamics_sigma_acc_km` (km/s^2/sqrt(s)) on RK45+STM predict; "
            "clock Q from HW2 RAFS PSDs. "
        )
    else:
        q_desc = (
            "**Constant CWNA:** fixed `process_noise_accel` (m^2/s^3). "
            "**Gravity-scaled:** q_a(r) approx (scale | GM/r^2)^2 each step (periapsis-aware). "
        )
    md = _markdown_table(
        rows,
        title=f"Process-noise sweep - {args.preset.upper()}",
        predict_label=predict_title,
        n_trials=n_trials,
        gravity_ref=g_ref,
        r_min_km=r_min,
        r_max_km=r_max,
        qa_peri=qa_peri,
        q_description=q_desc,
    )
    md_path.write_text(md, encoding="utf-8")

    print(md)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
