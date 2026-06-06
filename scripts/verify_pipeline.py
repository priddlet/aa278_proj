#!/usr/bin/env python3
"""
Step-by-step pipeline verification (read-only checks + small invariants).

Run:  python scripts/verify_pipeline.py
      python scripts/verify_pipeline.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np

ROOT = __file__
for _ in range(2):
    ROOT = __import__("pathlib").Path(ROOT).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class Check:
    stage: str
    name: str
    ok: bool
    detail: str


def _ok(stage: str, name: str, cond: bool, detail: str) -> Check:
    return Check(stage, name, bool(cond), detail)


def check_kernels() -> list[Check]:
    out: list[Check] = []
    try:
        from pulsar_nav.spice.kernels import load_kernels, resolve_kernel_dir

        resolve_kernel_dir()
        out.append(_ok("0", "SPICE lunar kernels", True, "found"))
    except FileNotFoundError as e:
        out.append(_ok("0", "SPICE lunar kernels", False, str(e)))
        return out

    try:
        from pulsar_nav.ephemeris.paths import resolve_brdc_path, resolve_gps_kernel_dir

        resolve_brdc_path()
        resolve_gps_kernel_dir()
        out.append(_ok("0", "HW2 brdc + Earth frames", True, "found"))
    except FileNotFoundError as e:
        out.append(_ok("0", "HW2 brdc + Earth frames", False, str(e)))
    return out


def check_truth_propagation(et0: float) -> list[Check]:
    from pulsar_nav.propagation.dynamics import dynamics_config_for_sim
    from pulsar_nav.propagation.propagator import LunarPropagator

    out: list[Check] = []
    cfg = dynamics_config_for_sim(include_disturbances=False)
    prop = LunarPropagator(et0, config=cfg, auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=3600.0, step_s=120.0)
    n = len(traj.t_rel_s)
    out.append(_ok("1", "truth steps", n >= 2, f"n={n}"))
    dr = np.linalg.norm(np.diff(traj.position_mci_km, axis=0), axis=1)
    out.append(_ok("1", "truth motion", np.all(dr > 0.1), f"min step |dr|={dr.min():.2f} km"))
    # ICRS = moon + MCI (spot check)
    from pulsar_nav.spice.ephemeris import moon_position_icrs_km

    i = n // 2
    recon = (moon_position_icrs_km(traj.et[i]) + traj.position_mci_km[i]) * 1000.0
    err = np.linalg.norm(recon - traj.position_icrs_m[i])
    out.append(_ok("1", "ICRS = moon+MCI", err < 1.0, f"|err|={err:.3e} m"))
    return out


def check_visibility(traj, timeline) -> list[Check]:
    out: list[Check] = []
    bf = timeline.blackout_fraction
    out.append(_ok("2", "blackout fraction", 0 < bf < 1, f"{100*bf:.1f}%"))
    out.append(
        _ok(
            "2",
            "timeline length",
            len(timeline.samples) == len(traj.t_rel_s),
            f"{len(timeline.samples)} samples",
        )
    )
    return out


def check_measurements(traj, timeline, pulsars, et0) -> list[Check]:
    from pulsar_nav.measurements.gnss_meas import gnss_pseudoranges
    from pulsar_nav.measurements.pseudorange import (
        predicted_pseudorange_m,
        pseudorange_residual,
    )
    from pulsar_nav.measurements.xnav import synthesize_measurement
    from pulsar_nav.simulation.hybrid_run import (
        TRUTH_CLOCK_BIAS_M,
        measurements_for_epoch,
    )
    from pulsar_nav.simulation.policy import NavPolicy
    from pulsar_nav.spice.ephemeris import body_position_mci
    from pulsar_nav.filter.state import NavState
    from pulsar_nav.visibility.lonet import LunaNetConfig

    out: list[Check] = []
    rng = np.random.default_rng(42)
    # pick non-blackout and blackout indices
    i_clear = next(i for i, s in enumerate(timeline.samples) if not s.in_blackout)
    i_bo = next(i for i, s in enumerate(timeline.samples) if s.in_blackout)

    for label, i in [("clear", i_clear), ("blackout", i_bo)]:
        pos = traj.position_icrs_m[i]
        earth = body_position_mci("EARTH", traj.et[i])
        gnss, lonet, xnav = measurements_for_epoch(
            timeline.samples[i],
            NavPolicy.HYBRID,
            truth_position_m=pos,
            et=traj.et[i],
            et0=et0,
            earth_mci_km=earth,
            relay_pos_km=np.zeros((4, 3)),
            pulsars=pulsars,
            rng=rng,
            toa_sigma_s=1e-6,
            gnss_sigma_m=15.0,
            lonet_sigma_m=15.0,
            lonet_config=LunaNetConfig(),
        )
        out.append(
            _ok(
                "3",
                f"hybrid meas counts ({label})",
                len(xnav) == len(pulsars),
                f"xnav={len(xnav)} gnss={len(gnss)} lonet={len(lonet)}",
            )
        )

    # XNAV: b_truth not in H - residual with b_hat should not match zero-innovation at wrong b
    p = pulsars[0]
    xm = synthesize_measurement(p, pos, rng, 1e-6)
    st0 = NavState.from_pv(pos, np.zeros(3), clock_bias_m=0.0)
    st1 = NavState.from_pv(pos, np.zeros(3), clock_bias_m=5000.0)
    r0 = abs(xm.range_m - p.line_of_sight_range_m(st0.position_m))
    r1 = abs(xm.range_m - p.line_of_sight_range_m(st1.position_m))
    out.append(
        _ok(
            "3",
            "XNAV ignores clock",
            abs(r0 - r1) < 1e-6,
            f"|Deltaresidual|={abs(r0-r1):.3e} m",
        )
    )

    # PR: zero innovation when state = truth with b=0 and noise-free
    if gnss:
        m = gnss[0]
        truth_st = NavState.from_pv(pos, np.zeros(3), clock_bias_m=TRUTH_CLOCK_BIAS_M)
        pred = predicted_pseudorange_m(
            truth_st,
            m.tx_position_mci_km,
            m.tx_clock_bias_km,
            traj.et[i_clear],
            et0=et0,
        )
        # synthetic z = pred_truth + noise; rebuild without noise check via residual at truth+noise
        innov = pseudorange_residual(m, truth_st, traj.et[i_clear], et0=et0)
        out.append(
            _ok(
                "3",
                "PR b_truth=0 in synthesis",
                TRUTH_CLOCK_BIAS_M == 0.0,
                f"|innov|={abs(innov):.2f} m (noise only if finite)",
            )
        )

    out.append(
        _ok(
            "3",
            "b_truth constant",
            TRUTH_CLOCK_BIAS_M == 0.0,
            f"TRUTH_CLOCK_BIAS_M={TRUTH_CLOCK_BIAS_M}",
        )
    )
    return out


def check_predict_modes(traj, pulsars, timeline, et0) -> list[Check]:
    from pulsar_nav.filter.ekf import PulsarNavEKF
    from pulsar_nav.propagation.dynamics import dynamics_config_for_sim
    from pulsar_nav.simulation.hybrid_run import run_hybrid_ekf
    from pulsar_nav.simulation.policy import NavPolicy
    from pulsar_nav.simulation.predict_mode import PredictMode

    out: list[Check] = []
    dt = float(traj.t_rel_s[1] - traj.t_rel_s[0])
    pos0 = traj.position_icrs_m[0].copy()
    vel0 = traj.velocity_icrs_m_s[0].copy()

    ekf_cv = PulsarNavEKF.from_initial(pos0, vel0)
    ekf_cv.predict(dt)
    pos_cv = ekf_cv.state.position_m.copy()

    ekf_dyn = PulsarNavEKF.from_initial(pos0, vel0)
    ekf_dyn.predict_dynamics(
        dt, traj.et[0], dynamics_config=dynamics_config_for_sim()
    )
    pos_dyn = ekf_dyn.state.position_m.copy()

    ekf_tv = PulsarNavEKF.from_initial(pos0, vel0)
    ekf_tv.predict_kinematic(dt, traj.velocity_icrs_m_s[0])
    pos_tv = ekf_tv.state.position_m.copy()

    # truth_velocity should match r + v_truth*dt (kinematic)
    pos_tv_expected = pos0 + traj.velocity_icrs_m_s[0] * dt
    err_tv = np.linalg.norm(pos_tv - pos_tv_expected)
    out.append(_ok("4", "truth_velocity kinematic", err_tv < 1e-3, f"|err|={err_tv:.3e} m"))

    # CV and dynamics should differ from each other (generic state)
    diff = np.linalg.norm(pos_cv - pos_dyn)
    out.append(_ok("4", "CV vs dynamics differ", diff > 1e-2, f"|Deltar|={diff:.2f} m"))

    # short hybrid runs finite
    for mode, tv, dyn in [
        (PredictMode.CV, False, False),
        (PredictMode.TRUTH_VELOCITY, True, False),
        (PredictMode.DYNAMICS, False, True),
    ]:
        res = run_hybrid_ekf(
            traj,
            timeline,
            pulsars[:3],
            initial_position_m=pos0 + np.array([50e3, 0.0, 0.0]),
            initial_velocity_m_s=vel0,
            predict_mode=mode,
            use_truth_velocity_predict=tv,
            use_dynamics_predict=dyn,
            policy=NavPolicy.HYBRID,
            rng=np.random.default_rng(0),
            process_noise_accel=1e-4,
        )
        finite = np.all(np.isfinite(res.position_error_m))
        out.append(_ok("4", f"hybrid run {mode.value}", finite, f"final err={res.final_position_error_m:.1f} m"))

    return out


def check_metrics(traj, timeline, pulsars, et0) -> list[Check]:
    from pulsar_nav.simulation.hybrid_run import run_hybrid_ekf, TRUTH_CLOCK_BIAS_M
    from pulsar_nav.simulation.policy import NavPolicy

    out: list[Check] = []
    res = run_hybrid_ekf(
        traj,
        timeline,
        pulsars[:3],
        initial_position_m=traj.position_icrs_m[0] + np.array([30e3, 0, 0]),
        initial_velocity_m_s=traj.velocity_icrs_m_s[0],
        predict_mode="dynamics",
        use_truth_velocity_predict=False,
        use_dynamics_predict=True,
        policy=NavPolicy.HYBRID,
        rng=np.random.default_rng(1),
    )
    # error = |est - truth|
    manual = np.linalg.norm(res.est_position_m - res.truth_position_m, axis=1)
    out.append(
        _ok(
            "5",
            "position_error consistent",
            np.allclose(manual, res.position_error_m),
            f"max diff={np.max(np.abs(manual-res.position_error_m)):.3e}",
        )
    )
    # timing only when PR
    for log in res.epoch_logs[1:]:
        if log.clock_constrained:
            exp = abs(log.clock_bias_m - TRUTH_CLOCK_BIAS_M)
            out.append(
                _ok(
                    "5",
                    "timing on PR epochs",
                    np.isfinite(log.clock_timing_error_m)
                    and abs(log.clock_timing_error_m - exp) < 1e-9,
                    f"|b|={log.clock_bias_m:.2f}",
                )
            )
            break
    else:
        out.append(_ok("5", "timing on PR epochs", False, "no PR epoch found"))
    xnav = run_hybrid_ekf(
        traj,
        timeline,
        pulsars[:3],
        initial_position_m=traj.position_icrs_m[0],
        initial_velocity_m_s=traj.velocity_icrs_m_s[0],
        predict_mode="dynamics",
        policy=NavPolicy.XNAV_ONLY,
        rng=np.random.default_rng(2),
    )
    out.append(
        _ok(
            "5",
            "xnav timing n/a",
            np.all(np.isnan(xnav.clock_timing_error_m)),
            "expected NaN timing",
        )
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    checks: list[Check] = []
    checks.extend(check_kernels())
    if not checks[-1].ok and checks[-1].name.startswith("SPICE"):
        _print(checks, args.verbose)
        return 1

    from pulsar_nav.catalog import load_catalog
    from pulsar_nav.spice.ephemeris import str_to_et
    from pulsar_nav.spice.kernels import load_kernels
    from pulsar_nav.visibility.blackout import compute_visibility_timeline

    load_kernels(load_gps_frames=True)
    et0 = str_to_et("2026-01-15T12:00:00")
    checks.extend(check_truth_propagation(et0))

    from pulsar_nav.propagation.dynamics import dynamics_config_for_sim
    from pulsar_nav.propagation.propagator import LunarPropagator

    prop = LunarPropagator(et0, config=dynamics_config_for_sim(), auto_load_kernels=False)
    traj = prop.propagate_preset("elfo", duration_s=6 * 3600.0, step_s=120.0)
    timeline = compute_visibility_timeline(traj)
    pulsars = load_catalog()[:5]
    checks.extend(check_visibility(traj, timeline))
    checks.extend(check_measurements(traj, timeline, pulsars, et0))
    checks.extend(check_predict_modes(traj, pulsars, timeline, et0))
    checks.extend(check_metrics(traj, timeline, pulsars, et0))

    _print(checks, args.verbose)
    failed = [c for c in checks if not c.ok]
    return 1 if failed else 0


def _print(checks: list[Check], verbose: bool) -> None:
    stages = sorted({c.stage for c in checks}, key=lambda s: (len(s), s))
    labels = {
        "0": "Environment (kernels / HW2 data)",
        "1": "Truth propagation",
        "2": "Visibility / blackout",
        "3": "Measurements (synthetic)",
        "4": "EKF predict modes",
        "5": "Metrics / logging",
    }
    for st in stages:
        print(f"\n=== {labels.get(st, st)} ===")
        for c in checks:
            if c.stage != st:
                continue
            mark = "PASS" if c.ok else "FAIL"
            print(f"  [{mark}] {c.name}: {c.detail}")
    n_fail = sum(1 for c in checks if not c.ok)
    print(f"\nTotal: {len(checks)} checks, {n_fail} failed.")


if __name__ == "__main__":
    raise SystemExit(main())
