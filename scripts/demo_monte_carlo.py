#!/usr/bin/env python3
"""
Week 9 Monte Carlo: hybrid vs XNAV-only vs GNSS-only on ELFO truth.

Compares final-position error distributions to the LunaNet 13.43 m pitch target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.simulation.monte_carlo import (
    MonteCarloConfig,
    comparison_summary_table,
    run_monte_carlo,
    run_preset_comparison,
    run_pulsar_count_sweep,
    run_toa_noise_sweep,
)
from pulsar_nav.simulation.monte_carlo_export import (
    MonteCarloExportBundle,
    export_monte_carlo_xlsx,
)
from pulsar_nav.constants import DEFAULT_MC_DURATION_S
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.spice.kernels import load_kernels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo navigation campaign")
    p.add_argument("--preset", choices=("elfo", "elfo_nav", "llo"), default="elfo")
    p.add_argument(
        "--compare-elfo",
        action="store_true",
        help="Run Monte Carlo for elfo (HW2 frozen) and elfo_nav (argp+180 deg OP phasing)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Hours (default: 2× ELFO period ≈26.4 hr)",
    )
    p.add_argument("--step", type=float, default=120.0, help="Seconds")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--toa-us",
        type=float,
        default=1.0,
        help="TOA 1-sigma in microseconds (default 1 µs → ~300 m range noise)",
    )
    p.add_argument("--pulsars", type=int, default=None, help="MSP count (default: all 5)")
    p.add_argument("--quick", action="store_true", help="5 trials, skip sweeps")
    p.add_argument("--sweep-pulsars", action="store_true", help="Run 1/3/5 pulsar sweep")
    p.add_argument("--sweep-toa", action="store_true", help="Run TOA noise sweep")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument(
        "--excel",
        type=str,
        default=None,
        help="Excel output path (default: results/monte_carlo_results.xlsx)",
    )
    p.add_argument("--no-export", action="store_true", help="Skip Excel export")
    p.add_argument(
        "--no-truth-velocity",
        action="store_true",
        help="Propagate with CV EKF model instead of truth velocity (more realistic)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_kernels(load_gps_frames=True)

    n_trials = 5 if args.quick else args.trials
    duration_s = (
        args.duration * 3600.0 if args.duration is not None else DEFAULT_MC_DURATION_S
    )
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_bundle = MonteCarloExportBundle()
    comparison = None
    pulsar_sweep = None
    toa_sweep = None
    result = None
    cfg = None

    use_truth_vel = not args.no_truth_velocity
    if args.compare_elfo:
        cfg = MonteCarloConfig(
            n_trials=n_trials,
            seed=args.seed,
            duration_s=duration_s,
            step_s=args.step,
            toa_sigma_s=args.toa_us * 1e-6,
            n_pulsars=args.pulsars,
            use_truth_velocity_predict=use_truth_vel,
            policies=(
                NavPolicy.XNAV_ONLY,
                NavPolicy.GNSS_ONLY,
                NavPolicy.HYBRID,
            ),
        )
        print("Running ELFO comparison (science vs GNSS-friendly apoapsis)...")
        comparison = run_preset_comparison(("elfo", "elfo_nav"), cfg)
        export_bundle.comparison = comparison
        print()
        print(comparison_summary_table(comparison))
        result = comparison["elfo"]
    else:
        cfg = MonteCarloConfig(
            n_trials=n_trials,
            seed=args.seed,
            preset=args.preset,
            duration_s=duration_s,
            step_s=args.step,
            toa_sigma_s=args.toa_us * 1e-6,
            n_pulsars=args.pulsars,
            use_truth_velocity_predict=use_truth_vel,
            policies=(
                NavPolicy.XNAV_ONLY,
                NavPolicy.GNSS_ONLY,
                NavPolicy.HYBRID,
            ),
        )

        print("Running main Monte Carlo campaign...")
        result = run_monte_carlo(cfg)
        export_bundle.main = result
        print()
        print(result.summary_table())

    if args.sweep_pulsars and not args.quick:
        print("\nPulsar count sweep (hybrid + xnav)...")
        pulsar_sweep = run_pulsar_count_sweep(
            (1, 3, 5),
            base_config=MonteCarloConfig(
                n_trials=n_trials,
                seed=args.seed + 10,
                preset=cfg.preset if not args.compare_elfo else "elfo",
                duration_s=cfg.duration_s,
                step_s=cfg.step_s,
                toa_sigma_s=cfg.toa_sigma_s,
                use_truth_velocity_predict=use_truth_vel,
                policies=(NavPolicy.HYBRID, NavPolicy.XNAV_ONLY),
            ),
        )
        export_bundle.pulsar_sweep = pulsar_sweep
        for n, res in pulsar_sweep.items():
            h = res.by_policy[NavPolicy.HYBRID]
            print(
                f"  n={n} MSPs: hybrid final mean={h.final_mean_m/1e3:.2f} km  "
                f"p95={h.final_p95_m/1e3:.2f} km"
            )

    if args.sweep_toa and not args.quick:
        print("\nTOA noise sweep...")
        toa_sweep = run_toa_noise_sweep(
            (0.1, 1.0, 10.0),
            base_config=MonteCarloConfig(
                n_trials=n_trials,
                seed=args.seed + 20,
                preset=cfg.preset if not args.compare_elfo else "elfo",
                duration_s=cfg.duration_s,
                step_s=cfg.step_s,
                n_pulsars=args.pulsars,
                use_truth_velocity_predict=use_truth_vel,
            ),
        )
        export_bundle.toa_sweep = toa_sweep
        for sig, res in toa_sweep.items():
            h = res.by_policy[NavPolicy.HYBRID]
            print(f"  TOA σ={sig:.0f} µs: hybrid final mean={h.final_mean_m/1e3:.2f} km")

    if not args.no_export:
        excel_path = (
            Path(args.excel)
            if args.excel
            else ROOT / "results" / "monte_carlo_results.xlsx"
        )
        try:
            saved = export_monte_carlo_xlsx(excel_path, export_bundle)
            print(f"\n  Excel saved: {saved}")
        except ImportError as exc:
            print(f"\n  Excel export skipped: {exc}")

    try:
        from pulsar_nav.visualization.monte_carlo_plots import (
            apply_presentation_style,
            plot_final_error_boxplot,
            plot_final_error_cdf,
            plot_policy_metrics_bars,
            plot_pulsar_count_sweep,
            plot_pulsar_sweep_comparison,
            plot_toa_noise_sweep,
            save_figure,
        )

        apply_presentation_style()
        plot_preset = args.preset if not args.compare_elfo else "elfo"
        fig = plot_final_error_boxplot(
            result,
            title=f"Monte Carlo — {plot_preset.upper()} ({n_trials} trials)",
        )
        p_main = save_figure(fig, out_dir / f"mc_{plot_preset}_boxplot.png")
        print(f"\n  saved: {p_main}")
        save_figure(
            plot_policy_metrics_bars(result),
            out_dir / f"mc_{plot_preset}_policy_bars.png",
        )
        save_figure(
            plot_final_error_cdf(result),
            out_dir / f"mc_{plot_preset}_final_cdf.png",
        )
        print(f"  saved: mc_{plot_preset}_policy_bars.png, mc_{plot_preset}_final_cdf.png")

        if pulsar_sweep is not None:
            fig2 = plot_pulsar_count_sweep(pulsar_sweep, policy=NavPolicy.HYBRID)
            p2 = save_figure(fig2, out_dir / f"mc_{plot_preset}_pulsar_sweep.png")
            print(f"  saved: {p2}")
            save_figure(
                plot_pulsar_sweep_comparison(pulsar_sweep),
                out_dir / f"mc_{plot_preset}_pulsar_sweep_compare.png",
            )

        if toa_sweep is not None:
            save_figure(
                plot_toa_noise_sweep(toa_sweep),
                out_dir / f"mc_{plot_preset}_toa_sweep.png",
            )
            print(f"  saved: mc_{plot_preset}_toa_sweep.png")

        if not args.no_show:
            import matplotlib.pyplot as plt

            plt.show()
    except ImportError:
        print("\n  (pip install -e '.[viz]' for plots)")


if __name__ == "__main__":
    main()
