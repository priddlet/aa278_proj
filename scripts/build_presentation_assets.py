#!/usr/bin/env python3
"""
Build presentation figures, markdown tables, and Excel for AA278 slides.

Two navigation figure pipelines (plus shared geometry):

  figures/presentation/common/          - visibility & orbit (no EKF)
  figures/presentation/truth_velocity/  - MC + traces, predict with truth v
  figures/presentation/filter_predict/  - MC + traces, EKF CV predict only

Results: results/presentation_{truth_velocity,filter_predict}.md and .xlsx
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.constants import (
    DEFAULT_MC_DURATION_S,
    elfo_orbit_summary,
    elfo_orbital_period_s,
)
from pulsar_nav.simulation.monte_carlo import (
    MonteCarloConfig,
    MonteCarloResult,
    run_monte_carlo,
    run_pulsar_count_sweep,
    run_toa_noise_sweep,
)
from pulsar_nav.simulation.monte_carlo_export import (
    MonteCarloExportBundle,
    export_monte_carlo_xlsx,
)
from pulsar_nav.simulation.presentation_tables import (
    export_presentation_tables,
    write_tables_index,
)
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.simulation.presentation_runs import (
    collect_error_envelopes,
    propagate_truth_arc,
    run_representative_policy_runs,
)
from pulsar_nav.simulation.predict_mode import PredictMode
from pulsar_nav.spice.kernels import load_kernels


@dataclass(frozen=True)
class NavPipeline:
    """One EKF propagation mode for figures and tables."""

    slug: str
    label: str
    predict_mode: PredictMode
    short_note: str

    @property
    def use_truth_velocity_predict(self) -> bool:
        return self.predict_mode == PredictMode.TRUTH_VELOCITY

    @property
    def use_dynamics_predict(self) -> bool:
        return self.predict_mode == PredictMode.DYNAMICS


PIPELINES = {
    "truth_velocity": NavPipeline(
        slug="truth_velocity",
        label="Truth-velocity predict",
        predict_mode=PredictMode.TRUTH_VELOCITY,
        short_note="Oracle motion between measurements (default sim; optimistic absolute errors).",
    ),
    "filter_predict": NavPipeline(
        slug="filter_predict",
        label="Filter CV predict",
        predict_mode=PredictMode.CV,
        short_note="EKF constant-velocity predict only; more realistic dynamics stress.",
    ),
    "filter_dynamics": NavPipeline(
        slug="filter_dynamics",
        label="Filter dynamics predict",
        predict_mode=PredictMode.DYNAMICS,
        short_note="EKF RK45+STM predict (HW2 Tier A); sigma_acc km/s^2/sqrt(s) process noise.",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate presentation figures and tables")
    p.add_argument("--preset", choices=("elfo", "llo"), default="elfo")
    p.add_argument("--mc-trials", type=int, default=20, help="Full Monte Carlo trial count")
    p.add_argument("--envelope-trials", type=int, default=10, help="Trials for error envelope bands")
    p.add_argument("--visibility-hr", type=float, default=30.0, help="Hours for orbit/visibility plots")
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="MC arc hours (default: 2 x  ELFO period)",
    )
    p.add_argument("--toa-us", type=float, default=1.0)
    p.add_argument("--quick", action="store_true", help="5 MC trials, skip sweeps and envelope")
    p.add_argument(
        "--pipelines",
        choices=("both", "truth_velocity", "filter_predict", "filter_dynamics", "nav", "common"),
        default="both",
        help="Which figure sets to build (nav = all filter pipelines; common = geometry only)",
    )
    p.add_argument(
        "--presentation-root",
        type=str,
        default=None,
        help="Root under figures/ (default: figures/presentation)",
    )
    p.add_argument(
        "--disturbed-dynamics",
        action="store_true",
        help="Truth + filter dynamics: Moon J2 and HW2 SRP (gamma=C_R*A/m)",
    )
    return p.parse_args()


def _summary_markdown(
    result: MonteCarloResult,
    *,
    preset: str,
    pipeline: NavPipeline,
) -> str:
    predict_labels = {
        PredictMode.TRUTH_VELOCITY: "truth velocity",
        PredictMode.CV: "filter CV",
        PredictMode.DYNAMICS: "filter dynamics (MCI force model)",
    }
    predict = predict_labels[pipeline.predict_mode]
    lines = [
        f"## Monte Carlo - {preset.upper()} ({pipeline.label})",
        "",
        f"EKF predict: **{predict}**  |  "
        f"Arc: **{result.config.duration_s / 3600:.1f} hr**  |  "
        f"Trials: **{result.config.n_trials}**  |  "
        f"TOA sigma: **{result.config.toa_sigma_s * 1e6:.1f} us**",
        "",
        f"_{pipeline.short_note}_",
        "",
    ]
    if result.config.include_disturbances:
        lines.append(
            "Force model: **Moon J2 + SRP** (HW2 P3 gamma) on truth arc; "
            "filter dynamics predict uses the same `DynamicsConfig`."
        )
        lines.append("")
    if result.timeline:
        lines.append(f"Blackout fraction: **{100.0 * result.timeline.blackout_fraction:.1f}%**")
        lines.append("")
    import math

    lines.extend(
        [
            "_Timing: |b_rx-b_truth| (m) on GNSS/LunaNet pseudorange epochs; "
            "XNAV-only / MSP-only blackout: clock not in H._",
            "",
            "| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Steady mu (km) | Steady RMS (km) | Blackout mu (km) | Non-blackout mu (km) | |b| mean (m) | |b| p95 (m) |",
            "|--------|-----------------|----------------|----------|---------------|-----------------|-----------------|---------------------|------------|-----------|",
        ]
    )
    from pulsar_nav.simulation.monte_carlo import STEADY_STATE_ARC_FRACTION

    pct = int(STEADY_STATE_ARC_FRACTION * 100)
    lines.append(
        f"_Steady mu / Steady RMS: last {pct}% of arc epochs (excludes epoch-0 init spike)._"
    )
    lines.append("")
    for pol in result.config.policies:
        s = result.by_policy[pol]
        if pol.value == "xnav_only":
            t_mean, t_p95 = "-", "-"
        else:
            t_mean = f"{s.timing_mean_m:.2f}" if math.isfinite(s.timing_mean_m) else "-"
            t_p95 = f"{s.timing_p95_m:.2f}" if math.isfinite(s.timing_p95_m) else "-"
        lines.append(
            f"| {pol.value} | {s.final_mean_m / 1e3:.2f} | {s.final_p95_m / 1e3:.2f} | "
            f"{s.rms_error_m / 1e3:.2f} | {s.steady_state_mean_m / 1e3:.2f} | "
            f"{s.steady_state_rms_m / 1e3:.2f} | {s.blackout_mean_m / 1e3:.2f} | "
            f"{s.non_blackout_mean_m / 1e3:.2f} | {t_mean} | {t_p95} |"
        )
    lines.append("")
    return "\n".join(lines)


def _mc_config(
    args: argparse.Namespace,
    *,
    pipeline: NavPipeline,
    n_trials: int,
    duration_s: float,
) -> MonteCarloConfig:
    return MonteCarloConfig(
        n_trials=n_trials,
        seed=0,
        preset=args.preset,
        duration_s=duration_s,
        step_s=120.0,
        toa_sigma_s=args.toa_us * 1e-6,
        predict_mode=pipeline.predict_mode,
        use_truth_velocity_predict=pipeline.use_truth_velocity_predict,
        use_dynamics_predict=pipeline.use_dynamics_predict,
        include_disturbances=args.disturbed_dynamics,
        policies=(
            NavPolicy.XNAV_ONLY,
            NavPolicy.GNSS_ONLY,
            NavPolicy.HYBRID,
        ),
    )


def build_common_figures(
    common_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[str], str]:
    """Visibility and orbit geometry (independent of EKF predict mode)."""
    from pulsar_nav.visualization.orbit_plots import plot_propagated_trajectory, save_propagation_figure
    from pulsar_nav.visualization.visibility_plots import (
        plot_orbit_blackout_xy,
        plot_orbit_colored_by_blackout,
        plot_orbit_colored_by_policy,
        plot_policy_segment_timeline,
        plot_visibility_timeline,
        save_figure as save_vis_figure,
    )

    common_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    print("\n=== Common geometry figures ===")
    vis_traj, vis_tl = propagate_truth_arc(
        preset=args.preset,
        duration_s=args.visibility_hr * 3600.0,
        step_s=120.0,
        include_disturbances=args.disturbed_dynamics,
    )
    bf = 100.0 * vis_tl.blackout_fraction
    orbit_line = ""
    if args.preset == "elfo":
        t_hr = elfo_orbital_period_s() / 3600.0
        n_rev = args.visibility_hr / t_hr
        orbit_line = f"  |  {elfo_orbit_summary()}  |  {n_rev:.2f} rev"
    print(f"  {args.preset.upper()} {args.visibility_hr:.0f} hr sim{orbit_line} - blackout {bf:.1f}%")
    from pulsar_nav.visibility.gnss_coverage import gnss_sidelobe_coverage_stats

    gnss_cov = gnss_sidelobe_coverage_stats(vis_traj, vis_tl)
    print(f"  GNSS sidelobe trackability: {gnss_cov.summary_line()}")

    def _save(fig, name: str) -> None:
        save_vis_figure(fig, common_dir / name, dpi=200)
        saved.append(name)
        print(f"  saved {name}")

    vis_title = f"ELFO visibility - {args.visibility_hr:.0f}-hr simulation"
    if args.preset == "elfo":
        vis_title += f" ({elfo_orbit_summary()})"
    _save(
        plot_visibility_timeline(vis_traj, vis_tl, title=vis_title),
        f"{args.preset}_visibility_timeline.png",
    )

    for pol in (NavPolicy.XNAV_ONLY, NavPolicy.GNSS_ONLY, NavPolicy.HYBRID):
        _save(
            plot_orbit_colored_by_policy(vis_traj, vis_tl, pol),
            f"{args.preset}_orbit_{pol.value}.png",
        )
        _save(
            plot_policy_segment_timeline(vis_tl, pol),
            f"{args.preset}_segments_{pol.value}.png",
        )

    _save(
        plot_orbit_colored_by_blackout(
            vis_traj, vis_tl, title=f"{args.preset.upper()} - GNSS blackout (3D)"
        ),
        f"{args.preset}_orbit_blackout_3d.png",
    )
    _save(
        plot_orbit_blackout_xy(vis_traj, vis_tl),
        f"{args.preset}_orbit_blackout_xy.png",
    )
    fig_prop = plot_propagated_trajectory(
        vis_traj,
        preset=args.preset,
        title=f"{args.preset.upper()} truth orbit ({args.visibility_hr:.0f} hr)",
    )
    save_propagation_figure(fig_prop, common_dir / f"{args.preset}_truth_propagation.png", dpi=200)
    saved.append(f"{args.preset}_truth_propagation.png")
    print(f"  saved {args.preset}_truth_propagation.png")

    orbit_md = ""
    if args.preset == "elfo":
        t_hr = elfo_orbital_period_s() / 3600.0
        orbit_md = (
            f"{elfo_orbit_summary()}  |  **{args.visibility_hr / t_hr:.2f}** revolutions in arc  |  "
        )
    md = (
        f"## Common geometry - {args.preset.upper()}\n\n"
        f"Simulation arc: **{args.visibility_hr:.0f} hr**  |  {orbit_md}"
        f"Blackout: **{bf:.1f}%**  |  Windows: **{len(vis_tl.windows)}**\n\n"
        "_Not the LCRNS 30-h reference orbit (aapprox11 300 km, Tapprox30 h). "
        f"Monte Carlo blackout **~64%** uses **{DEFAULT_MC_DURATION_S / 3600:.1f} hr** (2 x  period).\n\n"
        f"**{gnss_cov.summary_line()}** (geometric non-blackout vs sidelobe PRNs)\n\n"
        "Orbit segment plots show **planned** policy phases; MC propagation plots use "
        "**measured** segments from the filter run.\n"
    )
    return saved, md


def build_nav_pipeline(
    nav_dir: Path,
    args: argparse.Namespace,
    pipeline: NavPipeline,
    *,
    n_mc: int,
    n_env: int,
    duration_s: float,
    results_dir: Path,
    tables_dir: Path,
) -> tuple[list[str], str]:
    """Monte Carlo figures for one predict mode."""
    from pulsar_nav.simulation.policy import NavPolicy
    from pulsar_nav.visualization.monte_carlo_plots import (
        plot_all_policies_envelope,
        plot_all_policies_propagation,
        plot_clock_timing_trace,
        plot_final_error_boxplot,
        plot_final_error_cdf,
        plot_policy_error_envelope,
        plot_policy_error_propagation,
        plot_policy_metrics_bars,
        plot_pulsar_sweep_comparison,
        plot_toa_noise_sweep,
        save_figure,
    )

    nav_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    print(f"\n=== Navigation pipeline: {pipeline.label} ===")
    cfg = _mc_config(args, pipeline=pipeline, n_trials=n_mc, duration_s=duration_s)

    mc_traj, mc_tl = propagate_truth_arc(
        preset=cfg.preset,
        epoch_utc=cfg.epoch_utc,
        duration_s=cfg.duration_s,
        step_s=cfg.step_s,
        dynamics_config=cfg.dynamics_config(),
    )

    print("Representative trial traces...")
    rep_runs, offset_m = run_representative_policy_runs(
        cfg, trial_id=0, traj=mc_traj, timeline=mc_tl
    )
    from pulsar_nav.visualization.presentation_style import policy_display_name

    for policy, run in rep_runs.items():
        fig = plot_policy_error_propagation(
            run,
            mc_tl,
            policy=policy,
            title=f"{policy_display_name(policy)} - position error",
        )
        fname = f"mc_{args.preset}_{policy.value}_propagation.png"
        save_figure(fig, nav_dir / fname)
        saved.append(fname)

        if policy in (NavPolicy.HYBRID, NavPolicy.GNSS_ONLY):
            fig_clk = plot_clock_timing_trace(
                run,
                title=f"{policy_display_name(policy)} - clock timing error",
            )
            if fig_clk is not None:
                clk_fname = f"mc_{args.preset}_{policy.value}_clock_timing.png"
                save_figure(fig_clk, nav_dir / clk_fname)
                saved.append(clk_fname)

    fig_cmp = plot_all_policies_propagation(
        rep_runs,
        mc_tl,
        title="Policy comparison - position error",
    )
    save_figure(fig_cmp, nav_dir / f"mc_{args.preset}_all_policies_propagation.png")
    saved.append(f"mc_{args.preset}_all_policies_propagation.png")

    if n_env > 0:
        print(f"Monte Carlo envelopes ({n_env} trials)...")
        envelopes = collect_error_envelopes(
            cfg, n_trials=n_env, traj=mc_traj, timeline=mc_tl
        )
        for policy, env in envelopes.items():
            fig = plot_policy_error_envelope(
                env,
                mc_tl,
                title=f"{policy_display_name(policy)} - Monte Carlo mean",
            )
            fname = f"mc_{args.preset}_{policy.value}_envelope.png"
            save_figure(fig, nav_dir / fname)
            saved.append(fname)
        save_figure(
            plot_all_policies_envelope(
                envelopes,
                mc_tl,
                title="Monte Carlo mean - all policies",
            ),
            nav_dir / f"mc_{args.preset}_all_policies_envelope.png",
        )
        saved.append(f"mc_{args.preset}_all_policies_envelope.png")

    md_parts = [
        f"# Presentation - {pipeline.label}",
        "",
        f"Directory: `figures/presentation/{pipeline.slug}/`",
        "",
        pipeline.short_note,
        "",
    ]

    export_bundle = MonteCarloExportBundle()
    print(f"Monte Carlo campaign ({n_mc} trials, {cfg.duration_s/3600:.1f} hr)...")
    result = run_monte_carlo(cfg, propagate_once=False)
    result.trajectory = mc_traj
    result.timeline = mc_tl
    export_bundle.main = result
    print(result.summary_table())
    md_parts.append(_summary_markdown(result, preset=args.preset, pipeline=pipeline))

    for plot_fn, fname in [
        (lambda: plot_final_error_boxplot(result, title="Final position error"), f"mc_{args.preset}_boxplot.png"),
        (lambda: plot_policy_metrics_bars(result), f"mc_{args.preset}_policy_bars.png"),
        (lambda: plot_final_error_cdf(result), f"mc_{args.preset}_final_cdf.png"),
    ]:
        save_figure(plot_fn(), nav_dir / fname)
        saved.append(fname)

    if not args.quick:
        print("Pulsar count sweep...")
        pulsar_sweep = run_pulsar_count_sweep(
            (1, 3, 5),
            base_config=replace(
                cfg,
                policies=(NavPolicy.HYBRID, NavPolicy.XNAV_ONLY),
            ),
        )
        export_bundle.pulsar_sweep = pulsar_sweep
        save_figure(
            plot_pulsar_sweep_comparison(pulsar_sweep),
            nav_dir / f"mc_{args.preset}_pulsar_sweep.png",
        )
        saved.append(f"mc_{args.preset}_pulsar_sweep.png")

        print("TOA noise sweep...")
        toa_sweep = run_toa_noise_sweep(
            (0.1, 1.0, 10.0),
            base_config=cfg,
        )
        export_bundle.toa_sweep = toa_sweep
        save_figure(
            plot_toa_noise_sweep(toa_sweep),
            nav_dir / f"mc_{args.preset}_toa_sweep.png",
        )
        saved.append(f"mc_{args.preset}_toa_sweep.png")

    tables_out = tables_dir / pipeline.slug
    table_files = export_presentation_tables(
        tables_out,
        export_bundle,
        preset=args.preset,
        predict_label=pipeline.label,
        predict_note=pipeline.short_note,
    )
    print(f"  Tables: {tables_out.relative_to(ROOT)}/ ({', '.join(table_files)})")

    try:
        xlsx = export_monte_carlo_xlsx(
            results_dir / f"monte_carlo_{pipeline.slug}.xlsx",
            export_bundle,
        )
        print(f"  Excel: {xlsx}")
        md_parts.append(f"\nSpreadsheet: `{xlsx.relative_to(ROOT)}`\n")
        md_parts.append(
            f"\nSlide tables: `presentation/tables/{pipeline.slug}/`\n"
        )
    except ImportError as exc:
        print(f"  Excel skipped: {exc}")

    md_parts.extend(["## Figures", "", "| File | Description |", "|------|-------------|"])
    for fname in sorted(saved):
        md_parts.append(f"| `{fname}` | MC / propagation ({pipeline.slug}) |")

    summary_path = results_dir / f"presentation_{pipeline.slug}.md"
    summary_path.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"  {len(saved)} figures in {nav_dir}/")

    return saved, "\n".join(md_parts[:6])


def main() -> None:
    args = parse_args()
    load_kernels(load_gps_frames=True)

    from pulsar_nav.visualization.monte_carlo_plots import apply_presentation_style

    apply_presentation_style()

    n_mc = 5 if args.quick else args.mc_trials
    n_env = 0 if args.quick else args.envelope_trials
    duration_s = (
        args.duration * 3600.0 if args.duration is not None else DEFAULT_MC_DURATION_S
    )

    pres_root = Path(args.presentation_root) if args.presentation_root else ROOT / "figures" / "presentation"
    common_dir = pres_root / "common"
    results_dir = ROOT / "results"
    tables_dir = ROOT / "presentation" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    index_pipeline_rows: list[tuple[str, str, str]] = []

    index_lines = [
        "# Presentation figure index",
        "",
        f"Generated by `scripts/build_presentation_assets.py` "
        f"({'quick' if args.quick else 'full'} mode, pipelines={args.pipelines}).",
        "",
        "| Pipeline | Directory | EKF time update | Use for slides |",
        "|----------|-----------|-----------------|----------------|",
        f"| Shared geometry | `{common_dir.relative_to(ROOT)}` | - | Blackout, policy segments, truth orbit |",
    ]

    run_common = args.pipelines in ("both", "common")
    run_nav = args.pipelines in (
        "both",
        "truth_velocity",
        "filter_predict",
        "filter_dynamics",
        "nav",
    )

    if run_common:
        build_common_figures(common_dir, args)

    if run_nav:
        pipelines_to_run: list[NavPipeline] = []
        if args.pipelines in ("both", "truth_velocity", "nav"):
            pipelines_to_run.append(PIPELINES["truth_velocity"])
        if args.pipelines in ("both", "filter_predict", "nav"):
            pipelines_to_run.append(PIPELINES["filter_predict"])
        if args.pipelines in ("both", "filter_dynamics", "nav"):
            pipelines_to_run.append(PIPELINES["filter_dynamics"])

        predict_row_labels = {
            PredictMode.TRUTH_VELOCITY: "Truth velocity between updates",
            PredictMode.CV: "Filter CV predict only",
            PredictMode.DYNAMICS: "Filter dynamics (RK45 + STM, HW2 Q)",
        }

        for pipeline in pipelines_to_run:
            predict_row = predict_row_labels[pipeline.predict_mode]
            index_lines.append(
                f"| {pipeline.label} | `{pres_root / pipeline.slug}` | "
                f"{predict_row} | "
                f"MC stats: `results/presentation_{pipeline.slug}.md`  |  "
                f"tables: `presentation/tables/{pipeline.slug}/` |"
            )
            index_pipeline_rows.append((pipeline.slug, pipeline.label, predict_row))
            build_nav_pipeline(
                pres_root / pipeline.slug,
                args,
                pipeline,
                n_mc=n_mc,
                n_env=n_env,
                duration_s=duration_s,
                results_dir=results_dir,
                tables_dir=tables_dir,
            )

        if index_pipeline_rows:
            write_tables_index(
                tables_dir / "INDEX.md",
                preset=args.preset,
                pipelines=index_pipeline_rows,
            )
            print(f"\nWrote {tables_dir / 'INDEX.md'}")

    index_path = results_dir / "presentation_INDEX.md"
    index_lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            "# Both pipelines + common geometry (full)",
            "python scripts/build_presentation_assets.py",
            "",
            "# Filter dynamics only (more realistic errors)",
            "python scripts/build_presentation_assets.py --pipelines filter_predict",
            "",
            "# Geometry only",
            "python scripts/build_presentation_assets.py --pipelines common",
            "```",
        ]
    )
    index_path.write_text("\n".join(index_lines), encoding="utf-8")
    print(f"\nWrote {index_path}")


if __name__ == "__main__":
    main()
