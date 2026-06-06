"""Unified presentation index and narrative tables (figures + tables together)."""

from __future__ import annotations

import math
from pathlib import Path

from pulsar_nav.constants import DEFAULT_MC_DURATION_S, elfo_orbit_summary
from pulsar_nav.simulation.monte_carlo import (
    LUNANET_REQUIREMENT_M,
    STEADY_STATE_ARC_FRACTION,
    MonteCarloResult,
    PolicyStats,
)
from pulsar_nav.simulation.policy import NavPolicy

PRIMARY_PIPELINE = "filter_dynamics"

FILTER_DYNAMICS_ASSUMPTIONS = """
## Model assumptions (filter dynamics - primary campaign)

| Item | Value |
|------|--------|
| **Truth orbit** | HW2 ELFO case 1: a = 6541.4 km, e = 0.6, **T approx 13.2 h** (not LCRNS 30 h) |
| **Truth dynamics** | Moon + Earth + Sun indirect (HW2 P2), SPICE DE440 |
| **EKF predict** | MCI RK45 + analytic STM; HW2 CWNA Q, `dynamics_sigma_acc_km = 1e-6` km/s^2/sqrt(s) |
| **Measurements** | Synthetic from **truth position** + noise (optimistic sensing) |
| **XNAV** | All 5 SEXTANT MSPs, sigma_TOA = **1 us** (~300 m range sigma), linear `n_hat dot r` |
| **GNSS / LunaNet** | Sidelobe PRNs, sigma = 15 m; LunaNet supplemental in blackout when relay visible |
| **MC arc** | **26.4 h** (2x orbital period), step **120 s**, 20 trials |
| **Blackout** | Earth < 5 deg elev.; **~65.7% one orbit**, **~64.1%** on MC arc |
| **Position metrics** | Report **Steady RMS** (last 10% of arc); full-arc RMS includes epoch-0 spike |
| **Timing metrics** | |b_rx - b_truth| (m) on GNSS/LunaNet pseudorange epochs only - not pulsar TOA |
""".strip()

COMMON_FIGURES: list[tuple[str, str]] = [
    ("elfo_blackout_fraction_specs.png", "Blackout % - one orbit (T approx 13.2 h) vs 30 h sim vs MC arc"),
    ("elfo_orbit_blackout_3d_one_orbit.png", "Orbit colored by blackout - one revolution"),
    ("elfo_orbit_blackout_3d.png", "Orbit colored by blackout - 30 h visibility arc"),
    ("elfo_orbit_blackout_xy.png", "Blackout map (Moon xy)"),
    ("elfo_truth_propagation.png", "Truth trajectory (30 h arc)"),
    ("elfo_visibility_one_orbit.png", "Visibility timeline - one orbit"),
    ("elfo_visibility_timeline.png", "Visibility timeline - 30 h (approx 2.3 rev)"),
    ("elfo_orbit_{policy}.png", "Orbit colored by planned policy phase"),
    ("elfo_segments_{policy}.png", "Policy segment timeline"),
    ("predict_mode_comparison_{policy}.png", "Predict-mode bar charts (hybrid / gnss_only / xnav_only)"),
]

FILTER_DYNAMICS_FIGURES: list[tuple[str, str]] = [
    ("mc_elfo_all_policies_propagation.png", "Position error propagation - all policies"),
    ("mc_elfo_{policy}_propagation.png", "Position error propagation per policy"),
    ("mc_elfo_{policy}_envelope.png", "MC error envelope (mean +/- band)"),
    ("mc_elfo_all_policies_envelope.png", "MC envelope - all policies"),
    ("mc_elfo_hybrid_clock_timing.png", "Clock timing error trace (hybrid)"),
    ("mc_elfo_gnss_only_clock_timing.png", "Clock timing trace (gnss_only)"),
    ("mc_elfo_policy_bars.png", "Bar chart - final / blackout / steady metrics"),
    ("mc_elfo_boxplot.png", "Boxplot - final position error by policy"),
    ("mc_elfo_final_cdf.png", "CDF - final position error"),
    ("mc_elfo_pulsar_sweep.png", "Pulsar count sweep figure"),
    ("mc_elfo_toa_sweep.png", "TOA noise sweep figure"),
    ("mc_elfo_q_diagnostics.png", "Q diagnostics - median NIS/df vs sigma_acc"),
]

FILTER_DYNAMICS_TABLES: list[tuple[str, str]] = [
    ("common/pulsar_catalog.md", "Pulsar catalog - f0 (Hz), nav vs timing roles"),
    ("common/blackout_specs.md", "Blackout specs - one-orbit vs MC arc %"),
    ("filter_dynamics/main_summary.md", "Quantitative results - position + timing"),
    ("filter_dynamics/main_policy_summary.csv", "Aggregated policy stats (CSV)"),
    ("filter_dynamics/main_trials.csv", "Per-trial rows"),
    ("filter_dynamics/timing_conclusion.md", "Timing conclusions (clock on PR epochs)"),
    ("filter_dynamics/pulsar_sweep.md", "Pulsar count sweep table"),
    ("filter_dynamics/toa_sweep.md", "TOA sigma sweep table"),
    ("filter_dynamics/q_sweep.md", "Process-noise / sigma_acc sweep (Q diagnostics)"),
    ("common/predict_mode_comparison.md", "Truth-velocity vs CV vs dynamics"),
]


def timing_conclusion_md(
    result: MonteCarloResult,
    *,
    preset: str,
    predict_label: str = "Filter dynamics predict",
) -> str:
    """Narrative timing section from aggregated MC stats."""
    cfg = result.config
    pct = int(STEADY_STATE_ARC_FRACTION * 100)
    lines = [
        f"## Timing conclusions - {preset.upper()} ({predict_label})",
        "",
        "**Metric:** mean and p95 of **|b_rx - b_truth|** (meters) on epochs with "
        "GNSS and/or LunaNet **pseudorange** updates only.",
        "",
        "Pulsar LOS measurements do **not** observe receiver clock bias in **H**; "
        "`xnav_only` has no defined timing metric in this simulator.",
        "",
        "| Policy | |b| mean (m) | |b| p95 (m) | Interpretation |",
        "|--------|------------|-----------|----------------|",
    ]

    interpretations = {
        NavPolicy.HYBRID: (
            "GNSS (+ LunaNet when relay) constrains clock on non-blackout PR epochs; "
            f"hybrid mean near LunaNet pitch positioning ref ({LUNANET_REQUIREMENT_M} m) is coincidental - "
            "that ref is for **position** p95, not clock."
        ),
        NavPolicy.GNSS_ONLY: (
            "Clock only updated when sidelobe GNSS/LunaNet PRs fire; "
            "blackout relies on pulsars (no clock obs.) -> larger |b| drift."
        ),
        NavPolicy.XNAV_ONLY: "-",
    }

    for pol in cfg.policies:
        s = result.by_policy[pol]
        if pol == NavPolicy.XNAV_ONLY:
            lines.append(
                f"| **{pol.value}** | - | - | MSP-only; clock not in measurement Jacobian |"
            )
            continue
        t_mean = f"{s.timing_mean_m:.2f}" if math.isfinite(s.timing_mean_m) else "-"
        t_p95 = f"{s.timing_p95_m:.2f}" if math.isfinite(s.timing_p95_m) else "-"
        lines.append(
            f"| **{pol.value}** | {t_mean} | {t_p95} | {interpretations[pol]} |"
        )

    hybrid = result.by_policy.get(NavPolicy.HYBRID)
    gnss = result.by_policy.get(NavPolicy.GNSS_ONLY)
    lines.extend(
        [
            "",
            "### Summary bullets",
            "",
            "- **Do not** report pulsar sigma_TOA as clock timing performance; TOA enters as **range sigma** on LOS.",
            f"- **Position** steady-state (~last {pct}% of arc) is reported separately in `main_summary.md`.",
            "- **Hybrid** is the only policy that routinely keeps |b| on PR epochs to tens of meters under dynamics predict.",
            "- **gnss_only** can match hybrid **position** in blackout (shared MSP updates) while **|b|** remains ~100x worse.",
            "",
            f"_Arc: {cfg.duration_s / 3600:.1f} hr | TOA sigma = {cfg.toa_sigma_s * 1e6:.1f} us | "
            f"trials = {cfg.n_trials}_",
            "",
        ]
    )
    if hybrid and gnss and math.isfinite(hybrid.timing_mean_m) and math.isfinite(gnss.timing_mean_m):
        ratio = gnss.timing_mean_m / max(hybrid.timing_mean_m, 1e-6)
        lines.append(
            f"**Headline:** hybrid |b| mean **{hybrid.timing_mean_m:.1f} m** vs "
            f"gnss_only **{gnss.timing_mean_m:.1f} m** (~{ratio:.0f}x) on PR epochs - "
            "fuse GNSS whenever visible to maintain clock observability."
        )
        lines.append("")
    return "\n".join(lines)


def write_timing_conclusion(
    out_dir: Path,
    result: MonteCarloResult,
    *,
    preset: str,
    predict_label: str,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "timing_conclusion.md"
    path.write_text(
        timing_conclusion_md(result, preset=preset, predict_label=predict_label),
        encoding="utf-8",
    )
    return path


def _figure_table(rows: list[tuple[str, str]], base: str) -> list[str]:
    lines = ["| File | Description |", "|------|-------------|"]
    for fname, desc in rows:
        lines.append(f"| `{base}/{fname}` | {desc} |")
    return lines


def unified_presentation_index_md(
    *,
    preset: str = "elfo",
    pipeline_results: dict[str, MonteCarloResult] | None = None,
    quick_mode: bool = False,
) -> str:
    """Single index: assumptions, geometry, filter_dynamics assets, predict-mode comparison."""
    mc_hr = DEFAULT_MC_DURATION_S / 3600.0
    lines = [
        "# Presentation assets",
        "",
        f"**Primary campaign:** `{PRIMARY_PIPELINE}` (EKF RK45+STM + HW2 process noise)",
        "",
        "Regenerate: `python scripts/build_presentation_assets.py --pipelines filter_dynamics`",
        "",
        FILTER_DYNAMICS_ASSUMPTIONS,
        "",
        "---",
        "",
        f"## Geometry & truth ({preset.upper()})",
        "",
        f"Orbit: {elfo_orbit_summary()}",
        "",
        "Use **one-orbit blackout (~65.7%)** for per-revolution geometry; "
        f"**MC arc ({mc_hr:.1f} h, ~64.1%)** for navigation statistics.",
        "",
        "### Figures - `figures/presentation/common/`",
        "",
        *_figure_table(COMMON_FIGURES, "figures/presentation/common"),
        "",
        "### Tables - `presentation/tables/common/`",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `presentation/tables/common/blackout_specs.md` | One-orbit vs MC blackout numbers |",
        "| `presentation/tables/common/pulsar_catalog.md` | SEXTANT MSP catalog (Hz, roles) |",
        "| `presentation/tables/common/predict_mode_comparison.md` | Cross-mode comparison |",
        "| `presentation/tables/common/predict_mode_comparison.csv` | Same data (CSV) |",
        "",
        "Per-mode detailed tables exist only under **`filter_dynamics/`** (primary campaign).",
        "",
        "---",
        "",
        "## Filter dynamics - results",
        "",
        "### Tables - `presentation/tables/filter_dynamics/`",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    for rel, desc in FILTER_DYNAMICS_TABLES:
        if rel.startswith("common/"):
            continue
        if quick_mode and rel.endswith(("pulsar_sweep.md", "toa_sweep.md", "q_sweep.md")):
            continue
        lines.append(f"| `presentation/tables/{rel}` | {desc} |")

    lines.extend(["", "### Figures - `figures/presentation/filter_dynamics/`", ""])
    fd_figs = [
        (f, d)
        for f, d in FILTER_DYNAMICS_FIGURES
        if not (
            quick_mode
            and f in ("mc_elfo_pulsar_sweep.png", "mc_elfo_toa_sweep.png", "mc_elfo_q_diagnostics.png")
        )
    ]
    lines.extend(_figure_table(fd_figs, "figures/presentation/filter_dynamics"))
    lines.extend(
        [
            "",
            "### Spreadsheet",
            "",
            f"`results/monte_carlo_{PRIMARY_PIPELINE}.xlsx`",
            "",
        ]
    )

    if pipeline_results and PRIMARY_PIPELINE in pipeline_results:
        res = pipeline_results[PRIMARY_PIPELINE]
        lines.extend(["### Latest quantitative snapshot (hybrid)", ""])
        s: PolicyStats = res.by_policy[NavPolicy.HYBRID]
        lines.extend(
            [
                f"- Final mean: **{s.final_mean_m / 1e3:.2f} km** | Steady RMS: **{s.steady_state_rms_m / 1e3:.2f} km**",
                f"- Blackout mu: **{s.blackout_mean_m / 1e3:.2f} km** | |b| mean (PR): **{s.timing_mean_m:.1f} m**",
                "",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Predict-mode comparison",
            "",
            "Other predict modes (`truth_velocity`, `filter_predict`) run in MC for comparison only - "
            "no separate figure/table trees.",
            "",
            "- `presentation/tables/common/predict_mode_comparison.md`",
            "- `figures/presentation/common/predict_mode_comparison_*.png`",
            "",
            "",
            "---",
            "",
            "## Docs",
            "",
            "- [docs/PRESENTATION.md](../docs/PRESENTATION.md) - slide order & talking points",
            "- [docs/SIMULATION_LIMITATIONS.md](../docs/SIMULATION_LIMITATIONS.md) - what not to over-claim",
            "",
        ]
    )
    return "\n".join(lines)


def write_unified_presentation_index(
    path: Path,
    *,
    preset: str = "elfo",
    pipeline_results: dict[str, MonteCarloResult] | None = None,
    quick_mode: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        unified_presentation_index_md(
            preset=preset,
            pipeline_results=pipeline_results,
            quick_mode=quick_mode,
        ),
        encoding="utf-8",
    )
    return path


def q_sweep_markdown(rows: list[dict], *, n_trials: int) -> str:
    lines = [
        "## Q diagnostics - HW2 sigma_acc sweep",
        "",
        f"**Trials:** {n_trials} | **Predict:** filter dynamics (RK45+STM)",
        "",
        "Target **median NIS/df approx 1**. Default MC uses sigma_acc = **1e-6** km/s^2/sqrt(s).",
        "",
        "| sigma_acc (km/s^2/sqrt(s)) | Policy | final mu (km) | med NIS/df | |b| mean (m) |",
        "|------------------|--------|--------------|------------|------------|",
    ]
    for r in rows:
        nis = r.get("nis_median_dof", "")
        tmg = r.get("timing_mean_m", "")
        final = r.get("final_mean_km", r.get("final_mean", ""))
        try:
            final_s = f"{float(final):.2f}"
        except (TypeError, ValueError):
            final_s = str(final)
        lines.append(
            f"| {r.get('q_label', r.get('q_param', ''))} | {r['policy']} | "
            f"{final_s} | {nis} | {tmg} |"
        )
    lines.append("")
    return "\n".join(lines)


def pipeline_results_markdown(
    result: MonteCarloResult,
    *,
    preset: str,
    pipeline_slug: str,
    pipeline_label: str,
    pipeline_note: str,
    figure_dir: Path,
    saved_figures: list[str],
) -> str:
    """One results doc per pipeline: assumptions snippet + tables + figures."""
    summary_lines = [
        f"# {pipeline_label}",
        "",
        pipeline_note,
        "",
        f"**Master index:** [presentation/INDEX.md](../../presentation/INDEX.md)",
        "",
        f"Figures: `figures/presentation/{pipeline_slug}/` | "
        f"Tables: `presentation/tables/{pipeline_slug}/`",
        "",
    ]
    if pipeline_slug == PRIMARY_PIPELINE:
        summary_lines.extend([FILTER_DYNAMICS_ASSUMPTIONS, ""])

    cfg = result.config
    bf = result.timeline.blackout_fraction if result.timeline else float("nan")
    pct = int(STEADY_STATE_ARC_FRACTION * 100)
    summary_lines.extend(
        [
            "## Quantitative results",
            "",
            f"Arc **{cfg.duration_s / 3600:.1f} hr** | trials **{cfg.n_trials}** | "
            f"TOA sigma **{cfg.toa_sigma_s * 1e6:.1f} us** | blackout **{100.0 * bf:.1f}%**",
            "",
            f"_Steady metrics: last {pct}% of arc (excludes epoch-0 init spike)._",
            "",
            "| Policy | Final mu (km) | Steady RMS (km) | Blackout mu (km) | |b| mean (m) | |b| p95 (m) |",
            "|--------|--------------|-----------------|-----------------|------------|-----------|",
        ]
    )
    for pol in cfg.policies:
        s = result.by_policy[pol]
        if pol == NavPolicy.XNAV_ONLY:
            t_mean, t_p95 = "-", "-"
        else:
            t_mean = f"{s.timing_mean_m:.2f}" if math.isfinite(s.timing_mean_m) else "-"
            t_p95 = f"{s.timing_p95_m:.2f}" if math.isfinite(s.timing_p95_m) else "-"
        summary_lines.append(
            f"| {pol.value} | {s.final_mean_m / 1e3:.2f} | {s.steady_state_rms_m / 1e3:.2f} | "
            f"{s.blackout_mean_m / 1e3:.2f} | {t_mean} | {t_p95} |"
        )

    summary_lines.extend(
        [
            "",
            f"Full table: `presentation/tables/{pipeline_slug}/main_summary.md` | "
            "timing narrative: `timing_conclusion.md`",
            "",
            "## Figures",
            "",
            "| File | Description |",
            "|------|-------------|",
        ]
    )
    desc_map = {f: d for f, d in FILTER_DYNAMICS_FIGURES}
    for fname in sorted(saved_figures):
        desc = desc_map.get(fname, "MC figure")
        for pattern, pd in FILTER_DYNAMICS_FIGURES:
            if "{" in pattern:
                continue
            if fname == pattern:
                desc = pd
                break
        summary_lines.append(f"| `{fname}` | {desc} |")

    summary_lines.extend(
        [
            "",
            "## Related tables",
            "",
            f"- `presentation/tables/{pipeline_slug}/main_summary.md`",
            f"- `presentation/tables/{pipeline_slug}/pulsar_sweep.md`",
            f"- `presentation/tables/{pipeline_slug}/toa_sweep.md`",
        ]
    )
    if pipeline_slug == PRIMARY_PIPELINE:
        summary_lines.append(f"- `presentation/tables/{pipeline_slug}/q_sweep.md`")
        summary_lines.append(f"- `presentation/tables/{pipeline_slug}/timing_conclusion.md`")
    summary_lines.append("- `presentation/tables/common/pulsar_catalog.md`")
    summary_lines.append("")
    return "\n".join(summary_lines)
