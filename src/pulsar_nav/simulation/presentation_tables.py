"""Markdown and CSV tables for slides (per navigation pipeline)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from pulsar_nav.simulation.monte_carlo import (
    STEADY_STATE_ARC_FRACTION,
    MonteCarloResult,
    PolicyStats,
)
from pulsar_nav.simulation.monte_carlo_export import (
    MonteCarloExportBundle,
    _summary_row,
    _trial_row,
)
from pulsar_nav.simulation.policy import NavPolicy


def _timing_display(policy: NavPolicy, stats: PolicyStats) -> tuple[str, str]:
    if policy == NavPolicy.XNAV_ONLY:
        return "-", "-"
    t_mean = f"{stats.timing_mean_m:.2f}" if math.isfinite(stats.timing_mean_m) else "-"
    t_p95 = f"{stats.timing_p95_m:.2f}" if math.isfinite(stats.timing_p95_m) else "-"
    return t_mean, t_p95


def _timing_csv_fields(stats: PolicyStats) -> dict[str, float]:
    return {
        "timing_mean_m": stats.timing_mean_m,
        "timing_final_m": stats.timing_final_m,
        "timing_p95_m": stats.timing_p95_m,
    }
def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _main_summary_md(
    result: MonteCarloResult,
    *,
    preset: str,
    predict_label: str,
    predict_note: str,
) -> str:
    cfg = result.config
    bf = result.timeline.blackout_fraction if result.timeline else float("nan")
    lines = [
        f"## Monte Carlo - {preset.upper()} ({predict_label})",
        "",
        f"EKF predict: **{predict_label}**  |  "
        f"Arc: **{cfg.duration_s / 3600:.1f} hr**  |  "
        f"Trials: **{cfg.n_trials}**  |  "
        f"TOA sigma: **{cfg.toa_sigma_s * 1e6:.1f} us**  |  pulsars: "
        f"**{cfg.n_pulsars if cfg.n_pulsars is not None else 'all (5)'}**",
        "",
        f"_{predict_note}_",
        "",
        f"Blackout fraction: **{100.0 * bf:.1f}%**",
        "",
        "_Timing: |b_rx-b_truth| (m) averaged over GNSS/LunaNet pseudorange epochs only. "
        "XNAV-only and MSP-only blackout do not constrain b in H._",
        "",
        f"_Steady mu / Steady RMS: mean and RMS over the last "
        f"{int(STEADY_STATE_ARC_FRACTION * 100)}% of arc epochs (post-convergence; "
        f"excludes epoch-0 init spike)._",
        "",
        "| Policy | Final mean (km) | Final p95 (km) | RMS (km) | Steady mu (km) | Steady RMS (km) | Blackout mu (km) | Non-blackout mu (km) | |b| mean (m) | |b| p95 (m) |",
        "|--------|-----------------|----------------|----------|---------------|-----------------|-----------------|---------------------|------------|-----------|",
    ]
    for pol in cfg.policies:
        s = result.by_policy[pol]
        t_mean, t_p95 = _timing_display(pol, s)
        lines.append(
            f"| **{pol.value}** | {s.final_mean_m / 1e3:.2f} | {s.final_p95_m / 1e3:.2f} | "
            f"{s.rms_error_m / 1e3:.2f} | {s.steady_state_mean_m / 1e3:.2f} | "
            f"{s.steady_state_rms_m / 1e3:.2f} | {s.blackout_mean_m / 1e3:.2f} | "
            f"{s.non_blackout_mean_m / 1e3:.2f} | {t_mean} | {t_p95} |"
        )
    lines.append("")
    return "\n".join(lines)


def _pulsar_sweep_md(sweep: dict[int, MonteCarloResult]) -> str:
    lines = [
        "## Pulsar count sweep",
        "",
        "_|b| timing on GNSS/LunaNet pseudorange epochs; XNAV-only: - (clock not in H)._",
        "",
        "| MSPs | Policy | Final mean (km) | Final p95 (km) | Blackout mu (km) | |b| mean (m) | |b| p95 (m) |",
        "|------|--------|-----------------|----------------|-----------------|------------|-----------|",
    ]
    for n in sorted(sweep.keys()):
        res = sweep[n]
        for pol in res.config.policies:
            s = res.by_policy[pol]
            t_mean, t_p95 = _timing_display(pol, s)
            lines.append(
                f"| {n} | {pol.value} | {s.final_mean_m / 1e3:.2f} | "
                f"{s.final_p95_m / 1e3:.2f} | {s.blackout_mean_m / 1e3:.2f} | {t_mean} | {t_p95} |"
            )
    lines.append("")
    return "\n".join(lines)


def _toa_sweep_md(sweep: dict[float, MonteCarloResult]) -> str:
    lines = [
        "## TOA noise sweep",
        "",
        "_|b| timing on GNSS/LunaNet pseudorange epochs; XNAV-only: - (clock not in H)._",
        "",
        "| TOA us | Policy | Final mean (km) | Final p95 (km) | Blackout mu (km) | |b| mean (m) | |b| p95 (m) |",
        "|--------|--------|-----------------|----------------|-----------------|------------|-----------|",
    ]
    for sig_us in sorted(sweep.keys()):
        res = sweep[sig_us]
        for pol in res.config.policies:
            s = res.by_policy[pol]
            t_mean, t_p95 = _timing_display(pol, s)
            lines.append(
                f"| {sig_us:.1f} | {pol.value} | {s.final_mean_m / 1e3:.2f} | "
                f"{s.final_p95_m / 1e3:.2f} | {s.blackout_mean_m / 1e3:.2f} | {t_mean} | {t_p95} |"
            )
    lines.append("")
    return "\n".join(lines)


def _pulsar_sweep_csv(sweep: dict[int, MonteCarloResult], *, predict_mode: str) -> list[dict]:
    rows: list[dict] = []
    for n in sorted(sweep.keys()):
        res = sweep[n]
        for pol in res.config.policies:
            s = res.by_policy[pol]
            rows.append(
                {
                    "predict_mode": predict_mode,
                    "n_msps": n,
                    "policy": pol.value,
                    "final_mean_km": round(s.final_mean_m / 1e3, 3),
                    "final_p95_km": round(s.final_p95_m / 1e3, 3),
                    "blackout_mean_km": round(s.blackout_mean_m / 1e3, 3),
                    **_timing_csv_fields(s),
                }
            )
    return rows


def _toa_sweep_csv(sweep: dict[float, MonteCarloResult], *, predict_mode: str) -> list[dict]:
    rows: list[dict] = []
    for sig_us in sorted(sweep.keys()):
        res = sweep[sig_us]
        for pol in res.config.policies:
            s = res.by_policy[pol]
            rows.append(
                {
                    "predict_mode": predict_mode,
                    "toa_sigma_us": sig_us,
                    "policy": pol.value,
                    "final_mean_km": round(s.final_mean_m / 1e3, 3),
                    "final_p95_km": round(s.final_p95_m / 1e3, 3),
                    "blackout_mean_km": round(s.blackout_mean_m / 1e3, 3),
                    **_timing_csv_fields(s),
                }
            )
    return rows


def export_presentation_tables(
    out_dir: Path,
    bundle: MonteCarloExportBundle,
    *,
    preset: str,
    predict_label: str,
    predict_note: str,
) -> list[str]:
    """
    Write ``main_*`` and sweep tables under ``out_dir`` (e.g. ``presentation/tables/truth_velocity``).

    Returns relative filenames written.
    """
    written: list[str] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bundle.main is None:
        return written

    campaign = f"main_{preset}"
    result = bundle.main
    from pulsar_nav.simulation.predict_mode import PredictMode, resolve_predict_mode

    mode = resolve_predict_mode(
        predict_mode=result.config.predict_mode,
        use_truth_velocity_predict=result.config.use_truth_velocity_predict,
        use_dynamics_predict=result.config.use_dynamics_predict,
    )
    predict_mode = {
        PredictMode.TRUTH_VELOCITY: "truth_velocity",
        PredictMode.CV: "filter_predict",
        PredictMode.DYNAMICS: "filter_dynamics",
    }[mode]

    summary_md = _main_summary_md(
        result,
        preset=preset,
        predict_label=predict_label,
        predict_note=predict_note,
    )
    (out_dir / "main_summary.md").write_text(summary_md, encoding="utf-8")
    written.append("main_summary.md")

    summary_rows = [_summary_row(campaign, result.by_policy[pol]) for pol in result.config.policies]
    _write_csv(out_dir / "main_policy_summary.csv", summary_rows)
    written.append("main_policy_summary.csv")

    trial_rows = [_trial_row(campaign, t) for t in result.trials]
    _write_csv(out_dir / "main_trials.csv", trial_rows)
    written.append("main_trials.csv")

    if bundle.pulsar_sweep:
        (out_dir / "pulsar_sweep.md").write_text(_pulsar_sweep_md(bundle.pulsar_sweep), encoding="utf-8")
        written.append("pulsar_sweep.md")
        _write_csv(out_dir / "pulsar_sweep.csv", _pulsar_sweep_csv(bundle.pulsar_sweep, predict_mode=predict_mode))
        written.append("pulsar_sweep.csv")

    if bundle.toa_sweep:
        (out_dir / "toa_sweep.md").write_text(_toa_sweep_md(bundle.toa_sweep), encoding="utf-8")
        written.append("toa_sweep.md")
        _write_csv(out_dir / "toa_sweep.csv", _toa_sweep_csv(bundle.toa_sweep, predict_mode=predict_mode))
        written.append("toa_sweep.csv")

    return written


def write_tables_index(
    index_path: Path,
    *,
    preset: str,
    pipelines: list[tuple[str, str, str]],
) -> None:
    """Write ``presentation/tables/INDEX.md`` listing both predict modes."""
    lines = [
        "# Presentation tables",
        "",
        "Generated by `python scripts/build_presentation_assets.py`. "
        "Each navigation pipeline has its own subdirectory.",
        "",
        "| Predict mode | Directory | EKF time update |",
        "|--------------|-----------|-----------------|",
    ]
    for slug, label, note in pipelines:
        rel = f"`{slug}/`"
        lines.append(f"| **{label}** | {rel} | {note} |")
    lines.extend(
        [
            "",
            "## Per-pipeline files",
            "",
            "- `main_summary.md` - policy comparison (km)",
            "- `main_policy_summary.csv` - aggregated stats",
            "- `main_trials.csv` - per-trial rows",
            "- `pulsar_sweep.md` / `.csv` - MSP count sensitivity; includes |b| timing (m) on PR epochs",
            "- `toa_sweep.md` / `.csv` - TOA sigma sensitivity; includes |b| timing (m) on PR epochs",
            "",
            f"## SEXTANT / NICER MSP catalog ({preset})",
            "",
            "See bundled catalog in `src/pulsar_nav/catalog/`.",
            "",
            "## Stress baseline (`gnss_coast`)",
            "",
            "Not in default three-policy tables. Enable `NavPolicy.GNSS_COAST` in a custom MC config.",
            "",
        ]
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("\n".join(lines), encoding="utf-8")


def export_pulsar_catalog_table(out_dir: Path, *, preset: str = "elfo") -> None:
    """Write ``pulsar_catalog.md`` and ``.csv`` under ``out_dir/common``."""
    from pulsar_nav.catalog import load_catalog

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pulsars = load_catalog()
    md_lines = [
        f"## SEXTANT MSP catalog - {preset.upper()} simulation",
        "",
        "All five millisecond pulsars are used for **navigation** (LOS range, sigma = c*sigma_TOA).",
        "Catalog **f0** supports **timing** / phase models; the MC EKF uses linear n_hat dot r only.",
        "",
        "Default campaign: **all 5 MSPs** every **120 s**; sigma_TOA = **1 us** (~300 m range sigma).",
        "",
        "| Name | J-name | RA (hms) | Dec (dms) | f0 (Hz) | f1 (Hz/s) | DM (pc/cm^3) | Navigation | Timing model |",
        "|------|--------|----------|-----------|---------|-----------|-------------|------------|--------------|",
    ]
    csv_rows: list[dict] = []
    for p in pulsars:
        md_lines.append(
            f"| {p.name} | {p.jname or ''} | {p.raj} | {p.decj} | {p.f0_hz:.3f} | {p.f1_hz_s:.2e} | "
            f"{p.dm if p.dm is not None else ''} | LOS XNAV | f0, f1, pepoch |"
        )
        csv_rows.append(
            {
                "name": p.name,
                "jname": p.jname or "",
                "raj": p.raj,
                "decj": p.decj,
                "f0_hz": p.f0_hz,
                "f1_hz_s": p.f1_hz_s,
                "dm": p.dm if p.dm is not None else "",
                "navigation_role": "LOS XNAV (n_hat dot r)",
            }
        )
    md_lines.append("")
    (out_dir / "pulsar_catalog.md").write_text("\n".join(md_lines), encoding="utf-8")
    _write_csv(out_dir / "pulsar_catalog.csv", csv_rows)
