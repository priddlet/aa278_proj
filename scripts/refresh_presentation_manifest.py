#!/usr/bin/env python3
"""Regenerate unified INDEX, timing conclusion, and q_sweep.md without re-running MC."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pulsar_nav.simulation.orbit_specs import blackout_specs_markdown
from pulsar_nav.simulation.presentation_manifest import (
    q_sweep_markdown,
    write_unified_presentation_index,
)
from pulsar_nav.simulation.presentation_tables import export_pulsar_catalog_table


def _read_policy_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _timing_from_csv(rows: list[dict], *, preset: str = "elfo") -> str:
    lines = [
        f"## Timing conclusions - {preset.upper()} (Filter dynamics predict)",
        "",
        "**Metric:** |b_rx - b_truth| (m) on GNSS/LunaNet pseudorange epochs.",
        "",
        "| Policy | |b| mean (m) | |b| p95 (m) |",
        "|--------|------------|-----------|",
    ]
    for row in rows:
        pol = row.get("policy", "")
        t_mean = row.get("timing_mean_m", "")
        t_p95 = row.get("timing_p95_m", "")
        if pol == "xnav_only" or t_mean in ("", "nan", "NaN"):
            lines.append(f"| **{pol}** | - | - |")
        else:
            lines.append(f"| **{pol}** | {float(t_mean):.2f} | {float(t_p95):.2f} |")
    lines.extend(
        [
            "",
            "See `presentation/INDEX.md` for interpretation. "
            "Hybrid maintains clock observability on PR epochs; gnss_only drift is larger.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    tables = ROOT / "presentation" / "tables"
    fd = tables / "filter_dynamics"
    common = tables / "common"
    common.mkdir(parents=True, exist_ok=True)

    export_pulsar_catalog_table(common, preset="elfo")
    (common / "blackout_specs.md").write_text(blackout_specs_markdown(), encoding="utf-8")

    policy_rows = _read_policy_csv(fd / "main_policy_summary.csv")
    if policy_rows:
        (fd / "timing_conclusion.md").write_text(
            _timing_from_csv(policy_rows),
            encoding="utf-8",
        )
        print(f"Wrote {fd / 'timing_conclusion.md'}")

    q_csv = fd / "q_sweep.csv"
    if q_csv.is_file():
        with q_csv.open(encoding="utf-8") as f:
            q_rows = list(csv.DictReader(f))
        (fd / "q_sweep.md").write_text(
            q_sweep_markdown(q_rows, n_trials=10),
            encoding="utf-8",
        )
        print(f"Wrote {fd / 'q_sweep.md'}")

    write_unified_presentation_index(
        ROOT / "presentation" / "INDEX.md",
        preset="elfo",
        pipeline_results=None,
        quick_mode=False,
    )
    print(f"Wrote {ROOT / 'presentation' / 'INDEX.md'}")

    fd_figs = sorted((ROOT / "figures" / "presentation" / "filter_dynamics").glob("*.png"))
    if policy_rows and fd_figs:
        lines = [
            "# Filter dynamics predict",
            "",
            "**Master index:** [presentation/INDEX.md](../presentation/INDEX.md)",
            "",
            "## Quantitative results",
            "",
            "See `presentation/tables/filter_dynamics/main_summary.md` and "
            "`timing_conclusion.md`.",
            "",
            "## Figures",
            "",
            "| File |",
            "|------|",
        ]
        for p in fd_figs:
            lines.append(f"| `{p.name}` |")
        lines.append("")
        out = ROOT / "results" / "presentation_filter_dynamics.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {out}")

    (ROOT / "results" / "presentation_INDEX.md").write_text(
        "# Presentation index\n\n"
        "See **[presentation/INDEX.md](../presentation/INDEX.md)**.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
