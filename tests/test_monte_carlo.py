"""Monte Carlo campaign tests."""

import numpy as np
import pytest

from pulsar_nav.simulation.monte_carlo import (
    LUNANET_REQUIREMENT_M,
    MonteCarloConfig,
    aggregate_policy_stats,
    run_monte_carlo,
    select_pulsars,
)
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.spice.kernels import resolve_kernel_dir

spiceypy = pytest.importorskip("spiceypy")


def _kernels_available() -> bool:
    try:
        resolve_kernel_dir()
        return True
    except FileNotFoundError:
        return False


@pytest.fixture(scope="module")
def spice_loaded():
    from pulsar_nav.spice.kernels import load_kernels

    load_kernels(load_gps_frames=True)


def test_select_pulsars_subset():
    p1 = select_pulsars(1)
    p3 = select_pulsars(3)
    assert len(p1) == 1
    assert len(p3) == 3


def test_aggregate_policy_stats():
    from pulsar_nav.simulation.monte_carlo import TrialMetrics

    trials = [
        TrialMetrics(
            trial_id=0,
            policy=NavPolicy.HYBRID,
            final_error_m=10.0,
            mean_error_m=8.0,
            rms_error_m=9.0,
            p95_error_m=12.0,
            max_error_m=15.0,
            blackout_mean_m=11.0,
            non_blackout_mean_m=5.0,
            n_pulsars=5,
            toa_sigma_s=1e-4,
            position_offset_m=50e3,
        ),
        TrialMetrics(
            trial_id=1,
            policy=NavPolicy.HYBRID,
            final_error_m=20.0,
            mean_error_m=18.0,
            rms_error_m=19.0,
            p95_error_m=22.0,
            max_error_m=25.0,
            blackout_mean_m=21.0,
            non_blackout_mean_m=9.0,
            n_pulsars=5,
            toa_sigma_s=1e-4,
            position_offset_m=60e3,
        ),
    ]
    stats = aggregate_policy_stats(trials, NavPolicy.HYBRID)
    assert stats.final_mean_m == 15.0
    assert stats.n_trials == 2
    assert stats.meets_lunanet_p95 is False


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_monte_carlo_runs_all_policies(spice_loaded):
    cfg = MonteCarloConfig(
        n_trials=2,
        seed=99,
        duration_s=1800.0,
        step_s=300.0,
        randomize_offset=False,
        position_offset_m=40_000.0,
        policies=(NavPolicy.HYBRID, NavPolicy.XNAV_ONLY, NavPolicy.GNSS_ONLY),
    )
    # Reuse short arc — run_monte_carlo propagates internally; just check structure
    result = run_monte_carlo(cfg)
    assert len(result.trials) == 2 * 3
    assert NavPolicy.HYBRID in result.by_policy
    for pol in cfg.policies:
        assert result.by_policy[pol].n_trials == 2
        assert result.by_policy[pol].final_mean_m < 500_000.0


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_hybrid_with_broadcast_gps_campaign(spice_loaded):
    """Hybrid MC with sidelobe GNSS completes; sparse PRNs may not beat XNAV-only finals."""
    cfg = MonteCarloConfig(
        n_trials=2,
        seed=11,
        duration_s=6.0 * 3600.0,
        step_s=180.0,
        randomize_offset=False,
        position_offset_m=50_000.0,
        policies=(NavPolicy.HYBRID, NavPolicy.XNAV_ONLY),
    )
    result = run_monte_carlo(cfg)
    h = result.by_policy[NavPolicy.HYBRID]
    x = result.by_policy[NavPolicy.XNAV_ONLY]
    assert h.n_trials == 2
    assert h.final_mean_m < 100_000.0
    assert x.final_mean_m < 100_000.0


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_elfo_presets_differ_in_visibility(spice_loaded):
    """Science vs argp+180 phasing produce different GNSS geometry at the same epoch."""
    from pulsar_nav.propagation.dynamics import DynamicsConfig
    from pulsar_nav.propagation.propagator import LunarPropagator
    from pulsar_nav.spice.ephemeris import str_to_et
    from pulsar_nav.visibility.blackout import compute_visibility_timeline

    et0 = str_to_et("2026-01-15 12:00:00")
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj_science = prop.propagate_preset("elfo", duration_s=6.0 * 3600.0, step_s=120.0)
    traj_nav = prop.propagate_preset("elfo_nav", duration_s=6.0 * 3600.0, step_s=120.0)
    tl_science = compute_visibility_timeline(traj_science)
    tl_nav = compute_visibility_timeline(traj_nav)

    assert abs(tl_science.blackout_fraction - tl_nav.blackout_fraction) > 0.25


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_monte_carlo_preset_comparison(spice_loaded):
    cfg = MonteCarloConfig(
        n_trials=2,
        seed=42,
        duration_s=6.0 * 3600.0,
        step_s=300.0,
        randomize_offset=False,
        position_offset_m=40_000.0,
        policies=(NavPolicy.HYBRID, NavPolicy.GNSS_ONLY),
    )
    from pulsar_nav.simulation.monte_carlo import run_preset_comparison

    results = run_preset_comparison(("elfo", "elfo_nav"), cfg)
    assert set(results) == {"elfo", "elfo_nav"}
    assert (
        abs(
            results["elfo_nav"].timeline.blackout_fraction
            - results["elfo"].timeline.blackout_fraction
        )
        > 0.25
    )


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_export_monte_carlo_xlsx(spice_loaded, tmp_path):
    pytest.importorskip("openpyxl")
    from pulsar_nav.simulation.monte_carlo_export import (
        MonteCarloExportBundle,
        export_monte_carlo_xlsx,
    )

    cfg = MonteCarloConfig(
        n_trials=2,
        seed=1,
        duration_s=1800.0,
        step_s=300.0,
        randomize_offset=False,
        position_offset_m=40_000.0,
        policies=(NavPolicy.HYBRID, NavPolicy.XNAV_ONLY),
    )
    result = run_monte_carlo(cfg)
    path = export_monte_carlo_xlsx(
        tmp_path / "mc.xlsx",
        MonteCarloExportBundle(main=result),
    )
    assert path.exists()

    from openpyxl import load_workbook

    wb = load_workbook(path)
    assert {"config", "summary", "trials"}.issubset(set(wb.sheetnames))
    assert wb["trials"].max_row >= 5  # header + 2 trials × 2 policies


@pytest.mark.skipif(not _kernels_available(), reason="SPICE kernels not on disk")
def test_hybrid_beats_gnss_only_in_blackout_heavy_arc(spice_loaded):
    cfg = MonteCarloConfig(
        n_trials=3,
        seed=7,
        preset="elfo_nav",
        duration_s=4.0 * 3600.0,
        step_s=180.0,
        randomize_offset=True,
        policies=(NavPolicy.HYBRID, NavPolicy.GNSS_ONLY),
    )
    result = run_monte_carlo(cfg)
    h = result.by_policy[NavPolicy.HYBRID]
    g = result.by_policy[NavPolicy.GNSS_ONLY]
    assert h.blackout_mean_m < g.blackout_mean_m
