"""Monte Carlo navigation campaigns (Week 9)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from pulsar_nav.catalog import load_catalog
from pulsar_nav.catalog.pulsar import Pulsar
from pulsar_nav.constants import DEFAULT_TOA_SIGMA_S, DEFAULT_MC_DURATION_S
from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.propagation.propagator import LunarPropagator, PropagatedTrajectory
from pulsar_nav.simulation.hybrid_run import HybridRunResult, run_hybrid_ekf, timing_metrics_from_run
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.simulation.xnav_run import offset_initial_position
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.visibility.blackout import VisibilityTimeline, compute_visibility_timeline

# LunaNet far-side positioning target from pitch / HW2 reference (meters).
LUNANET_REQUIREMENT_M = 13.43


class MonteCarloSweep(str, Enum):
    """Optional one-dimensional sweep axis for a campaign."""

    NONE = "none"
    PULSAR_COUNT = "pulsar_count"
    TOA_NOISE = "toa_noise"


@dataclass
class MonteCarloConfig:
    """Single-scenario Monte Carlo settings."""

    n_trials: int = 20
    seed: int = 0
    preset: str = "elfo"
    epoch_utc: str = "2026-01-15T12:00:00"
    duration_s: float = DEFAULT_MC_DURATION_S
    step_s: float = 120.0
    position_offset_m: float = 50_000.0
    randomize_offset: bool = True
    offset_min_m: float = 30_000.0
    offset_max_m: float = 100_000.0
    toa_sigma_s: float = DEFAULT_TOA_SIGMA_S
    gnss_sigma_m: float = 15.0
    lonet_sigma_m: float = 15.0
    n_pulsars: int | None = None  # None = full SEXTANT set
    process_noise_accel: float = 1e-4
    use_truth_velocity_predict: bool = True
    policies: tuple[NavPolicy, ...] = (
        NavPolicy.XNAV_ONLY,
        NavPolicy.GNSS_ONLY,
        NavPolicy.HYBRID,
    )


@dataclass
class TrialMetrics:
    """Scalar metrics from one filter trial."""

    trial_id: int
    policy: NavPolicy
    final_error_m: float
    mean_error_m: float
    rms_error_m: float
    p95_error_m: float
    max_error_m: float
    blackout_mean_m: float
    non_blackout_mean_m: float
    n_pulsars: int
    toa_sigma_s: float
    position_offset_m: float
    sweep_label: str = ""
    timing_mean_m: float = float("nan")
    timing_final_m: float = float("nan")
    timing_p95_m: float = float("nan")


@dataclass
class PolicyStats:
    """Aggregated statistics for one policy over trials."""

    policy: NavPolicy
    n_trials: int
    final_mean_m: float
    final_std_m: float
    final_p95_m: float
    mean_error_m: float
    rms_error_m: float
    blackout_mean_m: float
    non_blackout_mean_m: float
    meets_lunanet_p95: bool  # p95 final error < 13.43 m (optimistic check)
    timing_mean_m: float = float("nan")
    timing_final_m: float = float("nan")
    timing_p95_m: float = float("nan")

    @property
    def final_mean_km(self) -> float:
        return self.final_mean_m / 1000.0


@dataclass
class MonteCarloResult:
    """Full campaign output."""

    config: MonteCarloConfig
    trials: list[TrialMetrics]
    by_policy: dict[NavPolicy, PolicyStats]
    trajectory: PropagatedTrajectory | None = None
    timeline: VisibilityTimeline | None = None

    def summary_table(self) -> str:
        lines = [
            f"Monte Carlo — {self.config.preset.upper()}  "
            f"{self.config.duration_s/3600:.1f} hr  n={self.config.n_trials}",
            f"TOA σ={self.config.toa_sigma_s*1e6:.1f} µs  "
            f"pulsars={self.config.n_pulsars or 'all'}",
            "",
            f"{'Policy':<12} {'Final mean':>12} {'Final p95':>12} "
            f"{'RMS':>10} {'Blackout μ':>12} {'|b| μ PR':>10} {'<13.43m p95':>12}",
        ]
        for pol in self.config.policies:
            s = self.by_policy[pol]
            ok = "yes" if s.meets_lunanet_p95 else "no"
            t_mu = (
                f"{s.timing_mean_m:>8.2f} m"
                if pol != NavPolicy.XNAV_ONLY and np.isfinite(s.timing_mean_m)
                else f"{'n/a':>10}"
            )
            lines.append(
                f"{pol.value:<12} {s.final_mean_m/1e3:>10.2f} km {s.final_p95_m/1e3:>10.2f} km "
                f"{s.rms_error_m/1e3:>8.2f} km {s.blackout_mean_m/1e3:>10.2f} km {t_mu:>10} {ok:>12}"
            )
        lines.append(
            f"\nLunaNet reference: {LUNANET_REQUIREMENT_M:.2f} m (pitch)"
            "\n|b| μ PR: mean |b_rx−b_truth| on GNSS/LunaNet pseudorange epochs only "
            "(XNAV-only / MSP-only blackout: clock not in H)."
        )
        return "\n".join(lines)


def comparison_summary_table(results: dict[str, MonteCarloResult]) -> str:
    """Side-by-side policy stats for multiple orbit presets."""
    presets = list(results.keys())
    if not presets:
        return "No results."

    first = results[presets[0]]
    policies = first.config.policies
    lines = [
        "Monte Carlo preset comparison",
        f"Duration: {first.config.duration_s/3600:.1f} hr  "
        f"Trials: {first.config.n_trials}  "
        f"TOA σ={first.config.toa_sigma_s*1e6:.1f} µs",
        "",
    ]

    for preset in presets:
        res = results[preset]
        bf = res.timeline.blackout_fraction if res.timeline else float("nan")
        lines.append(f"=== {preset.upper()} (blackout {100.0 * bf:.1f}%) ===")
        lines.append(res.summary_table())
        lines.append("")

    lines.append("Hybrid final mean (km) by preset:")
    row = "  " + "  ".join(
        f"{p}: {results[p].by_policy[NavPolicy.HYBRID].final_mean_km:.2f}"
        for p in presets
        if NavPolicy.HYBRID in results[p].by_policy
    )
    lines.append(row)
    return "\n".join(lines)


def run_preset_comparison(
    presets: tuple[str, ...],
    config: MonteCarloConfig | None = None,
) -> dict[str, MonteCarloResult]:
    """Run Monte Carlo for each preset with shared trial count and noise settings."""
    base = config or MonteCarloConfig()
    results: dict[str, MonteCarloResult] = {}
    for i, preset in enumerate(presets):
        cfg = MonteCarloConfig(
            n_trials=base.n_trials,
            seed=base.seed + i * 1000,
            preset=preset,
            epoch_utc=base.epoch_utc,
            duration_s=base.duration_s,
            step_s=base.step_s,
            position_offset_m=base.position_offset_m,
            randomize_offset=base.randomize_offset,
            offset_min_m=base.offset_min_m,
            offset_max_m=base.offset_max_m,
            toa_sigma_s=base.toa_sigma_s,
            gnss_sigma_m=base.gnss_sigma_m,
            lonet_sigma_m=base.lonet_sigma_m,
            n_pulsars=base.n_pulsars,
            process_noise_accel=base.process_noise_accel,
            use_truth_velocity_predict=base.use_truth_velocity_predict,
            policies=base.policies,
        )
        results[preset] = run_monte_carlo(cfg)
    return results


def select_pulsars(n: int | None) -> list[Pulsar]:
    catalog = load_catalog()
    if n is None or n >= len(catalog):
        return catalog
    if n < 1:
        raise ValueError("n_pulsars must be >= 1")
    return catalog[:n]


def _trial_offset_m(config: MonteCarloConfig, rng: np.random.Generator) -> float:
    if config.randomize_offset:
        return float(rng.uniform(config.offset_min_m, config.offset_max_m))
    return config.position_offset_m


def _percentile(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def _nanmean_finite(a: np.ndarray) -> float:
    """Mean over finite entries; NaN if empty or all-NaN (no RuntimeWarning)."""
    if a.size == 0:
        return float("nan")
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def aggregate_policy_stats(trials: list[TrialMetrics], policy: NavPolicy) -> PolicyStats:
    subset = [t for t in trials if t.policy == policy]
    if not subset:
        raise ValueError(f"No trials for policy {policy}")
    final = np.array([t.final_error_m for t in subset])
    mean_e = np.array([t.mean_error_m for t in subset])
    rms_e = np.array([t.rms_error_m for t in subset])
    bo = np.array([t.blackout_mean_m for t in subset])
    nb = np.array([t.non_blackout_mean_m for t in subset])
    t_mean = np.array([t.timing_mean_m for t in subset])
    t_final = np.array([t.timing_final_m for t in subset])
    t_p95 = np.array([t.timing_p95_m for t in subset])
    p95 = _percentile(final, 95.0)
    return PolicyStats(
        policy=policy,
        n_trials=len(subset),
        final_mean_m=float(np.mean(final)),
        final_std_m=float(np.std(final)),
        final_p95_m=p95,
        mean_error_m=float(np.mean(mean_e)),
        rms_error_m=float(np.mean(rms_e)),
        blackout_mean_m=float(np.nanmean(bo)),
        non_blackout_mean_m=float(np.nanmean(nb)),
        meets_lunanet_p95=p95 < LUNANET_REQUIREMENT_M,
        timing_mean_m=_nanmean_finite(t_mean),
        timing_final_m=_nanmean_finite(t_final),
        timing_p95_m=_nanmean_finite(t_p95),
    )


def run_single_trial(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    pulsars: list[Pulsar],
    policy: NavPolicy,
    *,
    trial_id: int,
    position_offset_m: float,
    rng: np.random.Generator,
    config: MonteCarloConfig,
    sweep_label: str = "",
) -> TrialMetrics:
    """One EKF trial on a fixed truth arc."""
    init_pos = offset_initial_position(
        traj.position_icrs_m[0],
        position_offset_m,
        rng,
    )
    init_vel = traj.velocity_icrs_m_s[0].copy()
    meas_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))

    result: HybridRunResult = run_hybrid_ekf(
        traj,
        timeline,
        pulsars,
        initial_position_m=init_pos,
        initial_velocity_m_s=init_vel,
        toa_sigma_s=config.toa_sigma_s,
        gnss_sigma_m=config.gnss_sigma_m,
        lonet_sigma_m=config.lonet_sigma_m,
        process_noise_accel=config.process_noise_accel,
        use_truth_velocity_predict=config.use_truth_velocity_predict,
        rng=meas_rng,
        policy=policy,
    )
    seg = result.segment_errors(timeline)
    errs = result.position_error_m
    timing = timing_metrics_from_run(result, policy)
    return TrialMetrics(
        trial_id=trial_id,
        policy=policy,
        final_error_m=result.final_position_error_m,
        mean_error_m=result.mean_position_error_m,
        rms_error_m=result.rms_position_error_m,
        p95_error_m=_percentile(errs, 95.0),
        max_error_m=float(np.max(errs)),
        blackout_mean_m=seg["blackout_mean_m"],
        non_blackout_mean_m=seg["non_blackout_mean_m"],
        n_pulsars=len(pulsars),
        toa_sigma_s=config.toa_sigma_s,
        position_offset_m=position_offset_m,
        sweep_label=sweep_label,
        timing_mean_m=timing["timing_mean_m"],
        timing_final_m=timing["timing_final_m"],
        timing_p95_m=timing["timing_p95_m"],
    )


def run_monte_carlo(
    config: MonteCarloConfig | None = None,
    *,
    propagate_once: bool = True,
) -> MonteCarloResult:
    """
    Run Monte Carlo trials for each ``NavPolicy`` in the config.

    Truth trajectory and visibility timeline are shared across trials;
    per-trial randomness is initial offset and measurement noise.
    """
    cfg = config or MonteCarloConfig()
    pulsars = select_pulsars(cfg.n_pulsars)

    from pulsar_nav.spice.kernels import load_kernels

    load_kernels(load_gps_frames=True)
    et0 = str_to_et(cfg.epoch_utc)
    prop = LunarPropagator(et0, config=DynamicsConfig(), auto_load_kernels=False)
    traj = prop.propagate_preset(cfg.preset, duration_s=cfg.duration_s, step_s=cfg.step_s)
    timeline = compute_visibility_timeline(traj)

    trials: list[TrialMetrics] = []
    master_rng = np.random.default_rng(cfg.seed)

    for trial_id in range(cfg.n_trials):
        offset_m = _trial_offset_m(cfg, master_rng)
        for policy in cfg.policies:
            trial_rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            trials.append(
                run_single_trial(
                    traj,
                    timeline,
                    pulsars,
                    policy,
                    trial_id=trial_id,
                    position_offset_m=offset_m,
                    rng=trial_rng,
                    config=cfg,
                )
            )

    by_policy = {pol: aggregate_policy_stats(trials, pol) for pol in cfg.policies}
    return MonteCarloResult(
        config=cfg,
        trials=trials,
        by_policy=by_policy,
        trajectory=traj if propagate_once else None,
        timeline=timeline,
    )


def run_pulsar_count_sweep(
    counts: tuple[int, ...] = (1, 3, 5),
    *,
    base_config: MonteCarloConfig | None = None,
) -> dict[int, MonteCarloResult]:
    """Monte Carlo vs number of pulsars (hybrid + xnav only)."""
    base = base_config or MonteCarloConfig(
        policies=(NavPolicy.HYBRID, NavPolicy.XNAV_ONLY),
    )
    results: dict[int, MonteCarloResult] = {}
    for n in counts:
        cfg = MonteCarloConfig(
            n_trials=base.n_trials,
            seed=base.seed,
            preset=base.preset,
            duration_s=base.duration_s,
            step_s=base.step_s,
            toa_sigma_s=base.toa_sigma_s,
            n_pulsars=n,
            policies=base.policies,
            randomize_offset=base.randomize_offset,
        )
        results[n] = run_monte_carlo(cfg)
    return results


def run_toa_noise_sweep(
    toa_sigmas_us: tuple[float, ...] = (50.0, 100.0, 200.0),
    *,
    base_config: MonteCarloConfig | None = None,
) -> dict[float, MonteCarloResult]:
    """Monte Carlo vs TOA noise (microseconds)."""
    base = base_config or MonteCarloConfig()
    results: dict[float, MonteCarloResult] = {}
    for i, sigma_us in enumerate(toa_sigmas_us):
        cfg = MonteCarloConfig(
            n_trials=base.n_trials,
            seed=base.seed,
            preset=base.preset,
            duration_s=base.duration_s,
            step_s=base.step_s,
            toa_sigma_s=sigma_us * 1e-6,
            n_pulsars=base.n_pulsars,
            policies=base.policies,
            randomize_offset=base.randomize_offset,
        )
        results[sigma_us] = run_monte_carlo(cfg)
    return results
