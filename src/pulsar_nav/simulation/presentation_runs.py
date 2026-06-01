"""Representative filter traces and error envelopes for presentation figures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar_nav.catalog.pulsar import Pulsar
from pulsar_nav.propagation.dynamics import DynamicsConfig, dynamics_config_for_sim
from pulsar_nav.propagation.propagator import LunarPropagator, PropagatedTrajectory
from pulsar_nav.simulation.hybrid_run import HybridRunResult, run_hybrid_ekf
from pulsar_nav.simulation.monte_carlo import MonteCarloConfig, _trial_offset_m, select_pulsars
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.simulation.xnav_run import offset_initial_position
from pulsar_nav.spice.ephemeris import str_to_et
from pulsar_nav.visibility.blackout import VisibilityTimeline, compute_visibility_timeline


@dataclass
class PolicyErrorEnvelope:
    """Across-trial position error statistics at each epoch."""

    policy: NavPolicy
    t_s: np.ndarray
    mean_m: np.ndarray
    p5_m: np.ndarray
    p95_m: np.ndarray
    n_trials: int


def propagate_truth_arc(
    *,
    preset: str = "elfo",
    epoch_utc: str = "2026-01-15T12:00:00",
    duration_s: float,
    step_s: float = 120.0,
    include_disturbances: bool = False,
    dynamics_config: DynamicsConfig | None = None,
) -> tuple[PropagatedTrajectory, VisibilityTimeline]:
    from pulsar_nav.spice.kernels import load_kernels

    load_kernels(load_gps_frames=True)
    et0 = str_to_et(epoch_utc)
    dyn_cfg = dynamics_config or dynamics_config_for_sim(
        include_disturbances=include_disturbances
    )
    prop = LunarPropagator(et0, config=dyn_cfg, auto_load_kernels=False)
    traj = prop.propagate_preset(preset, duration_s=duration_s, step_s=step_s)
    timeline = compute_visibility_timeline(traj)
    return traj, timeline


def _run_policy_trace(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    pulsars: list[Pulsar],
    policy: NavPolicy,
    *,
    offset_m: float,
    trial_rng: np.random.Generator,
    config: MonteCarloConfig,
) -> HybridRunResult:
    init_pos = offset_initial_position(traj.position_icrs_m[0], offset_m, trial_rng)
    init_vel = traj.velocity_icrs_m_s[0].copy()
    meas_rng = np.random.default_rng(trial_rng.integers(0, 2**63 - 1))
    return run_hybrid_ekf(
        traj,
        timeline,
        pulsars,
        initial_position_m=init_pos,
        initial_velocity_m_s=init_vel,
        toa_sigma_s=config.toa_sigma_s,
        gnss_sigma_m=config.gnss_sigma_m,
        lonet_sigma_m=config.lonet_sigma_m,
        process_noise_accel=config.process_noise_accel,
        gravity_scaled_q=config.gravity_scaled_q,
        q_accel_scale=config.q_accel_scale,
        predict_mode=config.predict_mode,
        use_truth_velocity_predict=config.use_truth_velocity_predict,
        use_dynamics_predict=config.use_dynamics_predict,
        dynamics_config=config.dynamics_config(),
        dynamics_sigma_acc_km=config.dynamics_sigma_acc_km,
        dynamics_use_hw2_process_noise=config.dynamics_use_hw2_process_noise,
        rng=meas_rng,
        policy=policy,
    )


def run_representative_policy_runs(
    config: MonteCarloConfig,
    *,
    trial_id: int = 0,
    traj: PropagatedTrajectory | None = None,
    timeline: VisibilityTimeline | None = None,
) -> tuple[dict[NavPolicy, HybridRunResult], float]:
    """
    Run one shared-offset trial for each policy (matches Monte Carlo trial index).

    Returns filter traces and the position offset (m) used.
    """
    if traj is None or timeline is None:
        traj, timeline = propagate_truth_arc(
            preset=config.preset,
            epoch_utc=config.epoch_utc,
            duration_s=config.duration_s,
            step_s=config.step_s,
            dynamics_config=config.dynamics_config(),
        )

    pulsars = select_pulsars(config.n_pulsars)
    master_rng = np.random.default_rng(config.seed)
    offset_m = 0.0
    runs: dict[NavPolicy, HybridRunResult] = {}

    for tid in range(trial_id + 1):
        offset_m = _trial_offset_m(config, master_rng)
        if tid < trial_id:
            for _ in config.policies:
                master_rng.integers(0, 2**63 - 1)
            continue
        for policy in config.policies:
            trial_rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            runs[policy] = _run_policy_trace(
                traj,
                timeline,
                pulsars,
                policy,
                offset_m=offset_m,
                trial_rng=trial_rng,
                config=config,
            )
    return runs, offset_m


def collect_error_envelopes(
    config: MonteCarloConfig,
    *,
    n_trials: int | None = None,
    traj: PropagatedTrajectory | None = None,
    timeline: VisibilityTimeline | None = None,
) -> dict[NavPolicy, PolicyErrorEnvelope]:
    """Stack per-trial error traces; return mean and p5/p95 at each epoch."""
    n = n_trials if n_trials is not None else config.n_trials
    if traj is None or timeline is None:
        traj, timeline = propagate_truth_arc(
            preset=config.preset,
            epoch_utc=config.epoch_utc,
            duration_s=config.duration_s,
            step_s=config.step_s,
            dynamics_config=config.dynamics_config(),
        )

    pulsars = select_pulsars(config.n_pulsars)
    stacks: dict[NavPolicy, list[np.ndarray]] = {p: [] for p in config.policies}
    t_ref: np.ndarray | None = None
    master_rng = np.random.default_rng(config.seed)

    for trial_id in range(n):
        offset_m = _trial_offset_m(config, master_rng)
        for policy in config.policies:
            trial_rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            run = _run_policy_trace(
                traj,
                timeline,
                pulsars,
                policy,
                offset_m=offset_m,
                trial_rng=trial_rng,
                config=config,
            )
            if t_ref is None:
                t_ref = run.t_s.copy()
            stacks[policy].append(run.position_error_m)

    assert t_ref is not None
    envelopes: dict[NavPolicy, PolicyErrorEnvelope] = {}
    for policy, traces in stacks.items():
        arr = np.vstack(traces)
        envelopes[policy] = PolicyErrorEnvelope(
            policy=policy,
            t_s=t_ref,
            mean_m=np.mean(arr, axis=0),
            p5_m=np.percentile(arr, 5, axis=0),
            p95_m=np.percentile(arr, 95, axis=0),
            n_trials=n,
        )
    return envelopes
