"""Hybrid navigation EKF: XNAV always, augmented by GNSS / LunaNet when visible."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pulsar_nav.catalog.pulsar import Pulsar
from pulsar_nav.constants import DEFAULT_TOA_SIGMA_S
from pulsar_nav.filter.ekf import PulsarNavEKF
from pulsar_nav.measurements.gnss_meas import gnss_pseudoranges
from pulsar_nav.measurements.lonet_meas import lonet_pseudoranges
from pulsar_nav.measurements.xnav import synthesize_measurement
from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.simulation.policy import NavPolicy
from pulsar_nav.simulation.xnav_run import offset_initial_position
from pulsar_nav.spice.ephemeris import body_position_mci
from pulsar_nav.visibility.blackout import NavMode, VisibilityTimeline, compute_visibility_timeline
from pulsar_nav.visibility.lonet import (
    LunaNetConfig,
    construct_walker_constellation,
    propagate_constellation_mci,
)


@dataclass
class HybridEpochLog:
    t_s: float
    nav_mode: NavMode
    n_gnss: int
    n_lonet: int
    n_pulsar: int
    in_blackout: bool


@dataclass
class HybridRunResult:
    """Hybrid filter outputs aligned with truth."""

    t_s: np.ndarray
    truth_position_m: np.ndarray
    est_position_m: np.ndarray
    position_error_m: np.ndarray
    nav_modes: np.ndarray
    epoch_logs: list[HybridEpochLog]
    pulsar_names: list[str]

    @property
    def final_position_error_m(self) -> float:
        return float(self.position_error_m[-1])

    @property
    def mean_position_error_m(self) -> float:
        return float(np.mean(self.position_error_m))

    @property
    def rms_position_error_m(self) -> float:
        return float(np.sqrt(np.mean(self.position_error_m**2)))

    def segment_errors(self, timeline: VisibilityTimeline) -> dict[str, float]:
        """Mean 3D error (m) over blackout vs non-blackout samples."""
        blackout_err: list[float] = []
        clear_err: list[float] = []
        for err, sample in zip(self.position_error_m, timeline.samples):
            if sample.in_blackout:
                blackout_err.append(float(err))
            else:
                clear_err.append(float(err))
        return {
            "blackout_mean_m": float(np.mean(blackout_err)) if blackout_err else float("nan"),
            "non_blackout_mean_m": float(np.mean(clear_err)) if clear_err else float("nan"),
        }


def _build_relay_positions(
    traj: PropagatedTrajectory,
    lonet_config: LunaNetConfig,
) -> np.ndarray:
    cfg = lonet_config
    coes = construct_walker_constellation(
        cfg.sma_km,
        cfg.eccentricity,
        cfg.inclination_rad,
        cfg.argp_rad,
        cfg.walker_f,
        cfg.n_sats,
        cfg.n_planes,
    )
    return propagate_constellation_mci(coes, traj.t_rel_s)


def measurements_for_epoch(
    mode: NavMode,
    policy: NavPolicy,
    *,
    truth_position_m: np.ndarray,
    et: float,
    et0: float,
    earth_mci_km: np.ndarray,
    relay_pos_km: np.ndarray,
    pulsars: list[Pulsar],
    rng: np.random.Generator,
    toa_sigma_s: float,
    gnss_sigma_m: float,
    lonet_sigma_m: float,
    lonet_config: LunaNetConfig,
) -> tuple[list, list, list]:
    """
    Return (gnss_meas, lonet_meas, xnav_meas) for this epoch.

    ``NavPolicy.HYBRID``: XNAV every epoch; GNSS when not in blackout;
    LunaNet when relays are visible.
    """
    xnav: list = []
    gnss: list = []
    lonet: list = []

    if policy in (NavPolicy.HYBRID, NavPolicy.XNAV_ONLY):
        xnav = [
            synthesize_measurement(p, truth_position_m, rng, toa_sigma_s)
            for p in pulsars
        ]

    if policy in (NavPolicy.HYBRID, NavPolicy.GNSS_ONLY) and mode in (
        NavMode.GNSS,
        NavMode.HYBRID,
    ):
        gnss = gnss_pseudoranges(
            truth_position_m,
            earth_mci_km,
            et,
            rng,
            sigma_m=gnss_sigma_m,
            et0=et0,
        )

    if policy in (NavPolicy.HYBRID, NavPolicy.LONET_ONLY) and mode in (
        NavMode.HYBRID,
        NavMode.LONET,
    ):
        lonet = lonet_pseudoranges(
            truth_position_m,
            relay_pos_km,
            et,
            rng,
            sigma_m=lonet_sigma_m,
            et0=et0,
            lonet_config=lonet_config,
        )

    return gnss, lonet, xnav


# Backward-compatible alias
def measurements_for_mode(
    mode: NavMode,
    *,
    truth_position_m: np.ndarray,
    truth_velocity_m_s: np.ndarray,
    et: float,
    et0: float,
    earth_mci_km: np.ndarray,
    relay_pos_km: np.ndarray,
    pulsars: list[Pulsar],
    rng: np.random.Generator,
    toa_sigma_s: float,
    gnss_sigma_m: float,
    lonet_sigma_m: float,
    lonet_config: LunaNetConfig,
    policy: NavPolicy = NavPolicy.HYBRID,
) -> tuple[list, list, list]:
    return measurements_for_epoch(
        mode,
        policy,
        truth_position_m=truth_position_m,
        et=et,
        et0=et0,
        earth_mci_km=earth_mci_km,
        relay_pos_km=relay_pos_km,
        pulsars=pulsars,
        rng=rng,
        toa_sigma_s=toa_sigma_s,
        gnss_sigma_m=gnss_sigma_m,
        lonet_sigma_m=lonet_sigma_m,
        lonet_config=lonet_config,
    )


def run_hybrid_ekf(
    traj: PropagatedTrajectory,
    timeline: VisibilityTimeline,
    pulsars: list[Pulsar],
    *,
    initial_position_m: np.ndarray,
    initial_velocity_m_s: np.ndarray,
    position_sigma_m: float = 100_000.0,
    velocity_sigma_m_s: float = 100.0,
    process_noise_accel: float = 1e-3,
    use_truth_velocity_predict: bool = True,
    toa_sigma_s: float = DEFAULT_TOA_SIGMA_S,
    gnss_sigma_m: float = 15.0,
    lonet_sigma_m: float = 15.0,
    lonet_config: LunaNetConfig | None = None,
    relay_positions: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    policy: NavPolicy = NavPolicy.HYBRID,
) -> HybridRunResult:
    """
    Mode-switching EKF on a propagated truth arc.

    At each step after the initial epoch, applies XNAV pulsars always, then
    augments with GNSS (non-blackout) and/or LunaNet per ``NavMode``.
    """
    if len(timeline.samples) != len(traj.t_rel_s):
        raise ValueError("timeline and trajectory must have the same length")

    rng = rng or np.random.default_rng(0)
    cfg = lonet_config or LunaNetConfig()
    relays = relay_positions if relay_positions is not None else _build_relay_positions(traj, cfg)

    ekf = PulsarNavEKF.from_initial(
        initial_position_m,
        initial_velocity_m_s,
        position_sigma_m=position_sigma_m,
        velocity_sigma_m_s=velocity_sigma_m_s,
    )
    ekf.process_noise_accel = process_noise_accel

    n = len(traj.t_rel_s)
    truth_pos = traj.position_icrs_m.copy()
    truth_vel = traj.velocity_icrs_m_s.copy()
    est_pos = np.zeros((n, 3))
    pos_err = np.zeros(n)
    modes = np.array([s.nav_mode.value for s in timeline.samples], dtype=object)
    logs: list[HybridEpochLog] = []

    for i in range(n):
        sample = timeline.samples[i]
        gnss: list = []
        lonet: list = []
        xnav: list = []

        if i > 0:
            dt = float(traj.t_rel_s[i] - traj.t_rel_s[i - 1])
            if use_truth_velocity_predict:
                ekf.predict_kinematic(dt, truth_vel[i - 1])
            else:
                ekf.predict(dt)

            earth = body_position_mci("EARTH", traj.et[i])
            relay_km = relays[:, i, :]
            gnss, lonet, xnav = measurements_for_epoch(
                sample.nav_mode,
                policy,
                truth_position_m=truth_pos[i],
                et=traj.et[i],
                et0=traj.et0,
                earth_mci_km=earth,
                relay_pos_km=relay_km,
                pulsars=pulsars,
                rng=rng,
                toa_sigma_s=toa_sigma_s,
                gnss_sigma_m=gnss_sigma_m,
                lonet_sigma_m=lonet_sigma_m,
                lonet_config=cfg,
            )

            pr_meas = gnss + lonet
            if pr_meas or xnav:
                ekf.update_navigation_epoch(
                    xnav,
                    pr_meas,
                    traj.et[i],
                    et0=traj.et0,
                )

        est_pos[i] = ekf.state.position_m
        pos_err[i] = np.linalg.norm(est_pos[i] - truth_pos[i])

        logs.append(
            HybridEpochLog(
                t_s=float(traj.t_rel_s[i]),
                nav_mode=sample.nav_mode,
                n_gnss=len(gnss),
                n_lonet=len(lonet),
                n_pulsar=len(xnav),
                in_blackout=sample.in_blackout,
            )
        )

    return HybridRunResult(
        t_s=traj.t_rel_s.copy(),
        truth_position_m=truth_pos,
        est_position_m=est_pos,
        position_error_m=pos_err,
        nav_modes=modes,
        epoch_logs=logs,
        pulsar_names=[p.name for p in pulsars],
    )


def run_hybrid_on_propagated(
    traj: PropagatedTrajectory,
    pulsars: list[Pulsar],
    *,
    position_offset_m: float = 50_000.0,
    timeline: VisibilityTimeline | None = None,
    policy: NavPolicy = NavPolicy.HYBRID,
    **kwargs,
) -> HybridRunResult:
    """Hybrid filter with perturbed initial state on propagated truth."""
    samples = traj.samples()
    rng = kwargs.pop("rng", None) or np.random.default_rng(1)
    init_pos = offset_initial_position(
        samples[0].position_m,
        position_offset_m,
        rng,
    )
    init_vel = samples[0].velocity_m_s.copy()
    tl = timeline or compute_visibility_timeline(traj)
    return run_hybrid_ekf(
        traj,
        tl,
        pulsars,
        initial_position_m=init_pos,
        initial_velocity_m_s=init_vel,
        policy=policy,
        **kwargs,
    )


def run_xnav_only_on_propagated(
    traj: PropagatedTrajectory,
    pulsars: list[Pulsar],
    **kwargs,
) -> HybridRunResult:
    """XNAV-only baseline (no GNSS / LunaNet)."""
    kwargs.pop("policy", None)
    return run_hybrid_on_propagated(
        traj, pulsars, policy=NavPolicy.XNAV_ONLY, **kwargs
    )
