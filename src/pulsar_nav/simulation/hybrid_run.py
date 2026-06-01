"""Hybrid navigation EKF with policy-based measurement switching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar_nav.catalog.pulsar import Pulsar
from pulsar_nav.constants import DEFAULT_TOA_SIGMA_S
from pulsar_nav.filter.ekf import PulsarNavEKF
from pulsar_nav.measurements.gnss_meas import gnss_pseudoranges, gnss_sidelobe_los_unit_rows
from pulsar_nav.visibility.gdop import position_dop_from_los
from pulsar_nav.measurements.lonet_meas import lonet_pseudoranges
from pulsar_nav.measurements.xnav import synthesize_measurement
from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.simulation.policy import NavPolicy, PolicySegment, segment_from_measurements
from pulsar_nav.simulation.xnav_run import offset_initial_position
from pulsar_nav.spice.ephemeris import body_position_mci
from pulsar_nav.visibility.blackout import NavMode, VisibilitySample, VisibilityTimeline, compute_visibility_timeline
from pulsar_nav.visibility.lonet import (
    LunaNetConfig,
    construct_walker_constellation,
    propagate_constellation_mci,
)

# Truth clock bias in simulation (synthetic measurements use b_tx = 0).
TRUTH_CLOCK_BIAS_M = 0.0


@dataclass
class HybridEpochLog:
    t_s: float
    nav_mode: NavMode
    policy_segment: PolicySegment
    n_gnss: int
    n_lonet: int
    n_pulsar: int
    in_blackout: bool
    gnss_pdop: float = float("nan")
    nis: float = float("nan")
    n_meas: int = 0
    clock_bias_m: float = 0.0
    clock_timing_error_m: float = float("nan")
    clock_constrained: bool = False


@dataclass
class HybridRunResult:
    """Hybrid filter outputs aligned with truth."""

    t_s: np.ndarray
    truth_position_m: np.ndarray
    est_position_m: np.ndarray
    position_error_m: np.ndarray
    nav_modes: np.ndarray
    policy_segments: np.ndarray
    epoch_logs: list[HybridEpochLog]
    pulsar_names: list[str]
    policy: NavPolicy
    nis: np.ndarray | None = None
    nis_dof: np.ndarray | None = None
    clock_bias_m: np.ndarray | None = None
    clock_timing_error_m: np.ndarray | None = None
    clock_constrained: np.ndarray | None = None

    @property
    def final_position_error_m(self) -> float:
        return float(self.position_error_m[-1])

    @property
    def mean_position_error_m(self) -> float:
        return float(np.mean(self.position_error_m))

    @property
    def rms_position_error_m(self) -> float:
        return float(np.sqrt(np.mean(self.position_error_m**2)))

    def timing_errors_constrained(self) -> np.ndarray:
        """|b_rx − b_truth| on epochs with GNSS and/or LunaNet pseudoranges."""
        if self.clock_timing_error_m is None or self.clock_constrained is None:
            return np.array([], dtype=float)
        mask = self.clock_constrained.astype(bool)
        return self.clock_timing_error_m[mask]

    @property
    def final_constrained_timing_error_m(self) -> float:
        if self.clock_timing_error_m is None or self.clock_constrained is None:
            return float("nan")
        for i in range(len(self.clock_constrained) - 1, -1, -1):
            if self.clock_constrained[i]:
                return float(self.clock_timing_error_m[i])
        return float("nan")

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


def timing_metrics_from_run(result: HybridRunResult, policy: NavPolicy) -> dict[str, float]:
    """
    |b_rx − b_truth| over pseudorange epochs (GNSS and/or LunaNet).

    XNAV-only and MSP-only blackout segments do not observe the clock in H;
    returns NaNs for ``NavPolicy.XNAV_ONLY``.
    """
    nan = float("nan")
    if policy == NavPolicy.XNAV_ONLY:
        return {"timing_mean_m": nan, "timing_final_m": nan, "timing_p95_m": nan}
    constrained = result.timing_errors_constrained()
    if constrained.size == 0:
        return {"timing_mean_m": nan, "timing_final_m": nan, "timing_p95_m": nan}
    return {
        "timing_mean_m": float(np.mean(constrained)),
        "timing_final_m": result.final_constrained_timing_error_m,
        "timing_p95_m": float(np.percentile(constrained, 95.0)),
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


def _xnav_measurements(
    pulsars: list[Pulsar],
    truth_position_m: np.ndarray,
    rng: np.random.Generator,
    toa_sigma_s: float,
) -> list:
    return [
        synthesize_measurement(p, truth_position_m, rng, toa_sigma_s)
        for p in pulsars
    ]


def _lonet_if_visible(
    sample: VisibilitySample,
    *,
    truth_position_m: np.ndarray,
    relay_pos_km: np.ndarray,
    et: float,
    et0: float,
    rng: np.random.Generator,
    lonet_sigma_m: float,
    lonet_config: LunaNetConfig,
) -> list:
    if not sample.lonet_visible:
        return []
    return lonet_pseudoranges(
        truth_position_m,
        relay_pos_km,
        et,
        rng,
        sigma_m=lonet_sigma_m,
        et0=et0,
        lonet_config=lonet_config,
    )


def measurements_for_epoch(
    sample: VisibilitySample,
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
    xnav_fallback_on_empty_gnss: bool = True,
) -> tuple[list, list, list]:
    """
    Return (gnss_meas, lonet_meas, xnav_meas) for this epoch.

    Three primary phases (``NavPolicy``), with **LunaNet supplemental** when
    ``sample.lonet_visible`` during blackout (all policies except ``GNSS_COAST``).

    - ``XNAV_ONLY``: pulsars every epoch.
    - ``GNSS_ONLY``: GNSS sidelobe PRNs when not in blackout; pulsars in blackout
      (+ supplemental LunaNet when relay visible).
    - ``HYBRID``: non-blackout **fuses** GNSS (+ LunaNet if relay) with pulsars;
      blackout uses pulsars + supplemental LunaNet when relay visible.
    - ``XNAV_ONLY``: pulsars every epoch; supplemental LunaNet in blackout when relay visible.
    - If a non-blackout epoch has 0 trackable PRNs, fall back to pulsars
      (``xnav_fallback_on_empty_gnss``).
    """
    xnav: list = []
    gnss: list = []
    lonet: list = []

    if policy == NavPolicy.XNAV_ONLY:
        xnav = _xnav_measurements(pulsars, truth_position_m, rng, toa_sigma_s)
        if sample.in_blackout:
            lonet = _lonet_if_visible(
                sample,
                truth_position_m=truth_position_m,
                relay_pos_km=relay_pos_km,
                et=et,
                et0=et0,
                rng=rng,
                lonet_sigma_m=lonet_sigma_m,
                lonet_config=lonet_config,
            )
        return gnss, lonet, xnav

    if sample.in_blackout:
        if policy == NavPolicy.GNSS_COAST:
            return gnss, lonet, xnav
        xnav = _xnav_measurements(pulsars, truth_position_m, rng, toa_sigma_s)
        lonet = _lonet_if_visible(
            sample,
            truth_position_m=truth_position_m,
            relay_pos_km=relay_pos_km,
            et=et,
            et0=et0,
            rng=rng,
            lonet_sigma_m=lonet_sigma_m,
            lonet_config=lonet_config,
        )
        return gnss, lonet, xnav

    # Non-blackout: geometric GNSS window (sidelobe PRNs may still be 0)
    gnss = gnss_pseudoranges(
        truth_position_m,
        earth_mci_km,
        et,
        rng,
        sigma_m=gnss_sigma_m,
        et0=et0,
    )

    if policy == NavPolicy.HYBRID:
        lonet = _lonet_if_visible(
            sample,
            truth_position_m=truth_position_m,
            relay_pos_km=relay_pos_km,
            et=et,
            et0=et0,
            rng=rng,
            lonet_sigma_m=lonet_sigma_m,
            lonet_config=lonet_config,
        )
        # Joint update: keep pulsar geometry when sidelobe GNSS is collinear (periapsis).
        xnav = _xnav_measurements(pulsars, truth_position_m, rng, toa_sigma_s)

    elif (
        xnav_fallback_on_empty_gnss
        and not gnss
        and policy == NavPolicy.GNSS_ONLY
    ):
        xnav = _xnav_measurements(pulsars, truth_position_m, rng, toa_sigma_s)

    return gnss, lonet, xnav


# Backward-compatible alias
def measurements_for_mode(
    mode: NavMode,
    *,
    sample: VisibilitySample | None = None,
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
    if sample is None:
        raise ValueError("sample is required")
    return measurements_for_epoch(
        sample,
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
    """EKF on a propagated truth arc with a switching ``NavPolicy``."""
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
    nis_arr = np.full(n, np.nan)
    dof_arr = np.zeros(n, dtype=int)
    clock_bias_arr = np.zeros(n)
    clock_timing_arr = np.full(n, np.nan)
    clock_constrained_arr = np.zeros(n, dtype=bool)
    modes = np.array([s.nav_mode.value for s in timeline.samples], dtype=object)
    segments = np.empty(n, dtype=object)
    logs: list[HybridEpochLog] = []

    for i in range(n):
        sample = timeline.samples[i]
        gnss: list = []
        lonet: list = []
        xnav: list = []
        nis = float("nan")
        n_meas = 0

        if i > 0:
            dt = float(traj.t_rel_s[i] - traj.t_rel_s[i - 1])
            if use_truth_velocity_predict:
                ekf.predict_kinematic(dt, truth_vel[i - 1])
            else:
                ekf.predict(dt)

            earth = body_position_mci("EARTH", traj.et[i])
            relay_km = relays[:, i, :]
            gnss, lonet, xnav = measurements_for_epoch(
                sample,
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
                nis = ekf.last_nis
                n_meas = ekf.last_dof
                nis_arr[i] = nis
                dof_arr[i] = n_meas

        est_pos[i] = ekf.state.position_m
        pos_err[i] = np.linalg.norm(est_pos[i] - truth_pos[i])

        pr_constrained = i > 0 and bool(gnss or lonet)
        bias_m = float(ekf.state.clock_bias_m)
        clock_bias_arr[i] = bias_m
        clock_constrained_arr[i] = pr_constrained
        if pr_constrained:
            clock_timing_arr[i] = abs(bias_m - TRUTH_CLOCK_BIAS_M)

        gnss_pdop = float("nan")
        if gnss:
            los = gnss_sidelobe_los_unit_rows(truth_pos[i], traj.et[i])
            gnss_pdop = position_dop_from_los(los)

        seg = segment_from_measurements(
            policy,
            sample,
            n_gnss=len(gnss),
            n_lonet=len(lonet),
            n_pulsar=len(xnav),
        )
        segments[i] = seg.value

        logs.append(
            HybridEpochLog(
                t_s=float(traj.t_rel_s[i]),
                nav_mode=sample.nav_mode,
                policy_segment=seg,
                n_gnss=len(gnss),
                n_lonet=len(lonet),
                n_pulsar=len(xnav),
                in_blackout=sample.in_blackout,
                gnss_pdop=gnss_pdop,
                nis=nis,
                n_meas=n_meas,
                clock_bias_m=bias_m,
                clock_timing_error_m=clock_timing_arr[i],
                clock_constrained=pr_constrained,
            )
        )

    return HybridRunResult(
        t_s=traj.t_rel_s.copy(),
        truth_position_m=truth_pos,
        est_position_m=est_pos,
        position_error_m=pos_err,
        nav_modes=modes,
        policy_segments=segments,
        epoch_logs=logs,
        pulsar_names=[p.name for p in pulsars],
        policy=policy,
        nis=nis_arr,
        nis_dof=dof_arr,
        clock_bias_m=clock_bias_arr,
        clock_timing_error_m=clock_timing_arr,
        clock_constrained=clock_constrained_arr,
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
    """XNAV-only baseline (no GNSS; LunaNet supplement in blackout when relay visible)."""
    kwargs.pop("policy", None)
    return run_hybrid_on_propagated(
        traj, pulsars, policy=NavPolicy.XNAV_ONLY, **kwargs
    )
