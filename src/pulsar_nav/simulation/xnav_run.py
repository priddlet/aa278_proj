"""Run XNAV EKF against a propagated truth trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar_nav.catalog.pulsar import Pulsar
from pulsar_nav.constants import DEFAULT_TOA_SIGMA_S
from pulsar_nav.filter.ekf import PulsarNavEKF
from pulsar_nav.measurements.xnav import synthesize_measurement
from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.simulation.truth import TrajectorySample


@dataclass
class XNAVRunResult:
    """EKF outputs aligned with truth samples."""

    t_s: np.ndarray
    truth_position_m: np.ndarray
    truth_velocity_m_s: np.ndarray
    est_position_m: np.ndarray
    est_velocity_m_s: np.ndarray
    position_error_m: np.ndarray
    los_error_m: np.ndarray | None  # only when single pulsar
    pulsar_names: list[str]
    innovations: list[list[float]]

    @property
    def final_position_rmse_m(self) -> float:
        return float(self.position_error_m[-1])

    @property
    def mean_position_error_m(self) -> float:
        return float(np.mean(self.position_error_m))


def offset_initial_position(
    truth_position_m: np.ndarray,
    offset_m: float,
    rng: np.random.Generator,
    *,
    pulsar_los: np.ndarray | None = None,
    make_unobservable: bool = False,
) -> np.ndarray:
    """
    Perturb initial position by ``offset_m``.

    If ``make_unobservable`` and ``pulsar_los`` given, offset is perpendicular to LOS.
    """
    direction = rng.standard_normal(3)
    if pulsar_los is not None and make_unobservable:
        n = pulsar_los / np.linalg.norm(pulsar_los)
        direction = direction - n * np.dot(direction, n)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0]) - n * n[0]
        direction = direction / np.linalg.norm(direction)
    else:
        direction = direction / np.linalg.norm(direction)
    return truth_position_m + offset_m * direction


def run_xnav_ekf(
    samples: list[TrajectorySample],
    pulsars: list[Pulsar],
    *,
    initial_position_m: np.ndarray,
    initial_velocity_m_s: np.ndarray,
    dt_s: float | None = None,
    toa_sigma_s: float = DEFAULT_TOA_SIGMA_S,
    position_sigma_m: float = 100_000.0,
    velocity_sigma_m_s: float = 100.0,
    process_noise_accel: float = 1e-3,
    use_truth_velocity_predict: bool = True,
    rng: np.random.Generator | None = None,
) -> XNAVRunResult:
    """
    Filter a truth arc with sequential XNAV LOS measurements (ICRS, meters).
    """
    if len(samples) < 2:
        raise ValueError("Need at least two trajectory samples")

    rng = rng or np.random.default_rng(0)
    ekf = PulsarNavEKF.from_initial(
        initial_position_m,
        initial_velocity_m_s,
        position_sigma_m=position_sigma_m,
        velocity_sigma_m_s=velocity_sigma_m_s,
    )
    ekf.process_noise_accel = process_noise_accel

    n = len(samples)
    truth_pos = np.zeros((n, 3))
    truth_vel = np.zeros((n, 3))
    est_pos = np.zeros((n, 3))
    est_vel = np.zeros((n, 3))
    pos_err = np.zeros(n)
    los_err = np.zeros(n) if len(pulsars) == 1 else None
    innovations: list[list[float]] = []

    n_hat = pulsars[0].unit_vector_icrs if len(pulsars) == 1 else None

    for i, sample in enumerate(samples):
        truth_pos[i] = sample.position_m
        truth_vel[i] = sample.velocity_m_s

        if i > 0:
            step_dt = dt_s if dt_s is not None else float(sample.t_s - samples[i - 1].t_s)
            meas = [
                synthesize_measurement(p, sample.position_m, rng, toa_sigma_s)
                for p in pulsars
            ]
            if use_truth_velocity_predict:
                ekf.predict_kinematic(step_dt, samples[i - 1].velocity_m_s)
            else:
                ekf.predict(step_dt)
            innovations.append(ekf.update_epoch(meas))
            ekf._history.append(ekf.state.copy())
        else:
            innovations.append([])

        est_pos[i] = ekf.state.position_m
        est_vel[i] = ekf.state.velocity_m_s
        pos_err[i] = np.linalg.norm(est_pos[i] - truth_pos[i])
        if los_err is not None and n_hat is not None:
            los_err[i] = abs(np.dot(est_pos[i] - truth_pos[i], n_hat))

    return XNAVRunResult(
        t_s=np.array([s.t_s for s in samples]),
        truth_position_m=truth_pos,
        truth_velocity_m_s=truth_vel,
        est_position_m=est_pos,
        est_velocity_m_s=est_vel,
        position_error_m=pos_err,
        los_error_m=los_err,
        pulsar_names=[p.name for p in pulsars],
        innovations=innovations,
    )


def run_xnav_on_propagated(
    traj: PropagatedTrajectory,
    pulsars: list[Pulsar],
    *,
    position_offset_m: float = 50_000.0,
    unobservable_offset: bool = False,
    **kwargs,
) -> XNAVRunResult:
    """Run EKF on ``PropagatedTrajectory.samples()`` with perturbed initial position."""
    samples = traj.samples()
    rng = kwargs.pop("rng", None) or np.random.default_rng(1)
    pulsar_los = pulsars[0].unit_vector_icrs if len(pulsars) == 1 else None
    init_pos = offset_initial_position(
        samples[0].position_m,
        position_offset_m,
        rng,
        pulsar_los=pulsar_los,
        make_unobservable=unobservable_offset,
    )
    init_vel = samples[0].velocity_m_s.copy()
    return run_xnav_ekf(
        samples,
        pulsars,
        initial_position_m=init_pos,
        initial_velocity_m_s=init_vel,
        **kwargs,
    )
