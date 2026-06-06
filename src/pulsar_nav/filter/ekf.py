"""Extended Kalman filter for pulsar-only navigation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pulsar_nav.filter.state import NAV_STATE_DIM, NavState, idx_clock_bias, idx_clock_drift
from pulsar_nav.measurements.pseudorange import (
    PseudorangeMeasurement,
    pseudorange_jacobian_m,
    pseudorange_residual,
)
from pulsar_nav.filter.consistency import normalized_innovation_squared
from pulsar_nav.filter.dynamics_predict import full_state_transition
from pulsar_nav.filter.hw2_process_noise import (
    DEFAULT_DYNAMICS_SIGMA_ACC_KM,
    process_noise_hw2,
)
from pulsar_nav.propagation.dynamics import DynamicsConfig
from pulsar_nav.measurements.xnav import (
    XNAVMeasurement,
    measurement_jacobian,
    range_residual,
)


def _gps_light_time_callback(sat_id: int | str):
    """Per-GPS-satellite light-time correction for broadcast ephemeris."""
    sid = str(sat_id)
    if not sid.startswith("G"):
        return None
    prn = int(sid[1:])

    def get_tx(et_tx: float, p: int = prn) -> tuple[np.ndarray, float]:
        from pulsar_nav.ephemeris.gps_posclk import get_gps_posclk_mci

        return get_gps_posclk_mci(et_tx, p)

    return get_tx


def joseph_covariance_update(
    covariance: np.ndarray,
    kalman_gain: np.ndarray,
    measurement_jacobian: np.ndarray,
    measurement_covariance: np.ndarray,
) -> np.ndarray:
    """
    Joseph-form covariance after a stacked measurement update.

    P+ = (I - K H) P (I - K H)^T + K R K^T - symmetric and PSD-preserving
    under ideal arithmetic (preferred over (I - K H) P).
    """
    n = covariance.shape[0]
    i_kh = np.eye(n) - kalman_gain @ measurement_jacobian
    return i_kh @ covariance @ i_kh.T + kalman_gain @ measurement_covariance @ kalman_gain.T


@dataclass
class PulsarNavEKF:
    """
    EKF with constant-velocity dynamics and XNAV LOS range measurements.

    Measurement model (pitch / Sheikh linearized):
        z = n_hat  |  r + noise
    """

    state: NavState
    covariance: np.ndarray
    process_noise_accel: float = 1e-6  # m^2/s^3 (tunable; CV / truth_velocity)
    process_noise_clock: float = 1.0  # m^2/s (clock random walk; CV)
    dynamics_sigma_acc_km: float = DEFAULT_DYNAMICS_SIGMA_ACC_KM
    dynamics_use_hw2_process_noise: bool = True

    last_nis: float = float("nan")
    last_dof: int = 0

    _history: list[NavState] = field(default_factory=list, repr=False)
    nis_history: list[tuple[float, int]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.covariance.shape != (NAV_STATE_DIM, NAV_STATE_DIM):
            raise ValueError(f"covariance must be ({NAV_STATE_DIM}, {NAV_STATE_DIM})")

    @classmethod
    def from_initial(
        cls,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        position_sigma_m: float = 100_000.0,
        velocity_sigma_m_s: float = 100.0,
    ) -> PulsarNavEKF:
        state = NavState.from_pv(position_m, velocity_m_s)
        P = np.diag(
            [position_sigma_m**2] * 3
            + [velocity_sigma_m_s**2] * 3
            + [1e6, 1e2, 1e4, 1e4]
        )
        return cls(state=state, covariance=P)

    def state_transition(self, dt_s: float) -> np.ndarray:
        """Phi for CV model with clock bias integrated from drift."""
        Phi = np.eye(NAV_STATE_DIM)
        for i in range(3):
            Phi[i, i + 3] = dt_s
        Phi[idx_clock_bias, idx_clock_drift] = dt_s
        return Phi

    def process_noise(self, dt_s: float) -> np.ndarray:
        """Simplified Q for position/velocity (CV) and clock states."""
        q_a = self.process_noise_accel
        q_c = self.process_noise_clock
        # Position/velocity block (continuous white noise on acceleration).
        q_block = np.array(
            [
                [dt_s**4 / 4, dt_s**3 / 2],
                [dt_s**3 / 2, dt_s**2],
            ]
        ) * q_a
        Q = np.zeros((NAV_STATE_DIM, NAV_STATE_DIM))
        for i in range(3):
            Q[i, i] = q_block[0, 0]
            Q[i + 3, i + 3] = q_block[1, 1]
            Q[i, i + 3] = q_block[0, 1]
            Q[i + 3, i] = q_block[1, 0]
        Q[idx_clock_bias, idx_clock_bias] = q_c * dt_s**2
        Q[idx_clock_drift, idx_clock_drift] = q_c * dt_s
        return Q

    def _record_nis(self, y: np.ndarray, S: np.ndarray) -> None:
        """Store NIS = y^T S^{-1} y; target median(NIS/dof) ~ 1 when consistent."""
        y = np.asarray(y, float).ravel()
        self.last_nis = normalized_innovation_squared(y, S)
        self.last_dof = int(y.size)
        self.nis_history.append((self.last_nis, self.last_dof))

    def predict(self, dt_s: float) -> None:
        Phi = self.state_transition(dt_s)
        Q = self.process_noise(dt_s)
        self.state = NavState(Phi @ self.state.vector)
        self.covariance = Phi @ self.covariance @ Phi.T + Q

    def predict_kinematic(
        self,
        dt_s: float,
        velocity_m_s: np.ndarray,
        *,
        add_process_noise: bool = True,
    ) -> None:
        """
        Time update using an external velocity (e.g. truth velocity in simulation).

        Covariance is propagated with the same CV Phi/Q as ``predict`` so
        process noise can absorb dynamics model mismatch in real filters.
        """
        vec = self.state.vector.copy()
        vec[0:3] += np.asarray(velocity_m_s, float) * dt_s
        vec[idx_clock_bias] += vec[idx_clock_drift] * dt_s
        Phi = self.state_transition(dt_s)
        Q = self.process_noise(dt_s) if add_process_noise else 0.0
        self.state = NavState(vec)
        self.covariance = Phi @ self.covariance @ Phi.T + Q

    def predict_dynamics(
        self,
        dt_s: float,
        et: float,
        *,
        dynamics_config: DynamicsConfig | None = None,
        use_hw2_process_noise: bool | None = None,
        sigma_acc_km: float | None = None,
    ) -> None:
        """
        Time update with MCI point-mass + third-body acceleration on the estimate.

        Tier A: ``solve_ivp`` RK45 mean propagation + analytic STM (dPhi/dt = JPhi) in MCI;
        optional HW2-style CWNA process noise (sigma_acc in km/s^2/sqrt(s)).
        """
        cfg = dynamics_config or DynamicsConfig()
        phi, vec = full_state_transition(
            self.state.position_m,
            self.state.velocity_m_s,
            self.state.clock_bias_m,
            self.state.clock_drift_m_s,
            et,
            dt_s,
            cfg,
            NAV_STATE_DIM,
        )
        if use_hw2_process_noise if use_hw2_process_noise is not None else self.dynamics_use_hw2_process_noise:
            sig = sigma_acc_km if sigma_acc_km is not None else self.dynamics_sigma_acc_km
            q = process_noise_hw2(dt_s, sigma_acc_km=sig, nav_state_dim=NAV_STATE_DIM)
        else:
            q = self.process_noise(dt_s)
        self.state = NavState(vec)
        self.covariance = phi @ self.covariance @ phi.T + q

    def update(self, measurement: XNAVMeasurement) -> float:
        """Process one pulsar measurement; return innovation (m)."""
        innovations = self.update_epoch([measurement])
        return innovations[0]

    def update_pseudorange_epoch(
        self,
        measurements: list[PseudorangeMeasurement],
        et_rx: float,
        *,
        et0: float | None = None,
    ) -> list[float]:
        """Apply stacked pseudorange updates at one epoch (meters)."""
        if not measurements:
            return []

        H = np.vstack(
            [
                pseudorange_jacobian_m(self.state, m.tx_position_mci_km, et_rx)
                for m in measurements
            ]
        )
        y = np.array(
            [
                pseudorange_residual(
                    m, self.state, et_rx, et0=et0
                )
                for m in measurements
            ]
        )
        R = np.diag([m.sigma_m**2 for m in measurements])
        return self._stacked_measurement_update(H, y, R)

    def _stacked_measurement_update(
        self,
        H: np.ndarray,
        y_vec: np.ndarray,
        R: np.ndarray,
    ) -> list[float]:
        """Apply one stacked update; return innovations (NIS via ``last_nis``)."""
        S = H @ self.covariance @ H.T + R
        self._record_nis(y_vec, S)
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = NavState(self.state.vector + (K @ y_vec).ravel())
        self.covariance = joseph_covariance_update(self.covariance, K, H, R)
        return [float(v) for v in y_vec]

    def update_navigation_epoch(
        self,
        xnav_measurements: list[XNAVMeasurement],
        pseudorange_measurements: list[PseudorangeMeasurement],
        et_rx: float,
        *,
        et0: float | None = None,
    ) -> list[float]:
        """
        Single stacked EKF update for pulsars + pseudoranges at one epoch.

        Sets ``last_nis`` / ``last_dof`` (chi-squared with df = # measurements).
        """
        if not xnav_measurements and not pseudorange_measurements:
            return []

        rows: list[np.ndarray] = []
        y: list[float] = []
        r_var: list[float] = []

        for m in pseudorange_measurements:
            rows.append(pseudorange_jacobian_m(self.state, m.tx_position_mci_km, et_rx))
            get_tx = _gps_light_time_callback(m.sat_id)
            y.append(
                pseudorange_residual(
                    m,
                    self.state,
                    et_rx,
                    et0=et0,
                    get_tx_position=get_tx,
                )
            )
            r_var.append(m.sigma_m**2)

        for m in xnav_measurements:
            rows.append(measurement_jacobian(m.pulsar))
            y.append(range_residual(m, self.state))
            r_var.append(m.sigma_m**2)

        H = np.vstack(rows)
        y_vec = np.array(y)
        R = np.diag(r_var)
        return self._stacked_measurement_update(H, y_vec, R)

    def update_epoch(self, measurements: list[XNAVMeasurement]) -> list[float]:
        """
        Apply all pulsar measurements at one epoch (stacked H, R).

        Sequential scalar updates over-count information when multiple
        pulsars are observed simultaneously.
        """
        if not measurements:
            return []

        H = np.vstack([measurement_jacobian(m.pulsar) for m in measurements])
        y = np.array([range_residual(m, self.state) for m in measurements])
        R = np.diag([m.sigma_m**2 for m in measurements])
        return self._stacked_measurement_update(H, y, R)

    def step(
        self,
        dt_s: float,
        measurements: list[XNAVMeasurement] | None = None,
    ) -> list[float]:
        self.predict(dt_s)
        innovations = []
        for meas in measurements or []:
            innovations.append(self.update(meas))
        self._history.append(self.state.copy())
        return innovations

    @property
    def position_error_m(self) -> float:
        return float(np.linalg.norm(self.state.position_m))

    def position_rmse_vs_truth(self, truth_m: np.ndarray) -> float:
        return float(np.linalg.norm(self.state.position_m - truth_m))
