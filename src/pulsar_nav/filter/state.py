"""Navigation state vector for hybrid / XNAV EKF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 10-state vector (pitch): position, velocity, clock bias/drift, spare timing states.
NAV_STATE_DIM = 10

idx_pos = slice(0, 3)
idx_vel = slice(3, 6)
idx_clock_bias = 6
idx_clock_drift = 7
idx_spare = slice(8, 10)


@dataclass
class NavState:
    """Navigation state in ICRS (m, m/s) with clock states in meters / (m/s)."""

    vector: np.ndarray

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=float).reshape(NAV_STATE_DIM)

    @classmethod
    def zeros(cls) -> NavState:
        return cls(np.zeros(NAV_STATE_DIM))

    @classmethod
    def from_pv(
        cls,
        position_m: np.ndarray,
        velocity_m_s: np.ndarray,
        clock_bias_m: float = 0.0,
        clock_drift_m_s: float = 0.0,
    ) -> NavState:
        x = np.zeros(NAV_STATE_DIM)
        x[idx_pos] = position_m
        x[idx_vel] = velocity_m_s
        x[idx_clock_bias] = clock_bias_m
        x[idx_clock_drift] = clock_drift_m_s
        return cls(x)

    @property
    def position_m(self) -> np.ndarray:
        return self.vector[idx_pos].copy()

    @property
    def velocity_m_s(self) -> np.ndarray:
        return self.vector[idx_vel].copy()

    @property
    def clock_bias_m(self) -> float:
        return float(self.vector[idx_clock_bias])

    @property
    def clock_drift_m_s(self) -> float:
        return float(self.vector[idx_clock_drift])

    def copy(self) -> NavState:
        return NavState(self.vector.copy())
