"""Truth trajectory propagation for lunar orbiters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp

from pulsar_nav.propagation.dynamics import DynamicsConfig, dynamics_ode
from pulsar_nav.propagation.elements import coe_to_cart
from pulsar_nav.spice import ephemeris as spice_ephem
from pulsar_nav.spice import kernels as spice_kernels
from pulsar_nav.spice.ephemeris import mci_to_icrs_position

OrbitPreset = Literal["elfo", "elfo_nav", "llo"]


@dataclass
class PropagatedTrajectory:
    """Truth arc in Moon-centered and ICRS frames."""

    et0: float
    t_rel_s: np.ndarray
    et: np.ndarray
    position_mci_km: np.ndarray
    velocity_mci_km_s: np.ndarray
    position_icrs_m: np.ndarray
    velocity_icrs_m_s: np.ndarray

    def samples(self):
        """List of TrajectorySample for XNAV simulation."""
        from pulsar_nav.simulation.truth import TrajectorySample

        out = []
        for i, t in enumerate(self.t_rel_s):
            out.append(
                TrajectorySample(
                    t_s=float(t),
                    position_m=self.position_icrs_m[i],
                    velocity_m_s=self.velocity_icrs_m_s[i],
                )
            )
        return out


def elfo_initial_coe_op() -> tuple[float, ...]:
    """Frozen elliptical lunar orbit (AA278 HW2 case 1)."""
    return (
        6541.4,
        0.6,
        np.deg2rad(52.0),
        np.deg2rad(300.0),
        np.deg2rad(90.0),
        0.0,
    )


def elfo_nav_initial_coe_op() -> tuple[float, ...]:
    """
    Alternate phasing of HW2 case 1: same elements but argp + 180 deg in OP frame.

    Swaps apoapsis/periapsis orientation relative to the Earth-Moon geometry.
    GNSS visibility differs modestly from the science ``elfo`` preset (not the
    large far-side/near-side split that appeared when COE were mapped via MOON_PA).
    """
    a, e, inc, raan, argp, m0 = elfo_initial_coe_op()
    return (a, e, inc, raan, (argp + np.pi) % (2.0 * np.pi), m0)


def coe_for_preset(preset: OrbitPreset) -> tuple[float, ...]:
    """Classical elements (Moon OP frame) for a named orbit preset."""
    if preset == "elfo":
        return elfo_initial_coe_op()
    if preset == "elfo_nav":
        return elfo_nav_initial_coe_op()
    if preset == "llo":
        return llo_initial_coe_op()
    raise ValueError(f"Unknown preset: {preset!r}")


def llo_initial_coe_op() -> tuple[float, ...]:
    """Low lunar orbit (~100 km altitude, near-circular)."""
    a = 1737.4 + 100.0
    return (a, 0.01, np.deg2rad(90.0), 0.0, 0.0, 0.0)


def initial_state_mci_from_coe_op(
    coe_op: tuple[float, ...],
    et0: float,
) -> np.ndarray:
    """COE in Moon OP frame -> MCI Cartesian at epoch."""
    x_op = coe_to_cart(coe_op)
    rot = spice_ephem.mci_to_op_rotation(et0)
    return rot.T @ x_op


class LunarPropagator:
    """
    High-fidelity Moon-centered propagator (scipy + SPICE ephemeris).

    Matches AA278 HW2 ``ode_dynamics`` force model; optional J2 and SRP.
    """

    def __init__(
        self,
        et0: float,
        config: DynamicsConfig | None = None,
        *,
        kernel_dir: str | None = None,
        auto_load_kernels: bool = True,
    ):
        self.et0 = float(et0)
        self.config = config or DynamicsConfig()
        if auto_load_kernels:
            spice_kernels.load_kernels(kernel_dir)

    def propagate(
        self,
        state0_mci: np.ndarray,
        t_eval_s: np.ndarray,
        *,
        rtol: float = 1e-9,
        atol: float = 1e-9,
    ) -> PropagatedTrajectory:
        """Integrate from state0 = [r, v] in MCI (km, km/s)."""
        t_eval = np.asarray(t_eval_s, dtype=float)
        t0 = float(t_eval[0])
        y0 = np.asarray(state0_mci, dtype=float)

        sol = solve_ivp(
            fun=lambda t, y: dynamics_ode(t, y, self.et0, self.config),
            t_span=(t0, float(t_eval[-1])),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Propagation failed: {sol.message}")

        pos_mci = sol.y[:3].T
        vel_mci = sol.y[3:6].T
        et = self.et0 + t_eval
        pos_icrs = np.zeros_like(pos_mci)
        vel_icrs = np.zeros_like(vel_mci)
        for i, et_i in enumerate(et):
            moon_pos = spice_ephem.moon_position_icrs_km(et_i)
            pos_icrs[i] = (moon_pos + pos_mci[i]) * 1000.0
            moon_vel = spice_ephem.moon_velocity_icrs_km_s(et_i)
            vel_icrs[i] = (moon_vel + vel_mci[i]) * 1000.0

        return PropagatedTrajectory(
            et0=self.et0,
            t_rel_s=t_eval,
            et=et,
            position_mci_km=pos_mci,
            velocity_mci_km_s=vel_mci,
            position_icrs_m=pos_icrs,
            velocity_icrs_m_s=vel_icrs,
        )

    def propagate_coe_op(
        self,
        coe_op: tuple[float, ...],
        t_eval_s: np.ndarray,
        **kwargs,
    ) -> PropagatedTrajectory:
        state0 = initial_state_mci_from_coe_op(coe_op, self.et0)
        return self.propagate(state0, t_eval_s, **kwargs)

    def propagate_preset(
        self,
        preset: OrbitPreset,
        duration_s: float,
        step_s: float,
        **kwargs,
    ) -> PropagatedTrajectory:
        coe = coe_for_preset(preset)
        t_eval = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
        return self.propagate_coe_op(coe, t_eval, **kwargs)
