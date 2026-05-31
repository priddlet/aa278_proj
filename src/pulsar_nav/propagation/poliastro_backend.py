"""Optional poliastro Cowell propagator with third-body perturbations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsar_nav.propagation.dynamics import (
    GM_EARTH,
    GM_MOON,
    GM_SUN,
    SRP_SCALE,
    DynamicsConfig,
    moon_j2_acceleration,
)
from pulsar_nav.propagation.propagator import PropagatedTrajectory
from pulsar_nav.spice import ephemeris as spice_ephem


@dataclass
class PoliastroLunarPropagator:
    """
    Propagate using poliastro Cowell with custom acceleration function.

    Requires poliastro and astropy to be installed.
    """

    et0: float
    config: DynamicsConfig

    def __post_init__(self) -> None:
        try:
            from astropy import units as u
            from astropy.time import Time
            from poliastro.bodies import Moon
            from poliastro.core.propagation import func_twobody
            from poliastro.core.perturbations import J2_perturbation, third_body
            from poliastro.twobody import Orbit
            from poliastro.twobody.propagation import CowellPropagator
            from poliastro.twobody.sampling import EpochsArray
        except ImportError as exc:
            raise ImportError(
                "poliastro backend requires poliastro and astropy. "
                "Install with: pip install -e ./poliastro"
            ) from exc

        self._u = u
        self._Time = Time
        self._Moon = Moon
        self._func_twobody = func_twobody
        self._J2_perturbation = J2_perturbation
        self._third_body = third_body
        self._Orbit = Orbit
        self._CowellPropagator = CowellPropagator
        self._EpochsArray = EpochsArray

    def propagate(
        self,
        state0_mci_km: np.ndarray,
        t_eval_s: np.ndarray,
        *,
        rtol: float = 1e-9,
    ) -> PropagatedTrajectory:
        u = self._u
        r0 = state0_mci_km[:3] << u.km
        v0 = state0_mci_km[3:6] << (u.km / u.s)
        epoch = self._Time(self.et0, format="sec", scale="tdb")
        orbit = self._Orbit.from_vectors(self._Moon, (r0, v0), epoch=epoch)

        tofs = (np.asarray(t_eval_s, float) << u.s) - (t_eval_s[0] << u.s)
        cfg = self.config
        k = GM_MOON

        def earth_at(t0, _state, _k):
            et = self.et0 + t0
            return spice_ephem.body_position_mci("EARTH", et)

        def sun_at(t0, _state, _k):
            et = self.et0 + t0
            return spice_ephem.body_position_mci("SUN", et)

        def accel(t0, state, k_val):
            du = self._func_twobody(t0, state, k_val)
            if cfg.include_earth:
                ax, ay, az = self._third_body(
                    t0, state, k_val, GM_EARTH, earth_at
                )
                du = du + np.array([0, 0, 0, ax, ay, az])
            if cfg.include_sun:
                ax, ay, az = self._third_body(
                    t0, state, k_val, GM_SUN, sun_at
                )
                du = du + np.array([0, 0, 0, ax, ay, az])
                if cfg.include_srp and cfg.gamma_srp != 0.0:
                    et = self.et0 + t0
                    r_sun = spice_ephem.body_position_mci("SUN", et)
                    rho = state[:3] - r_sun
                    a_srp = -SRP_SCALE * cfg.gamma_srp * rho / np.linalg.norm(rho) ** 3
                    du[3:6] += a_srp
            if cfg.include_moon_j2:
                du[3:6] += moon_j2_acceleration(state[:3])
            return du

        ephem = orbit.to_ephem(
            self._EpochsArray(
                orbit.epoch + tofs,
                method=self._CowellPropagator(rtol=rtol, f=accel),
            )
        )
        coords = ephem.sample()
        pos_mci = coords.xyz.to_value(u.km)
        vel_mci = coords.differentials["s"].d_xyz.to_value(u.km / u.s)

        et = self.et0 + t_eval_s
        pos_icrs = np.zeros((len(t_eval_s), 3))
        vel_icrs = np.zeros((len(t_eval_s), 3))
        for i, et_i in enumerate(et):
            pos_icrs[i] = spice_ephem.mci_to_icrs_position(pos_mci[i], et_i) * 1000.0
            if i == 0 and len(et) > 1:
                dt = et[1] - et[0]
            elif i == len(et) - 1:
                dt = et[-1] - et[-2]
            else:
                dt = et[i + 1] - et[i - 1]
            moon_vel = (
                spice_ephem.moon_position_icrs_km(et_i + dt)
                - spice_ephem.moon_position_icrs_km(et_i - dt)
            ) / (2.0 * dt)
            vel_icrs[i] = (moon_vel + vel_mci[i]) * 1000.0

        return PropagatedTrajectory(
            et0=self.et0,
            t_rel_s=np.asarray(t_eval_s, float),
            et=et,
            position_mci_km=pos_mci,
            velocity_mci_km_s=vel_mci,
            position_icrs_m=pos_icrs,
            velocity_icrs_m_s=vel_icrs,
        )
