from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = float | np.ndarray


@dataclass(frozen=True)
class RoverModel:
    """Reduced EMRS/Hu rover dynamic and power model.

    The model uses the projected SE(2) state ``[x_G, y_G, psi_G]`` and the
    projected control ``[v_Bpi, omega_Bpi]``. For the simplified AA222 model,
    terrain resistance is a C0-only constant-force model:

        P_res = C0 |v_B|

    The default C0 is the flat, no-grade EMRS breadboard-scaled lunar value
    from ``modeling/rolling_resistance/emrs_resistive_forces.json``:

        C0 = R_compression + R_rolling + R_bulldozing = 63.5441666 N

    Grade/slope is intentionally kept out of C0 and handled separately by the
    translational term ``m g sin(phi)`` to avoid double counting.
    """

    mass: float = 84.0
    inertia_z: float = 7.111679166666666
    gravity: float = 1.62
    c0: float = 63.54416662174218
    p_base: float = 100.0
    p_rtg: float = 0.0
    p_solar: float = 0.0
    phi: ArrayLike = 0.0
    xi: ArrayLike = 0.0

    @property
    def p_available(self) -> float:
        """Hybrid RTG plus solar power availability from Eq. (6)."""

        return self.p_rtg + self.p_solar

    @staticmethod
    def _as_array(value: ArrayLike) -> np.ndarray:
        return np.asarray(value, dtype=float)

    @staticmethod
    def _safe_cos(angle: ArrayLike, name: str, min_abs: float = 1.0e-6) -> np.ndarray:
        cos_value = np.cos(RoverModel._as_array(angle))
        if np.any(np.abs(cos_value) < min_abs):
            raise ValueError(f"cos({name}) is too close to zero for a stable projection")
        return cos_value

    @staticmethod
    def _positive_part(value: ArrayLike) -> np.ndarray:
        return np.maximum(RoverModel._as_array(value), 0.0)

    def body_longitudinal_velocity(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        phi: ArrayLike | None = None,
    ) -> np.ndarray:
        """Body-frame longitudinal velocity from Eq. (1)."""

        phi = self.phi if phi is None else phi
        v_global = np.hypot(self._as_array(x_dot), self._as_array(y_dot))
        return v_global / self._safe_cos(phi, "phi")

    def body_longitudinal_acceleration(
        self,
        x_ddot: ArrayLike,
        y_ddot: ArrayLike,
        psi: ArrayLike,
        phi: ArrayLike | None = None,
    ) -> np.ndarray:
        """Body-frame longitudinal acceleration from Eq. (2)."""

        phi = self.phi if phi is None else phi
        x_ddot = self._as_array(x_ddot)
        y_ddot = self._as_array(y_ddot)
        psi = self._as_array(psi)
        projected_accel = x_ddot * np.cos(psi) + y_ddot * np.sin(psi)
        return projected_accel / self._safe_cos(phi, "phi")

    def body_yaw_rate(self, omega: ArrayLike, xi: ArrayLike | None = None) -> np.ndarray:
        """Body-frame yaw rate from Eq. (3)."""

        xi = self.xi if xi is None else xi
        return self._as_array(omega) / self._safe_cos(xi, "xi")

    def body_yaw_acceleration(self, omega_dot: ArrayLike, xi: ArrayLike | None = None) -> np.ndarray:
        """Body-frame yaw acceleration from Eq. (4)."""

        xi = self.xi if xi is None else xi
        return self._as_array(omega_dot) / self._safe_cos(xi, "xi")

    def projected_reference_control(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        omega: ArrayLike,
    ) -> np.ndarray:
        """Build the projected NMPC reference input ``[v_ref, omega_ref]``."""

        return np.stack((np.hypot(self._as_array(x_dot), self._as_array(y_dot)), self._as_array(omega)), axis=-1)

    def body_command_from_projected(
        self,
        v_projected: ArrayLike,
        omega_projected: ArrayLike,
        phi: ArrayLike | None = None,
        xi: ArrayLike | None = None,
    ) -> np.ndarray:
        """Map projected-frame control to body-frame command."""

        phi = self.phi if phi is None else phi
        xi = self.xi if xi is None else xi
        return np.stack(
            (
                self._as_array(v_projected) / self._safe_cos(phi, "phi"),
                self._as_array(omega_projected) / self._safe_cos(xi, "xi"),
            ),
            axis=-1,
        )

    def se2_kinematics(self, state: ArrayLike, control: ArrayLike) -> np.ndarray:
        """Continuous projected differential-drive dynamics.

        ``state`` has final dimension ``[x, y, psi]`` and ``control`` has
        final dimension ``[v_Bpi, omega_Bpi]``.
        """

        state = self._as_array(state)
        control = self._as_array(control)
        psi = state[..., 2]
        v = control[..., 0]
        omega = control[..., 1]
        return np.stack((v * np.cos(psi), v * np.sin(psi), omega), axis=-1)

    def euler_step(self, state: ArrayLike, control: ArrayLike, dt: float) -> np.ndarray:
        """Forward-Euler step for the projected SE(2) kinematic model."""

        return self._as_array(state) + dt * self.se2_kinematics(state, control)

    def linear_motion_power(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        x_ddot: ArrayLike,
        y_ddot: ArrayLike,
        psi: ArrayLike,
        phi: ArrayLike | None = None,
    ) -> np.ndarray:
        """Nonnegative translational mechanical power.

        This implements ``[(m a_B + m g sin(phi)) v_B]_+``. Downhill motion or
        braking is not modeled as regenerative energy.
        """

        phi = self.phi if phi is None else phi
        v_body = self.body_longitudinal_velocity(x_dot, y_dot, phi)
        a_body = self.body_longitudinal_acceleration(x_ddot, y_ddot, psi, phi)
        raw_power = self.mass * (a_body + self.gravity * np.sin(self._as_array(phi))) * v_body
        return self._positive_part(raw_power)

    def rotational_motion_power(
        self,
        omega: ArrayLike,
        omega_dot: ArrayLike,
        xi: ArrayLike | None = None,
    ) -> np.ndarray:
        """Nonnegative yaw mechanical power ``[I_z alpha omega]_+``."""

        xi = self.xi if xi is None else xi
        omega_body = self.body_yaw_rate(omega, xi)
        omega_dot_body = self.body_yaw_acceleration(omega_dot, xi)
        return self._positive_part(self.inertia_z * omega_dot_body * omega_body)

    def resistive_power(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        phi: ArrayLike | None = None,
        *,
        signed_velocity: bool = False,
    ) -> np.ndarray:
        """C0-only terrain resistance power.

        ``C0`` represents compression, rolling/internal, and bulldozing
        resistance on flat terrain. It excludes grade resistance.

        The default is dissipative for reverse motion too. Set
        ``signed_velocity=True`` only if the caller explicitly wants signed
        mechanical power.
        """

        phi = self.phi if phi is None else phi
        v_body = self.body_longitudinal_velocity(x_dot, y_dot, phi)
        multiplier = v_body if signed_velocity else np.abs(v_body)
        return self.c0 * multiplier

    def power_breakdown(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        x_ddot: ArrayLike,
        y_ddot: ArrayLike,
        psi: ArrayLike,
        omega: ArrayLike,
        omega_dot: ArrayLike,
        phi: ArrayLike | None = None,
        xi: ArrayLike | None = None,
        p_available: ArrayLike | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute motion, baseline, total, and available power."""

        p_linear = self.linear_motion_power(x_dot, y_dot, x_ddot, y_ddot, psi, phi)
        p_rotational = self.rotational_motion_power(omega, omega_dot, xi)
        p_resistive = self.resistive_power(x_dot, y_dot, phi)
        p_motion = p_linear + p_rotational + p_resistive
        p_total = p_motion + self.p_base
        available = self.p_available if p_available is None else p_available
        p_available_array = np.broadcast_to(self._as_array(available), np.shape(p_total))
        p_base = np.broadcast_to(self._as_array(self.p_base), np.shape(p_total))
        return {
            "p_linear": p_linear,
            "p_rotational": p_rotational,
            "p_resistive": p_resistive,
            "p_motion": p_motion,
            "p_base": p_base,
            "p_total": p_total,
            "p_available": p_available_array,
            "margin": p_available_array - p_total,
            "smooth_motion_norm": np.hypot(p_linear, p_rotational),
        }

    def cumulative_energy(self, power: ArrayLike, time: ArrayLike) -> float:
        """Integrate power over time with the trapezoidal rule."""

        trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(trapezoid(self._as_array(power), self._as_array(time)))

    # The functions below are the smooth power-limit penalty from the paper.
    # They are kept here as commented reference code because the current
    # battery-runout example only needs raw power and energy accounting.
    # Re-enable them when using RoverModel inside a trajectory optimizer.

    # def softplus(self, z: ArrayLike, kappa: float = 10.0) -> np.ndarray:
    #     """Numerically stable softplus used by Eq. (20)."""

    #     z_scaled = kappa * self._as_array(z)
    #     return np.logaddexp(0.0, z_scaled) / kappa

    # def sigmoid(self, z: ArrayLike) -> np.ndarray:
    #     """Numerically stable logistic sigmoid."""

    #     z = self._as_array(z)
    #     return np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

    # def power_limit_penalty(
    #     self,
    #     breakdown: dict[str, np.ndarray],
    #     weight: float = 1.0,
    #     kappa: float = 10.0,
    # ) -> np.ndarray:
    #     """Smooth instantaneous power-limit penalty from Eqs. (18)-(21)."""

    #     tau = breakdown["p_available"] - breakdown["p_total"]
    #     z = breakdown["smooth_motion_norm"] - tau
    #     sp = self.softplus(z, kappa)
    #     return weight * sp**2

    # def power_limit_penalty_gradient_scale(
    #     self,
    #     breakdown: dict[str, np.ndarray],
    #     weight: float = 1.0,
    #     kappa: float = 10.0,
    #     eps: float = 1.0e-12,
    # ) -> tuple[np.ndarray, np.ndarray]:
    #     """Return ``dJ/dP_lin`` and ``dJ/dP_rot`` from Eqs. (22)-(24)."""

    #     tau = breakdown["p_available"] - breakdown["p_total"]
    #     z = breakdown["smooth_motion_norm"] - tau
    #     d_j_d_z = 2.0 * weight * self.softplus(z, kappa) * self.sigmoid(kappa * z)
    #     norm = np.maximum(breakdown["smooth_motion_norm"], eps)
    #     return d_j_d_z * breakdown["p_linear"] / norm, d_j_d_z * breakdown["p_rotational"] / norm

    # This feasibility check belongs with the penalty helpers above. It is not
    # used by the open-loop battery example, which allows temporary overloads
    # and drains the battery by the excess power instead.

    # def is_power_feasible(
    #     self,
    #     breakdown: dict[str, np.ndarray],
    #     tolerance: float = 0.0,
    # ) -> bool:
    #     """Check the instantaneous constraint ``P_cons <= P_avail``."""

    #     return bool(np.all(breakdown["p_total"] <= breakdown["p_available"] + tolerance))
