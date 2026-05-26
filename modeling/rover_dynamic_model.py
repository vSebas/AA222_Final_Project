from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = float | np.ndarray


@dataclass
class RoverModel:
    """Reduced EMRS/Hu rover dynamic and power model.

    The model owns the current rover pose, command, battery charge, and power
    metrics. Calling ``step(dt)`` advances the committed state with RK4.

    Terrain resistance is simplified to the C0-only form

        P_res = C0 |v_B|

    where the default C0 is the flat, no-grade EMRS breadboard-scaled lunar
    terramechanics estimate:

        C0 = R_compression + R_rolling + R_bulldozing = 63.5441666 N

    Grade is not included in C0; slope enters through ``m g sin(phi)``.
    """

    mass: float = 84.0
    inertia_z: float = 7.111679166666666
    gravity: float = 1.62
    c0: float = 63.54416662174218
    p_base: float = 100.0
    power_generation_w: float = 65.0
    battery_charge_j: float = 20_000.0
    phi: ArrayLike = 0.0
    xi: ArrayLike = 0.0
    x_g: float = 0.0
    y_g: float = 0.0
    psi_g: float = 0.0
    v_command_mps: float = 0.45
    omega_command_radps: float = 0.18
    time_s: float = 0.0
    power_consumption_w: float = 0.0
    battery_discharge_rate_j_per_s: float = 0.0

    def __post_init__(self) -> None:
        self.update_power_metrics()

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

    @property
    def pose(self) -> np.ndarray:
        """Current projected SE(2) pose ``[x_G, y_G, psi_G]``."""

        return np.array([self.x_g, self.y_g, self.psi_g], dtype=float)

    @property
    def control(self) -> np.ndarray:
        """Current projected command ``[v_Bpi, omega_Bpi]``."""

        return np.array([self.v_command_mps, self.omega_command_radps], dtype=float)

    @property
    def _state(self) -> np.ndarray:
        """Current augmented integration state ``[x_G, y_G, psi_G, E]``."""

        return np.array([self.x_g, self.y_g, self.psi_g, self.battery_charge_j], dtype=float)

    def body_longitudinal_velocity(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        phi: ArrayLike | None = None,
    ) -> np.ndarray:
        """Body-frame longitudinal velocity from projected global rates."""

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
        """Body-frame longitudinal acceleration from projected global rates."""

        phi = self.phi if phi is None else phi
        x_ddot = self._as_array(x_ddot)
        y_ddot = self._as_array(y_ddot)
        psi = self._as_array(psi)
        projected_accel = x_ddot * np.cos(psi) + y_ddot * np.sin(psi)
        return projected_accel / self._safe_cos(phi, "phi")

    def body_yaw_rate(self, omega: ArrayLike, xi: ArrayLike | None = None) -> np.ndarray:
        """Body-frame yaw rate from projected yaw rate."""

        xi = self.xi if xi is None else xi
        return self._as_array(omega) / self._safe_cos(xi, "xi")

    def body_yaw_acceleration(self, omega_dot: ArrayLike, xi: ArrayLike | None = None) -> np.ndarray:
        """Body-frame yaw acceleration from projected yaw acceleration."""

        xi = self.xi if xi is None else xi
        return self._as_array(omega_dot) / self._safe_cos(xi, "xi")

    def se2_kinematics(self, state: ArrayLike, control: ArrayLike) -> np.ndarray:
        """Projected SE(2) rates ``[x_dot_G, y_dot_G, psi_dot_G]``."""

        state = self._as_array(state)
        control = self._as_array(control)
        psi = state[..., 2]
        v = control[..., 0]
        omega = control[..., 1]
        return np.stack((v * np.cos(psi), v * np.sin(psi), omega), axis=-1)

    def linear_motion_power(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        x_ddot: ArrayLike,
        y_ddot: ArrayLike,
        psi: ArrayLike,
        phi: ArrayLike | None = None,
    ) -> np.ndarray:
        """Nonnegative translational mechanical power."""

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
        """Nonnegative yaw mechanical power."""

        xi = self.xi if xi is None else xi
        omega_body = self.body_yaw_rate(omega, xi)
        omega_dot_body = self.body_yaw_acceleration(omega_dot, xi)
        return self._positive_part(self.inertia_z * omega_dot_body * omega_body)

    def resistive_power(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        phi: ArrayLike | None = None,
    ) -> np.ndarray:
        """C0-only terrain resistance power."""

        phi = self.phi if phi is None else phi
        v_body = self.body_longitudinal_velocity(x_dot, y_dot, phi)
        return self.c0 * np.abs(v_body)

    def power_breakdown(
        self,
        x_dot: ArrayLike,
        y_dot: ArrayLike,
        x_ddot: ArrayLike,
        y_ddot: ArrayLike,
        psi: ArrayLike,
        omega: ArrayLike,
        omega_dot: ArrayLike,
    ) -> dict[str, np.ndarray]:
        """Compute motion, baseline, consumption, and generation power."""

        p_linear = self.linear_motion_power(x_dot, y_dot, x_ddot, y_ddot, psi)
        p_rotational = self.rotational_motion_power(omega, omega_dot)
        p_resistive = self.resistive_power(x_dot, y_dot)
        p_motion = p_linear + p_rotational + p_resistive
        power_consumption_w = p_motion + self.p_base
        power_generation_w = np.broadcast_to(
            self._as_array(self.power_generation_w),
            np.shape(power_consumption_w),
        )
        return {
            "p_linear": p_linear,
            "p_rotational": p_rotational,
            "p_resistive": p_resistive,
            "p_motion": p_motion,
            "p_base": np.broadcast_to(self._as_array(self.p_base), np.shape(power_consumption_w)),
            "power_consumption_w": power_consumption_w,
            "power_generation_w": power_generation_w,
            "power_margin_w": power_generation_w - power_consumption_w,
        }

    def _power_at_state(self, state: np.ndarray) -> float:
        """Instantaneous consumption for an augmented state and current command."""

        pose = state[:3]
        v = self.v_command_mps
        omega = self.omega_command_radps
        global_rates = self.se2_kinematics(pose, self.control)

        # Constant commanded v and omega imply centripetal projected acceleration.
        x_ddot = -v * omega * np.sin(pose[2])
        y_ddot = v * omega * np.cos(pose[2])

        breakdown = self.power_breakdown(
            x_dot=global_rates[0],
            y_dot=global_rates[1],
            x_ddot=x_ddot,
            y_ddot=y_ddot,
            psi=pose[2],
            omega=omega,
            omega_dot=0.0,
        )
        return float(breakdown["power_consumption_w"])

    def _dynamics(self, state: np.ndarray) -> np.ndarray:
        """Augmented dynamics for RK4 integration."""

        pose_dot = self.se2_kinematics(state[:3], self.control)
        power_consumption_w = self._power_at_state(state)
        battery_discharge_rate_j_per_s = max(
            power_consumption_w - self.power_generation_w,
            0.0,
        )
        return np.array([pose_dot[0], pose_dot[1], pose_dot[2], -battery_discharge_rate_j_per_s])

    def _rk4(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Fourth-order Runge-Kutta integration for the model's augmented state."""

        k1 = self._dynamics(state)
        k2 = self._dynamics(state + 0.5 * dt * k1)
        k3 = self._dynamics(state + 0.5 * dt * k2)
        k4 = self._dynamics(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def update_power_metrics(self) -> None:
        """Refresh current power consumption and battery discharge rate."""

        self.power_consumption_w = self._power_at_state(self._state)
        self.battery_discharge_rate_j_per_s = max(
            self.power_consumption_w - self.power_generation_w,
            0.0,
        )

    def step(self, dt: float) -> None:
        """Advance pose and battery by one committed RK4 timestep."""

        if self.battery_charge_j <= 0.0:
            self.battery_charge_j = 0.0
            self.update_power_metrics()
            return

        next_state = self._rk4(self._state, dt)
        self.x_g = float(next_state[0])
        self.y_g = float(next_state[1])
        self.psi_g = float(next_state[2])
        self.battery_charge_j = max(float(next_state[3]), 0.0)
        self.time_s += dt
        self.update_power_metrics()
