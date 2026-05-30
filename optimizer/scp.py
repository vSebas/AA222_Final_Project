from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np
import cvxpy as cp

MODELING_DIR = Path(__file__).resolve().parents[1] / "modeling"
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))

from rover_dynamic_model import RoverModel

ArrayLike = float | np.ndarray

@dataclass
class SCP:
    # Augmented state: [X, Y, E, speed, heading, omega]
    n_states: int = 6
    # Control: [speed, heading]
    n_inputs: int = 2
    dt: float = 0.1
    final_time_s: float = 1.0
    max_iterations: int = 25

    nu_k: ArrayLike = 0.0
    A_k: ArrayLike = 0.0
    B_k: ArrayLike = 0.0

    x_bar: ArrayLike = 0.0
    u_bar: ArrayLike = 0.0

    model: RoverModel = field(
        default_factory=lambda: RoverModel(
            battery_charge_j=20_000.0,
            power_generation_w=65.0,
            phi=np.deg2rad(0.0),
            xi=np.deg2rad(0.0),
        )
    )

    ct: float = 0.0
    ce: float = 1.0
    c_nu: float = 1.0e4

    rho_x: float = 10.0
    rho_u: float = 1.0

    high_level_path: list = field(default_factory=list)
    history: list = field(default_factory=list)

    beta_grow: float = 1.1
    beta_shrink: float = 0.5
    eps_dyn: float = 0.5
    eps_nu: float = 1.0e-4
    eps_x: float = 1.0e-3
    eps_J: float = 1.0e-3

    @property
    def v_max(self):
        return self.model.max_speed_mps

    @property
    def E_min(self):
        return self.model.min_battery_charge_j

    @property
    def E_max(self):
        return self.model.max_battery_charge_j

    @property
    def P_cons_max(self):
        return self.model.max_power_consumption_w

    @property
    def horizon_steps(self):
        return int(round(self.final_time_s / self.dt))

    def __post_init__(self):
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.final_time_s <= 0.0:
            raise ValueError("final_time_s must be positive")
        if not np.isclose(self.horizon_steps * self.dt, self.final_time_s):
            raise ValueError("final_time_s must be an integer multiple of dt")
    
    def linearize_dynamics(self, x_bar, u_bar, eps=1.0e-5):
        """First-order Taylor expansion of x+ = F(x, u, dt).

        Returns A, B, c such that

            F(x, u) ~= A @ x + B @ u + c

        around the nominal knot (x_bar, u_bar).
        """

        x_bar = np.asarray(x_bar, dtype=float)
        u_bar = np.asarray(u_bar, dtype=float)
        f_bar = self.model.F(x_bar, u_bar, self.dt)

        A_k = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            dx = np.zeros(self.n_states)
            dx[i] = eps
            A_k[:, i] = (
                self.model.F(x_bar + dx, u_bar, self.dt)
                - self.model.F(x_bar - dx, u_bar, self.dt)
            ) / (2.0 * eps)

        B_k = np.zeros((self.n_states, self.n_inputs))
        for i in range(self.n_inputs):
            du = np.zeros(self.n_inputs)
            du[i] = eps
            B_k[:, i] = (
                self.model.F(x_bar, u_bar + du, self.dt)
                - self.model.F(x_bar, u_bar - du, self.dt)
            ) / (2.0 * eps)

        c_k = f_bar - A_k @ x_bar - B_k @ u_bar
        return A_k, B_k, c_k

    def linearize_power(self, x_bar, u_bar, eps=1.0e-5):
        """First-order Taylor expansion of P(x, u, dt)."""

        x_bar = np.asarray(x_bar, dtype=float)
        u_bar = np.asarray(u_bar, dtype=float)
        p_bar = self.model.P(x_bar, u_bar, self.dt)

        p_x = np.zeros(self.n_states)
        for i in range(self.n_states):
            dx = np.zeros(self.n_states)
            dx[i] = eps
            p_x[i] = (
                self.model.P(x_bar + dx, u_bar, self.dt)
                - self.model.P(x_bar - dx, u_bar, self.dt)
            ) / (2.0 * eps)

        p_u = np.zeros(self.n_inputs)
        for i in range(self.n_inputs):
            du = np.zeros(self.n_inputs)
            du[i] = eps
            p_u[i] = (
                self.model.P(x_bar, u_bar + du, self.dt)
                - self.model.P(x_bar, u_bar - du, self.dt)
            ) / (2.0 * eps)

        p_c = p_bar - p_x @ x_bar - p_u @ u_bar
        return p_x, p_u, p_c

    def nominal_state_from_waypoint(self, waypoint):
        """Build an internal nominal state from a waypoint.

        A 2D waypoint only specifies position. The remaining entries are just
        an initial guess for linearization, not user-specified constraints.
        """

        waypoint = np.asarray(waypoint, dtype=float)
        if waypoint.size == self.n_states:
            return waypoint
        if waypoint.size == 3:
            return np.array([waypoint[0], waypoint[1], waypoint[2], 0.0, 0.0, 0.0])
        if waypoint.size == 2:
            return np.array(
                [waypoint[0], waypoint[1], self.model.battery_charge_j, 0.0, 0.0, 0.0]
            )
        raise ValueError("waypoint must be [X, Y], [X, Y, E], or [X, Y, E, speed, heading, omega]")

    def add_waypoint_constraint(self, constraints, variable, waypoint):
        """Constrain only the waypoint components the user provided."""

        waypoint = np.asarray(waypoint, dtype=float)
        if waypoint.size == self.n_states:
            constraints += [variable == waypoint]
        elif waypoint.size == 3:
            constraints += [variable[:3] == waypoint]
        elif waypoint.size == 2:
            constraints += [variable[:2] == waypoint]
        else:
            raise ValueError(
                "waypoint must be [X, Y], [X, Y, E], or [X, Y, E, speed, heading, omega]"
            )
        return constraints

    def initialize_nominal_trajectory(self):
        """Build an initial state/control guess from the high-level path."""

        if len(self.high_level_path) < 2:
            raise ValueError("high_level_path must contain at least a start and goal waypoint")

        waypoints = [self.nominal_state_from_waypoint(waypoint) for waypoint in self.high_level_path]
        N = self.horizon_steps
        x_bar = np.zeros((self.n_states, N + 1))
        u_bar = np.zeros((self.n_inputs, N))
        x_bar[:, 0] = waypoints[0]

        path_positions = np.array([waypoint[:2] for waypoint in waypoints])
        segment_lengths = np.linalg.norm(np.diff(path_positions, axis=0), axis=1)
        path_length = float(np.sum(segment_lengths))

        for k in range(N):
            current = x_bar[:, k]
            if path_length <= 1.0e-9:
                target_position = path_positions[-1]
            else:
                distance_along_path = path_length * (k + 1) / N
                segment_start_distance = 0.0
                target_position = path_positions[-1]
                for segment_index, segment_length in enumerate(segment_lengths):
                    segment_end_distance = segment_start_distance + segment_length
                    if distance_along_path <= segment_end_distance or segment_index == len(segment_lengths) - 1:
                        if segment_length <= 1.0e-9:
                            target_position = path_positions[segment_index + 1]
                        else:
                            alpha = (distance_along_path - segment_start_distance) / segment_length
                            target_position = (
                                (1.0 - alpha) * path_positions[segment_index]
                                + alpha * path_positions[segment_index + 1]
                            )
                        break
                    segment_start_distance = segment_end_distance

            delta_xy = target_position - current[:2]
            distance = np.linalg.norm(delta_xy)
            heading = current[4] if distance <= 1.0e-9 else np.arctan2(delta_xy[1], delta_xy[0])
            speed = min(self.v_max, distance / self.dt)

            u_bar[:, k] = [speed, heading]
            x_next = self.model.F(current, u_bar[:, k], self.dt)

            # Keep the path geometry from the supplied route, but use the
            # model-consistent auxiliary states and battery estimate.
            x_next[:2] = target_position
            x_bar[:, k + 1] = x_next

        self.x_bar = x_bar
        self.u_bar = u_bar
        self.nu_k = np.zeros((self.n_states, N))
        return x_bar, u_bar

    def nonlinear_rollout(self, x0, u):
        """Roll out the true discrete dynamics under a control sequence."""

        u = np.asarray(u, dtype=float)
        n_segments = u.shape[1]
        x = np.zeros((self.n_states, n_segments + 1))
        x[:, 0] = np.asarray(x0, dtype=float)
        for k in range(n_segments):
            x[:, k + 1] = self.model.F(x[:, k], u[:, k], self.dt)
        return x

    def nonlinear_dynamics_defect(self, x, u):
        """Maximum one-step mismatch against the true nonlinear dynamics."""

        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        defects = []
        for k in range(u.shape[1]):
            defects.append(x[:, k + 1] - self.model.F(x[:, k], u[:, k], self.dt))
        return 0.0 if not defects else float(np.max(np.linalg.norm(defects, ord=np.inf, axis=1)))

    def nonlinear_objective(self, x, u):
        """Evaluate the nonlinear energy/time objective on a trajectory."""

        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        total_power_cost = 0.0
        for k in range(u.shape[1]):
            total_power_cost += self.model.P(x[:, k], u[:, k], self.dt) * self.dt
        return float(self.ce * total_power_cost + self.ct * self.final_time_s)

    def convex_ocp(self):
        if cp is None:
            raise ImportError("cvxpy is required to solve the convex SCP subproblem")
        if len(self.high_level_path) < 2:
            raise ValueError("high_level_path must contain at least a start and goal waypoint")

        x_bar = np.asarray(self.x_bar, dtype=float)
        u_bar = np.asarray(self.u_bar, dtype=float)
        N = self.horizon_steps
        if x_bar.shape != (self.n_states, N + 1) or u_bar.shape != (self.n_inputs, N):
            x_bar, u_bar = self.initialize_nominal_trajectory()

        x = cp.Variable((self.n_states, N + 1))  # X, Y, E, speed, heading, omega
        u = cp.Variable((self.n_inputs, N))      # speed, heading
        nu = cp.Variable((self.n_states, N))     # virtual dynamics control

        # Initial state and cost weights

        start = np.asarray(self.high_level_path[0], dtype=float)
        goal = np.asarray(self.high_level_path[-1], dtype=float)
        constraints = []
        constraints = self.add_waypoint_constraint(constraints, x[:, 0], start)
        constraints = self.add_waypoint_constraint(constraints, x[:, N], goal)
        cost = 0

        # Define cost function and dynamics over the horizon
        for t in range(N):

            # No se afiniza/lineariza en cada timestep de una iteracion del SCP
            # es una vez por iteracion
            A_k, B_k, c_k = self.linearize_dynamics(x_bar[:, t], u_bar[:, t])
            p_x_k, p_u_k, p_c_k = self.linearize_power(x_bar[:, t], u_bar[:, t])
            P_cons_k = p_x_k @ x[:, t] + p_u_k @ u[:, t] + p_c_k
            cost += self.ce * P_cons_k * self.dt
            cost += self.c_nu * cp.norm1(nu[:, t])

            constraints += [
                            x[:, t + 1] == A_k @ x[:, t] + B_k @ u[:, t] + c_k + nu[:, t],
                            # P_k \in C_k     # corridor, from high-level path
                            u[0, t] >= 0,
                            u[0, t] <= self.v_max,
                            # accel constraints
                            x[2, t] >= self.E_min,
                            x[2, t] <= self.E_max,
                            x[2, t + 1] >= self.E_min,
                            x[2, t + 1] <= self.E_max,
                            P_cons_k <= self.P_cons_max,
                            cp.norm_inf(x[:, t] - x_bar[:, t]) <= self.rho_x,
                            cp.norm_inf(u[:, t] - u_bar[:, t]) <= self.rho_u,
                            ]
            
        cost += self.ct * self.final_time_s

        objective = cp.Minimize(cost)
        prob = cp.Problem(objective, constraints)
        prob.solve()

        x_star = x.value
        u_star = u.value
        nu_star = nu.value

        if x_star is None or u_star is None or nu_star is None:
            raise RuntimeError(f"convex SCP subproblem failed with status {prob.status}")

        return x_star, u_star, nu_star

    def scp_algorithm(self):
        """Run the SCP trust-region loop."""

        if cp is None:
            raise ImportError("cvxpy is required to run SCP")

        x_bar = np.asarray(self.x_bar, dtype=float)
        u_bar = np.asarray(self.u_bar, dtype=float)
        N = self.horizon_steps
        if x_bar.shape != (self.n_states, N + 1) or u_bar.shape != (self.n_inputs, N):
            x_bar, u_bar = self.initialize_nominal_trajectory()

        J_bar = self.nonlinear_objective(x_bar, u_bar)
        self.history = []

        for iteration in range(self.max_iterations):
            try:
                x_candidate, u_candidate, nu_candidate = self.convex_ocp()
            except RuntimeError as exc:
                self.history.append(
                    {
                        "iteration": iteration,
                        "accepted": False,
                        "error": str(exc),
                        "rho_x": self.rho_x,
                        "rho_u": self.rho_u,
                    }
                )
                self.rho_x *= self.beta_shrink
                self.rho_u *= self.beta_shrink
                continue

            defect_dyn = self.nonlinear_dynamics_defect(x_candidate, u_candidate)
            virtual_norm = float(np.max(np.abs(nu_candidate)))
            state_change = float(np.max(np.abs(x_candidate - x_bar)))
            control_change = float(np.max(np.abs(u_candidate - u_bar)))
            J_candidate = self.nonlinear_objective(x_candidate, u_candidate)
            cost_change = abs(J_bar - J_candidate)

            accepted = defect_dyn <= self.eps_dyn and J_candidate <= J_bar + self.eps_J
            self.history.append(
                {
                    "iteration": iteration,
                    "accepted": accepted,
                    "J_nl": J_candidate,
                    "defect_dyn": defect_dyn,
                    "virtual_norm": virtual_norm,
                    "state_change": state_change,
                    "control_change": control_change,
                    "rho_x": self.rho_x,
                    "rho_u": self.rho_u,
                }
            )

            if accepted:
                self.x_bar = x_candidate
                self.u_bar = u_candidate
                self.nu_k = nu_candidate
                x_bar = x_candidate
                u_bar = u_candidate

                converged = (
                    defect_dyn <= self.eps_dyn
                    and virtual_norm <= self.eps_nu
                    and state_change <= self.eps_x
                    and cost_change <= self.eps_J
                )
                J_bar = J_candidate
                self.rho_x *= self.beta_grow
                self.rho_u *= self.beta_grow

                if converged:
                    break
            else:
                self.rho_x *= self.beta_shrink
                self.rho_u *= self.beta_shrink

        return self.x_bar, self.u_bar
