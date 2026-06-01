from dataclasses import dataclass, field
from pathlib import Path
import os
import sys

import cvxpy as cvx

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax

import numpy as np

MODELING_DIR = Path(__file__).resolve().parents[1] / "modeling"
if str(MODELING_DIR) not in sys.path:
    sys.path.insert(0, str(MODELING_DIR))
from rover_dynamic_model import RoverModel

@dataclass
class SCP:
    n_state: int = 6     # state dimension
    m_control: int = 2   # control dimension
    dt: float = 0.1
    eps: int = 1e-3      # SCP convergence tolerance
    N: int = 100         # MPC horizon
    N_scp: int = 25      # maximum number of SCP iterations
    final_time_s: float = 0.0
    # Preserved for a later variable-final-time version:
    # dt_min_factor: float = 0.5
    # dt_max_factor: float = 2.0

    model: RoverModel = field(init=False)

    # ct: float = 0.0
    ce: float = 1.0
    c_nu: float = 1.0e4

    rho_state: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 2000.0, 0.2, 0.3, 0.1]))
    rho_u: float = 1.0

    high_level_path: list = field(default_factory=list)
    history: list = field(default_factory=list)
    corridor_radius_m: float = 100.0
    a_max: float = 0.05
    omega_max: float = 0.20
    omega_dot_max: float = 0.05

    beta_grow: float = 1.1
    beta_shrink: float = 0.5
    eps_dyn: float = 0.5
    eps_nu: float = 1.0e-4
    eps_x: float = 1.0e-3
    eps_J: float = 1.0e-3
    defect_weight: float = 1.0e4
    virtual_weight: float = 1.0e4

    max_power_consumption_w: float = 200.0

    def __post_init__(self) -> None:
        self.model = RoverModel(
            dt=self.dt,
            battery_charge_j=20_000.0,
        )
        self.final_time_s = self.N * self.dt

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
        return self.max_power_consumption_w

    @property
    def horizon_steps(self):
        return self.N

    def affinize(self, f, x_bar, u_bar):
        """
        Affinize the function `f(s, u)` around `(x_bar, u_bar)`.
        First-order Taylor Expansion
        """
        x_bar = np.asarray(x_bar)
        u_bar = np.asarray(u_bar)
        def one_step(x, u):
            A, B = jax.jacobian(f, argnums=(0, 1))(x, u)
            c = f(x, u) - A @ x - B @ u
            return A, B, c

        A, B, c = jax.vmap(one_step)(x_bar, u_bar)
        return np.array(A), np.array(B), np.array(c)

    # Preserved for a later variable-final-time version:
    # def affinize_with_dt(self, f, x_bar, u_bar, dt_bar):
    #     """Linearize a stage function with respect to state, control, and dt."""
    #
    #     x_bar = np.asarray(x_bar)
    #     u_bar = np.asarray(u_bar)
    #
    #     def one_step(x, u):
    #         A, B, D = jax.jacobian(f, argnums=(0, 1, 2))(x, u, dt_bar)
    #         c = f(x, u, dt_bar) - A @ x - B @ u - D * dt_bar
    #         return A, B, D, c
    #
    #     A, B, D, c = jax.vmap(one_step)(x_bar, u_bar)
    #     return np.array(A), np.array(B), np.array(D), np.array(c)

    def path_positions(self, n_points: int) -> np.ndarray | None:
        """Interpolate the stored high-level path to ``n_points`` samples."""

        pts = np.asarray(self.high_level_path, dtype=float)
        if pts.size == 0:
            return None
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("high_level_path must contain XY waypoints")
        if len(pts) == 1:
            return np.repeat(pts[:1, :2], n_points, axis=0)

        # Compute segment length
        seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)

        # Build cum arch length per segment
        s = np.concatenate(([0.0], np.cumsum(seg)))
        if s[-1] <= 0.0:
            return np.repeat(pts[:1, :2], n_points, axis=0)
        s_eval = np.linspace(0.0, s[-1], n_points)
        x = np.interp(s_eval, s, pts[:, 0])
        y = np.interp(s_eval, s, pts[:, 1])
        return np.column_stack((x, y))

    def nonlinear_objective(self, x: np.ndarray, u: np.ndarray) -> float:
        energy = sum(self.model.P(x[k], u[k]) * self.dt for k in range(u.shape[0]))
        # return float(self.ce * energy + self.ct * self.final_time_s)
        return float(self.ce * energy)

    def nonlinear_dynamics_defect(self, x: np.ndarray, u: np.ndarray) -> float:
        defects = [x[k + 1] - self.model.F(x[k], u[k]) for k in range(u.shape[0])]
        if not defects:
            return 0.0
        return float(np.max(np.linalg.norm(defects, ord=np.inf, axis=1)))

    def merit(self, x: np.ndarray, u: np.ndarray, nu: np.ndarray | None = None) -> tuple[float, float, float, float]:
        objective = self.nonlinear_objective(x, u)
        defect = self.nonlinear_dynamics_defect(x, u)
        virtual = 0.0 if nu is None else float(np.max(np.abs(nu)))
        merit_value = objective + self.defect_weight * defect + self.virtual_weight * virtual
        return merit_value, objective, defect, virtual

    def scp_iteration(self, x0, goal_position, x_prev, u_prev, terminal_stop: bool = True):
        """Solve a single SCP sub-problem for trajectory optimization."""

        Af, Bf, cf = self.affinize(lambda x, u: self.model.jax_F(x, u), x_prev[:-1], u_prev)
        Ap, Bp, pc = self.affinize(lambda x, u: self.model.jax_P(x, u), x_prev[:-1], u_prev)
        path_xy = self.path_positions(self.N + 1)

        x_opt = cvx.Variable((self.N + 1, self.n_state))    # X, Y, E, speed, heading, omega
        u_opt = cvx.Variable((self.N, self.m_control))      # speed, heading
        nu_opt = cvx.Variable((self.N, self.n_state))       # virtual dynamics control

        # objective = self.ct * self.final_time_s
        objective = 0.0
        constraints = [
            x_opt[0, :] == x0,
            x_opt[self.N, :2] == np.asarray(goal_position, dtype=float),
        ]
        if terminal_stop:
            constraints += [
                x_opt[self.N, 3] == 0.0,
                x_opt[self.N, 5] == 0.0,
            ]
        # Preserved for a later variable-final-time version:
        # dt_opt = cvx.Variable(nonneg=True)
        # objective = self.ct * self.N * dt_opt
        # constraints += [
        #     dt_opt >= self.dt_min_factor * dt_prev,
        #     dt_opt <= self.dt_max_factor * dt_prev,
        # ]
        for t in range(self.N):
            Power_cons = Ap[t] @ x_opt[t,:] + Bp[t] @ u_opt[t,:] + pc[t]

            objective += self.ce * Power_cons * self.dt
            objective += self.c_nu * cvx.norm1(nu_opt[t,:])

            constraints += [
                            # Next state
                            x_opt[t + 1,:] == Af[t] @ x_opt[t,:] + Bf[t] @ u_opt[t,:] + cf[t] + nu_opt[t,:],
                            # Corridor
                            cvx.norm_inf(x_opt[t, :2] - path_xy[t]) <= self.corridor_radius_m,
                            cvx.norm_inf(x_opt[t + 1, :2] - path_xy[t + 1]) <= self.corridor_radius_m,
                            # Control constraints
                            u_opt[t,0] >= 0,
                            u_opt[t,0] <= self.v_max,
                            # State rates constraints
                            cvx.abs((u_opt[t, 0] - x_opt[t, 3]) / self.dt) <= self.a_max,
                            cvx.abs((u_opt[t, 1] - x_opt[t, 4]) / self.dt) <= self.omega_max,
                            cvx.abs(((u_opt[t, 1] - x_opt[t, 4]) / self.dt - x_opt[t, 5]) / self.dt) <= self.omega_dot_max,
                            # Power constraints
                            x_opt[t,2] >= self.E_min,
                            x_opt[t,2] <= self.E_max,
                            x_opt[t + 1,2] >= self.E_min,
                            x_opt[t + 1,2] <= self.E_max,
                            Power_cons >= 0,
                            Power_cons <= self.P_cons_max,
                            
                            cvx.abs(x_opt[t,:] - x_prev[t,:]) <= self.rho_state,
                            cvx.norm_inf(u_opt[t,:] - u_prev[t,:]) <= self.rho_u,
                            ]
                            # Preserved for a later variable-final-time version:
                            # cvx.abs(dt_opt - dt_prev) <= rho_dt

        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        prob.solve()
        # try:
        #     prob.solve(solver=cvx.CLARABEL)
        # except cvx.error.SolverError:
        #     prob.solve(solver=cvx.SCS, max_iters=5000, eps=1e-4, verbose=False)

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
        
        return x_opt.value, u_opt.value, nu_opt.value, prob.objective.value

    def solve_scp(self,x0,goal_position,N,eps,x_init=None,u_init=None,convergence_error=False,terminal_stop: bool = True):
        """Solve the obstacle avoidance problem via SCP."""

        # Initialize trajectory
        if x_init is None or u_init is None:
            x_bar = np.zeros((self.N + 1, self.n_state))
            u_bar = np.zeros((self.N, self.m_control))
            x_bar[0] = x0
            for k in range(self.N):
                x_bar[k + 1] = self.model.F(x_bar[k], u_bar[k], self.dt)
        else:
            x_bar = np.copy(x_init)
            u_bar = np.copy(u_init)

        converged = False
        merit_bar, objective_bar, defect_bar, virtual_bar = self.merit(x_bar, u_bar)

        for i in range(self.N_scp):
            x_old = x_bar.copy()
            u_old = u_bar.copy()
            x_new, u_new, nu_new, convex_objective = self.scp_iteration(
                x0,
                goal_position,
                x_bar,
                u_bar,
                terminal_stop=terminal_stop,
            )
            merit_new, objective_new, defect_new, virtual_new = self.merit(x_new, u_new, nu_new)
            accepted = merit_new <= merit_bar + self.eps_J

            if accepted:
                x_bar = x_new
                u_bar = u_new
                d_merit = abs(merit_bar - merit_new)
                merit_bar = merit_new
                objective_bar = objective_new
                defect_bar = defect_new
                virtual_bar = virtual_new
                self.rho_state *= self.beta_grow
                self.rho_u *= self.beta_grow
            else:
                d_merit = abs(merit_new - merit_bar)
                x_bar = x_old
                u_bar = u_old
                self.rho_state *= self.beta_shrink
                self.rho_u *= self.beta_shrink

            self.history.append(
                {
                    "iteration": i,
                    "accepted": bool(accepted),
                    "convex_objective": float(convex_objective),
                    "objective": float(objective_bar),
                    "candidate_objective": float(objective_new),
                    "merit": float(merit_bar),
                    "candidate_merit": float(merit_new),
                    "delta_merit": float(d_merit),
                    "defect": float(defect_bar),
                    "candidate_defect": float(defect_new),
                    "virtual": float(virtual_bar),
                    "candidate_virtual": float(virtual_new),
                    "final_time_s": float(self.final_time_s),
                    "rho_state_max": float(np.max(self.rho_state)),
                    "rho_u": float(self.rho_u),
                }
            )

            if accepted and d_merit < self.eps_J and defect_bar < self.eps_dyn and virtual_bar < self.eps_nu:
                converged = True
                break

        if not converged and convergence_error:
            raise RuntimeError("SCP did not converge!")
        
        return x_bar, u_bar
