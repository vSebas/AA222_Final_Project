from functools import partial
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

    model: RoverModel = field(init=False)

    ct: float = 0.0
    ce: float = 1.0
    c_nu: float = 1.0e4

    rho_x: float = 10.0
    rho_u: float = 1.0

    high_level_path: list = field(default_factory=list)
    history: list = field(default_factory=list)
    corridor_radius_m: float = 1500.0

    beta_grow: float = 1.1
    beta_shrink: float = 0.5
    eps_dyn: float = 0.5
    eps_nu: float = 1.0e-4
    eps_x: float = 1.0e-3
    eps_J: float = 1.0e-3

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

    def path_positions(self, n_points: int) -> np.ndarray | None:
        """Interpolate the stored high-level path to ``n_points`` samples."""

        pts = np.asarray(self.high_level_path, dtype=float)
        if pts.size == 0:
            return None
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("high_level_path must contain XY waypoints")
        if len(pts) == 1:
            return np.repeat(pts[:1, :2], n_points, axis=0)

        seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        if s[-1] <= 0.0:
            return np.repeat(pts[:1, :2], n_points, axis=0)
        s_eval = np.linspace(0.0, s[-1], n_points)
        x = np.interp(s_eval, s, pts[:, 0])
        y = np.interp(s_eval, s, pts[:, 1])
        return np.column_stack((x, y))

    def scp_iteration(self, x0, x_goal, x_prev, u_prev):
        """Solve a single SCP sub-problem for trajectory optimization."""

        Af, Bf, cf = self.affinize(lambda x, u: self.model.jax_F(x, u), x_prev[:-1], u_prev)
        Ap, Bp, pc = self.affinize(lambda x, u: self.model.jax_P(x, u), x_prev[:-1], u_prev)
        path_xy = self.path_positions(self.N + 1)

        x_opt = cvx.Variable((self.N + 1, self.n_state))    # X, Y, E, speed, heading, omega
        u_opt = cvx.Variable((self.N, self.m_control))      # speed, heading
        nu_opt = cvx.Variable((self.N, self.n_state))       # virtual dynamics control

        objective = 0.0
        constraints = [ x_opt[0,:] == x0,
                        x_opt[self.N,:] == x_goal ]
        # objective = self.ct * self.final_time_s

        for t in range(self.N):
            Power_cons = Ap[t] @ x_opt[t,:] + Bp[t] @ u_opt[t,:] + pc[t]

            objective += self.ce * Power_cons * self.dt
            objective += self.c_nu * cvx.norm1(nu_opt[t,:])

            constraints += [
                            x_opt[t + 1,:] == Af[t] @ x_opt[t,:] + Bf[t] @ u_opt[t,:] + cf[t] + nu_opt[t,:],
                            cvx.norm_inf(x_opt[t, :2] - path_xy[t]) <= self.corridor_radius_m,
                            cvx.norm_inf(x_opt[t + 1, :2] - path_xy[t + 1]) <= self.corridor_radius_m,
                            u_opt[t,0] >= 0,
                            u_opt[t,0] <= self.v_max,
                            # accel constraints
                            x_opt[t,2] >= self.E_min,
                            x_opt[t,2] <= self.E_max,
                            x_opt[t + 1,2] >= self.E_min,
                            x_opt[t + 1,2] <= self.E_max,
                            Power_cons <= self.P_cons_max,
                            cvx.norm_inf(x_opt[t,:] - x_prev[t,:]) <= self.rho_x,
                            cvx.norm_inf(u_opt[t,:] - u_prev[t,:]) <= self.rho_u,
                            ]

        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        try:
            prob.solve(solver=cvx.CLARABEL)
        except cvx.error.SolverError:
            prob.solve(solver=cvx.SCS, max_iters=5000, eps=1e-4, verbose=False)

        if prob.status not in {"optimal", "optimal_inaccurate"}:
            raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
        
        return x_opt.value, u_opt.value, nu_opt.value, prob.objective.value

    def solve_scp(self,x0,x_goal,N,eps,
                                     x_init=None,u_init=None,convergence_error=False):
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

        # Do SCP until convergence or maximum number of iterations is reached
        converged = False
        J_bar = np.zeros(self.N_scp + 1)
        J_bar[0] = np.inf

        for i in range(self.N_scp):
            x_bar, u_bar, nu_bar, J_bar[i + 1] = self.scp_iteration(x0, x_goal, x_bar, u_bar)
            dJ_bar = np.abs(J_bar[i + 1] - J_bar[i])
            self.history.append(
                {
                    "iteration": i,
                    "objective": float(J_bar[i + 1]),
                    "delta_objective": float(dJ_bar),
                }
            )

            # defect_dyn = self.nonlinear_dynamics_defect(x_bar, u_bar)
            # virtual_norm = float(np.max(np.abs(nu_bar)))
            # state_change = float(np.max(np.abs(x_bar - x_bar)))
            # control_change = float(np.max(np.abs(u_bar - u_bar)))
            # J_bar = self.nonlinear_objective(x_bar, u_bar)
            # cost_change = abs(J_bar - J_bar)

            # accepted = defect_dyn <= self.eps_dyn and J_candidate <= J_bar + self.eps_J
            
            if dJ_bar < eps:
                converged = True
                break
            else:
                self.rho_x *= self.beta_shrink
                self.rho_u *= self.beta_shrink

        if not converged and convergence_error:
            raise RuntimeError("SCP did not converge!")
        
        return x_bar, u_bar
