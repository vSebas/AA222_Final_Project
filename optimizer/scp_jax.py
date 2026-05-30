from functools import partial
from dataclasses import dataclass, field
from time import time

import cvxpy as cvx

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import numpy as np

from tqdm.auto import tqdm

MODELING_DIR = Path(__file__).resolve().parents[1] / "modeling"
from rover_dynamic_model import RoverModel

@dataclass
class SCP:
    n_state: int = 6     # state dimension
    m_control: int = 2   # control dimension
    eps: int = 1e-3      # SCP convergence tolerance
    final_time_s: float = 1.0

    N = 5           # MPC horizon
    N_scp = 25       # maximum number of SCP iterations

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

    # x0 = np.array([-1.0, -1.0, 0.0, 0.0])  # initial state
    # x_goal = np.array([1.0, 1.0, 0.0, 0.0])  # desired final state
    # P = 1e2 * np.eye(n_state)  # terminal state cost matrix
    # Q = 1e1 * np.eye(n_state)  # state cost matrix
    # R = 1e-2 * np.eye(m_control)  # control cost matrix
    # # Set obstacle center points and radii
    # centers = np.array(
    #     [
    #         [-0.6, -0.4],
    #         [0.6, 0.1],
    #     ]
    # )
    # radii = np.array([0.5, 0.5])

    @partial(jax.jit, static_argnums=(0,))
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def affinize(self, f, x_bar, u_bar):
        """
        Affinize the function `f(s, u)` around `(x_bar, u_bar)`.
        First-order Taylor Expansion
        """
        A,B = jax.jacobian(f, argnums=(0,1))(x_bar,u_bar)
        c = f(x_bar,u_bar) - A @ x_bar - B @ u_bar
        return np.array(A), np.array(B), np.array(c)

    # def nonlinear_dynamics_defect(self, x, u):
    #     """Maximum one-step mismatch against the true nonlinear dynamics."""

    #     x = np.asarray(x, dtype=float)
    #     u = np.asarray(u, dtype=float)
    #     defects = []
    #     for k in range(u.shape[1]):
    #         defects.append(x[:, k + 1] - self.model.F(x[:, k], u[:, k], self.dt))
    #     return 0.0 if not defects else float(np.max(np.linalg.norm(defects, ord=np.inf, axis=1)))


    def scp_iteration(self, x0, x_goal, x_prev, u_prev):
        """Solve a single SCP sub-problem for trajectory optimization."""

        Af, Bf, cf = self.affinize(self.model.F, x_prev[:-1], u_prev)
        Ap, Bp, cp = self.affinize(self.model.P, x_prev[:-1], u_prev)

        x_opt = cvx.Variable((self.N + 1, self.n_state))    # X, Y, E, speed, heading, omega
        u_opt = cvx.Variable((self.N, self.m_control))      # speed, heading
        nu_opt = cp.Variable((self.N, self.n_states))       # virtual dynamics control

        constraints = [ x_opt[0,:] == x0,
                        x_opt[self.N,:] == x_goal ]
        objective = self.ct * self.final_time_s

        for t in range(self.N):
            Power_cons = Ap @ x_opt[t,:] + Bp @ u_opt[t,:] + cp

            objective += self.ce * Power_cons * self.dt
            objective += self.c_nu * cp.norm1(nu_opt[t,:])

            constraints += [
                            x_opt[t + 1,:] == Af @ x_opt[t,:] + Bf @ u_opt[t,:] + cf + nu_opt[t,:],
                            # P_k \in C_k     # corridor, from high-level path
                            u_opt[t,0] >= 0,
                            u_opt[t,0] <= self.v_max,
                            # accel constraints
                            x_opt[t,2] >= self.E_min,
                            x_opt[t,2] <= self.E_max,
                            x_opt[t + 1,2] >= self.E_min,
                            x_opt[t + 1,2] <= self.E_max,
                            Power_cons <= self.P_cons_max,
                            cp.norm_inf(x_opt[t,:] - x_opt[t,:]) <= self.rho_x,
                            cp.norm_inf(u_opt[t,:] - u_opt[t,:]) <= self.rho_u,
                            ]

        prob = cvx.Problem(cvx.Minimize(objective), constraints)
        prob.solve()

        if prob.status != "optimal":
            raise RuntimeError("SCP solve failed. Problem status: " + prob.status)
        
        return x_opt.value, u_opt.value, nu_opt.value, prob.objective.value

    def solve_scp(self,x0,x_goal,N,eps,
                                     x_init=None,u_init=None,convergence_error=False):
        """Solve the obstacle avoidance problem via SCP."""

        # Initialize trajectory
        if x_init is None or u_init is None:
            x_bar = np.zeros((self.N + 1, self.n))
            u_bar = np.zeros((self.N, self.m))
            x_bar[0] = x0
            for k in range(self.N):
                x_bar[k + 1] = self.model.f(x_bar[k], u_bar[k])
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