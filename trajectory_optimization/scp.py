import cvxpy as cp
import numpy as np

from dataclasses import dataclass

from rover_dynamic_model import RoverModel

ArrayLike = float | np.ndarray

# def diff(f, x, u, dt):
#     # Numerical differentiation
#     # Central difference method
#     # f'(x) = [f(xk + dt, uk) - f(xk - dt, uk)] / 2*dt
#     return ( f((x + dt), u) - f((x - dt), u) ) / (2*dt)

@dataclass
class SCP:
    nu_k: ArrayLike = 0.0
    A_k: ArrayLike = 0.0
    B_k: ArrayLike = 0.0

    x_bar: ArrayLike = 0.0

    model = RoverModel(
        battery_charge_j=20_000.0,
        power_generation_w=65.0,
        phi=np.deg2rad(4.0),
        xi=np.deg2rad(3.0),
    )

    ct: float = 0.0
    ce: float = 0.0

    v_max: float = 0.0
    E_min: float = 0.0
    E_max: float = 0.0
    P_cons_max: float = 0.0

    rho_x: float = 0.0
    rho_u: float = 0.0

    high_level_path = []

    beta_grow: float = 1.1
    beta_shrink: float = 0.5
    eps_dyn: float = 0.5

    
    def linearize_nonlinear_dynamics(f,x,u):
        # x is x_k
        # u is u_k
        A_k = ( f(x + self.dt, u) - f(x - self.dt, u) ) / (2*self.dt)
        B_k = ( f(x, u + self.dt) - f(x, u - self.dt) ) / (2*self.dt)

        f_k = self.model.f(x_bar, u_bar) + A_k @ (x - self.x_bar) + B_k @ (u_k - u_bar)
        return A_k, B_k, f_k

    def convex_ocp():
        n_states = 5
        n_inputs = 2
        N = 1           # dummy, horizon

        x = cp.Variable((n_states, N + 1))  # x_pos, y_pos, psi (heading), v (forward speed), E (energy)
        u = cp.Variable((n_inputs, N))      # speed, psi

        # Initial state and cost weights

        x0 = self.high_level_path[0]
        xN = self.high_level_path[-1]

        constraints = [ x[:, 0] == x0,
                        x[:, N] == xN,]
        cost = 0

        # Define cost function and dynamics over the horizon
        for t in range(N):
            # cost += cp.sum_squares(x[:, t]) + cp.sum_squares(u[:, t])

            P_cons_k = self.linearize_nonlinear_dynamics(f,x,u)*self.dt + self.nu_k
            cost += self.ce/N * P_cons_k * self.dt

            constraints += [
                            x[:, t+1] == self.linearize_nonlinear_dynamics(f,x,u) + self.nu_k,
                            # P_k \in C_k     # corridor, from high-level path
                            -u[0] <= 0,
                            u[0] <= self.v_max,
                            # accel constraints
                            -x[4] <= self.E_min,
                            x[4] <= self.E_max,
                            P_cons_k <= self.P_cons_max,
                            cp.norm_inf(x - x_bar) <= self.rho_x,
                            cp.norm_inf(u - u_bar) <= self.rho_u,
                            ]
            
        cost += [ self.ct*Tf ]# (is Tf == N?)

        objective = cp.Minimize(cost)
        prob = cp.Problem(objective, constraints)
        prob.solve()

        u_star = u.value
        x_star = x.value

        return u_star, x_star

    def scp_algorithm():

        stop_criteria = True

        j = 0
        x_bar = 0
        u_star = 0

        while(stop_criteria):
            # 1. Build Convex Optimal Control Subproblem

            u_star, x_star = self.convex_ocp()

            # 2. Update Rule

            alpha_dyn = max(np.linalg.norm(x_star - F(x_star, u_star), ord=np.inf))
            # J_nl(x_star, u_star) = ce * sum(P_cons_k) * self.dt + self.ct*tf
            # J_nl(x_bar, u_bar) = ce * sum(P_cons_k) * self.dt + self.ct*tf

            
            if alpha_dyn <= self.eps_dyn and self.J_nl_star <= self.J_nl_bar:
                self.rho_x *= self.beta_grow
                self.rho_u *= self.beta_grow

                x_bar = x_star
                u_bar = u_star

            else:
                self.rho_x *= self.beta_shrink
                self.rho_u *= self.beta_shrink

            
            # Stopping Criteria
            defect_dyn = alpha_dyn <= self.eps_dyn
            virtual = (max(np.linal.norm(self.nu_k,ord=np.inf))) <= self.eps_nu
            position = (max(np.linal.norm(self.x_star - x_bar,ord=np.inf))) <= self.eps_nu
            cost = abs(Jnl (j+1) - Jnl(j)) <= self.eps_J
            stop_criteria = defect_dyn and virtual and position and cost


        return x_star, u_star

def main():
    scp = SCP()

    # set high level path
    x_star,u_star = scp.scp_algorithm()


if __name__ == "__main__":
    main()