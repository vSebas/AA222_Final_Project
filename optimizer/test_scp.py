from __future__ import annotations

import os
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from scp_jax import SCP


def build_initial_guess(scp: SCP, start: np.ndarray, goal: np.ndarray):
    x_init = np.zeros((scp.N + 1, scp.n_state), dtype=float)
    u_init = np.zeros((scp.N, scp.m_control), dtype=float)
    x_init[0] = start

    delta = goal[:2] - start[:2]
    distance = float(np.linalg.norm(delta))
    heading = float(np.arctan2(delta[1], delta[0])) if distance > 0.0 else 0.0
    speed = min(scp.v_max, distance / max(scp.N * scp.dt, 1.0))

    for k in range(scp.N):
        alpha = (k + 1) / scp.N
        x_init[k + 1] = (1.0 - alpha) * start + alpha * goal
        x_init[k + 1, 3] = speed
        x_init[k + 1, 4] = heading
        x_init[k + 1, 5] = 0.0
        u_init[k] = [speed, heading]

    return x_init, u_init


def main():
    scp = SCP(dt=1.0, final_time_s=1000.0)

    start = np.array([0.0, 0.0, scp.model.battery_charge_j, 0.0, 0.0, 0.0], dtype=float)
    goal = np.array([1000.0, 0.0, scp.model.min_battery_charge_j, 0.0, 0.0, 0.0], dtype=float)
    x_init, u_init = build_initial_guess(scp, start, goal)

    x_star, u_star = scp.solve_scp(start, goal, scp.N, scp.eps, x_init=x_init, u_init=u_init)

    print(f"x_star shape: {x_star.shape}")
    print(f"u_star shape: {u_star.shape}")
    print(f"start: {x_star[0, :3]}")
    print(f"goal:  {x_star[-1, :3]}")


if __name__ == "__main__":
    main()
