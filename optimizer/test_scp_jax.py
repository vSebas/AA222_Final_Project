from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

from scp_jax import SCP
import cvxpy as cvx


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


def power_trace(scp: SCP, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.array([scp.model.P(x[t], u[t], scp.dt) for t in range(u.shape[0])], dtype=float)

def plot_solution(scp: SCP, x: np.ndarray, u: np.ndarray, output_path: Path) -> None:
    t_state = np.linspace(0.0, scp.final_time_s, x.shape[0])
    t_control = np.linspace(0.0, scp.final_time_s, u.shape[0], endpoint=False)
    power = power_trace(scp, x, u)
    waypoints = np.asarray(scp.high_level_path, dtype=float)
    if waypoints.ndim == 1:
        waypoints = waypoints.reshape(-1, 2)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    axes[0, 0].plot(x[:, 0], x[:, 1], linewidth=2.0, label="SCP solution")
    axes[0, 0].scatter(waypoints[:, 0], waypoints[:, 1], s=35, label="path waypoints")
    axes[0, 0].set_title("Planar Trajectory")
    axes[0, 0].set_xlabel("X [m]")
    axes[0, 0].set_ylabel("Y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(t_state, x[2], color="tab:green", linewidth=2.0)
    axes[0, 1].axhline(scp.E_min, color="black", linestyle="--", linewidth=1.0, label="E_min")
    axes[0, 1].set_title("Battery Energy")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("E [J]")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].step(t_control, u[:, 0], where="post", linewidth=2.0)
    axes[1, 0].axhline(scp.v_max, color="black", linestyle="--", linewidth=1.0, label="v_max")
    axes[1, 0].set_title("Speed Command")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("speed [m/s]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].step(t_control, u[:, 1], where="post", linewidth=2.0, color="tab:orange")
    axes[1, 1].set_title("Heading Command")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("heading [rad]")
    axes[1, 1].grid(True)

    axes[2, 0].step(t_control, power, where="post", linewidth=2.0, color="tab:red")
    axes[2, 0].axhline(scp.P_cons_max, color="black", linestyle="--", linewidth=1.0, label="P_cons_max")
    axes[2, 0].axhline(scp.model.power_generation_w, color="tab:blue", linestyle="--", linewidth=1.0, label="generation")
    axes[2, 0].set_title("Power Consumption")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("power [W]")
    axes[2, 0].grid(True)
    axes[2, 0].legend()

    if scp.history:
        iterations = [entry.get("iteration", i) for i, entry in enumerate(scp.history)]
        objectives = [entry["objective"] for entry in scp.history if "objective" in entry]
        if objectives:
            axes[2, 1].plot(iterations[: len(objectives)], objectives, marker="o", label="objective")
            axes[2, 1].set_ylabel("objective")
            axes[2, 1].set_title("SCP History")
            axes[2, 1].set_xlabel("iteration")
            axes[2, 1].grid(True)
    else:
        axes[2, 1].text(0.5, 0.5, "No SCP history", ha="center", va="center")
        axes[2, 1].set_axis_off()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    scp = SCP(dt=1.0, final_time_s=1000.0)

    start = np.array([0.0, 0.0, scp.model.battery_charge_j, 0.0, 0.0, 0.0], dtype=float)
    goal = np.array([500.0, 500.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    x_init, u_init = build_initial_guess(scp, start, goal)

    x_star, u_star = scp.solve_scp(start, goal, scp.N, scp.eps, x_init=x_init, u_init=u_init)

    print(f"x_star shape: {x_star.shape}")
    print(f"u_star shape: {u_star.shape}")
    print(f"start: {x_star[0, :3]}")
    print(f"goal:  {x_star[-1, :3]}")
    plot_solution(scp, x_star, u_star, Path("scp_solution.png"))


if __name__ == "__main__":
    main()
