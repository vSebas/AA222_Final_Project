from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_PATH = PROJECT_DIR / "planner_scp_trajectory.npz"
OUTPUT_PATH = PROJECT_DIR / "planner_scp_solution_hours.png"


def main() -> None:
    data = np.load(CACHE_PATH)
    raw = data["raw_path"]
    high = data["high_level_path"]
    x = data["x_star"]
    u = data["u_star"]
    t_state_h = data["t_state"] / 3600.0
    t_control_h = data["t_control"] / 3600.0
    power_w = data["power_w"]
    cumulative_energy_mj = data["cumulative_energy_j"] / 1.0e6
    battery_mj = data["battery_state_j"] / 1.0e6
    rollout_battery_mj = data["battery_rollout_j"] / 1.0e6

    # This is the same unweighted objective convention stored in the cache:
    # J = energy [J] + time [s]. It is useful as an optimization diagnostic,
    # not as a physical unit by itself.
    cumulative_J = data["cumulative_energy_j"] + data["t_state"]
    cumulative_J_million = cumulative_J / 1.0e6

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    axes[0, 0].plot(raw[:, 0] / 1e3, raw[:, 1] / 1e3, color="0.65", lw=1.0, label="A* raw")
    axes[0, 0].plot(high[:, 0] / 1e3, high[:, 1] / 1e3, color="tab:green", lw=2.0, label="smoothed path")
    axes[0, 0].plot(x[:, 0] / 1e3, x[:, 1] / 1e3, color="tab:blue", lw=2.0, label="SCP trajectory")
    axes[0, 0].set_title("Full-Route Trajectory")
    axes[0, 0].set_xlabel("x [km]")
    axes[0, 0].set_ylabel("y [km]")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(t_state_h, battery_mj, color="tab:green", lw=2.0, label="SCP state")
    axes[0, 1].plot(t_state_h, rollout_battery_mj, color="tab:olive", ls="--", lw=1.5, label="nonlinear rollout")
    axes[0, 1].set_title("Battery Energy")
    axes[0, 1].set_xlabel("time [h]")
    axes[0, 1].set_ylabel("E [MJ]")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(t_state_h, x[:, 3], color="tab:purple", lw=2.0, label="state speed")
    axes[1, 0].step(t_control_h, u[:, 0], where="post", color="tab:blue", lw=1.4, alpha=0.75, label="speed command")
    axes[1, 0].set_title("Speed")
    axes[1, 0].set_xlabel("time [h]")
    axes[1, 0].set_ylabel("speed [m/s]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(t_state_h, x[:, 4], color="tab:brown", lw=2.0, label="state heading")
    axes[1, 1].step(t_control_h, u[:, 1], where="post", color="tab:orange", lw=1.4, alpha=0.75, label="heading command")
    axes[1, 1].set_title("Heading")
    axes[1, 1].set_xlabel("time [h]")
    axes[1, 1].set_ylabel("heading [rad]")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

    axes[2, 0].step(t_control_h, power_w, where="post", color="tab:red", lw=2.0)
    axes[2, 0].set_title("Power Consumption")
    axes[2, 0].set_xlabel("time [h]")
    axes[2, 0].set_ylabel("power [W]")
    axes[2, 0].grid(True)

    axes[2, 1].plot(t_state_h, cumulative_energy_mj, color="tab:cyan", lw=2.0, label="energy")
    axes[2, 1].plot(t_state_h, cumulative_J_million, color="tab:red", ls="--", lw=1.5, label="J = energy + time")
    axes[2, 1].set_title("Cumulative Energy and Objective")
    axes[2, 1].set_xlabel("time [h]")
    axes[2, 1].set_ylabel("MJ / 1e6 objective units")
    axes[2, 1].grid(True)
    axes[2, 1].legend()

    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
