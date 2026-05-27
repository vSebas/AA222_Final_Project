from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scp import SCP


OUTPUT_PATH = Path(__file__).with_name("scp_solution.png")


def build_problem() -> SCP:
    return SCP(
        dt=0.1,
        final_time_s=4.0,
        max_iterations=20,
        high_level_path=[
            [0.0, 0.0],
            [0.6, 0.0],
            [1.0, 0.4],
        ],
    )


def power_trace(scp: SCP, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.array([scp.model.P(x[:, k], u[:, k], scp.dt) for k in range(scp.horizon_steps)])


def plot_solution(scp: SCP, x: np.ndarray, u: np.ndarray, output_path: Path) -> None:
    t_state = np.linspace(0.0, scp.final_time_s, scp.horizon_steps + 1)
    t_control = np.arange(scp.horizon_steps) * scp.dt
    power = power_trace(scp, x, u)
    waypoints = np.asarray(scp.high_level_path, dtype=float)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)

    axes[0, 0].plot(x[0], x[1], linewidth=2.0, label="SCP solution")
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

    axes[1, 0].step(t_control, u[0], where="post", linewidth=2.0)
    axes[1, 0].axhline(scp.v_max, color="black", linestyle="--", linewidth=1.0, label="v_max")
    axes[1, 0].set_title("Speed Command")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("speed [m/s]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].step(t_control, u[1], where="post", linewidth=2.0, color="tab:orange")
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
        iterations = [entry["iteration"] for entry in scp.history if "J_nl" in entry]
        objectives = [entry["J_nl"] for entry in scp.history if "J_nl" in entry]
        defects = [entry["defect_dyn"] for entry in scp.history if "defect_dyn" in entry]
        axes[2, 1].plot(iterations, objectives, marker="o", label="J_nl")
        axes[2, 1].set_ylabel("objective")
        ax_defect = axes[2, 1].twinx()
        ax_defect.plot(iterations, defects, marker="s", color="tab:purple", label="defect")
        ax_defect.set_ylabel("dynamics defect")
        axes[2, 1].set_title("SCP History")
        axes[2, 1].set_xlabel("iteration")
        axes[2, 1].grid(True)
    else:
        axes[2, 1].text(0.5, 0.5, "No SCP history", ha="center", va="center")
        axes[2, 1].set_axis_off()

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    scp = build_problem()
    x_star, u_star = scp.scp_algorithm()
    plot_solution(scp, x_star, u_star, OUTPUT_PATH)

    print(f"Saved SCP solution plot: {OUTPUT_PATH}")
    print(f"Final position: {x_star[:2, -1]}")
    print(f"Final battery: {x_star[2, -1]:.2f} J")
    print(f"Objective: {scp.nonlinear_objective(x_star, u_star):.6f}")
    if scp.history:
        print(f"Iterations: {len(scp.history)}")
        print(f"Last history entry: {scp.history[-1]}")


if __name__ == "__main__":
    main()
