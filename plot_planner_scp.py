from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from optimizer.scp import SCP, cp
from path_planner.planner import build_planner_solution

THIS_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = THIS_DIR / "planner_scp_solution.png"


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))


def build_planner_path(n_waypoints: int = 40):
    return build_planner_solution(
        start_req=(-13_000.0, 0.0),
        goal_req=(13_000.0, 0.0),
        w_risk=10.0,
        n_resample=n_waypoints,
        smoothing_passes=3,
    )


def build_scp_from_path(path_points, max_horizon_steps: int = 80) -> SCP:
    points = np.asarray(path_points, dtype=float)
    nominal_speed = 0.4
    final_time_s = path_length(points) / nominal_speed
    horizon_steps = min(max_horizon_steps, max(2, len(points) - 1))
    dt = final_time_s / horizon_steps
    final_time_s = horizon_steps * dt

    return SCP(
        dt=dt,
        final_time_s=final_time_s,
        max_iterations=5,
        high_level_path=[point.tolist() for point in points],
    )


def power_trace(scp: SCP, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.array([scp.model.P(x[:, k], u[:, k], scp.dt) for k in range(scp.horizon_steps)])


def plot_results(
    terrain,
    raw_path,
    smooth_path,
    scp: SCP,
    x_nominal: np.ndarray,
    u_nominal: np.ndarray,
    x_star: np.ndarray | None,
    u_star: np.ndarray | None,
    output_path: Path,
) -> None:
    raw = np.asarray(raw_path, dtype=float)
    smooth = np.asarray(smooth_path, dtype=float)
    t_state = np.linspace(0.0, scp.final_time_s, scp.horizon_steps + 1)
    t_control = np.arange(scp.horizon_steps) * scp.dt

    x_plot = x_star if x_star is not None else x_nominal
    u_plot = u_star if u_star is not None else u_nominal
    power = power_trace(scp, x_plot, u_plot)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)

    extent_km = [value / 1e3 for value in terrain.extent_m]
    risk_show = np.clip(terrain.risk, 0, np.percentile(terrain.risk, 99.5))
    image = axes[0, 0].imshow(risk_show, extent=extent_km, origin="lower", cmap="inferno")
    fig.colorbar(image, ax=axes[0, 0], label="risk")
    axes[0, 0].plot(raw[:, 0] / 1e3, raw[:, 1] / 1e3, color="white", lw=1.2, label="A* raw")
    axes[0, 0].plot(smooth[:, 0] / 1e3, smooth[:, 1] / 1e3, color="lime", lw=2.0, label="smoothed path")
    axes[0, 0].plot(x_plot[0] / 1e3, x_plot[1] / 1e3, color="cyan", lw=2.0, label="SCP trajectory")
    axes[0, 0].set_title("Planner Path and SCP Trajectory")
    axes[0, 0].set_xlabel("x [km]")
    axes[0, 0].set_ylabel("y [km]")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(t_state, x_plot[2], color="tab:green", lw=2.0)
    axes[0, 1].axhline(scp.E_min, color="black", ls="--", lw=1.0, label="E_min")
    axes[0, 1].set_title("Battery Energy")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("E [J]")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].step(t_control, u_plot[0], where="post", lw=2.0)
    axes[1, 0].axhline(scp.v_max, color="black", ls="--", lw=1.0, label="v_max")
    axes[1, 0].set_title("Speed Command")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("speed [m/s]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].step(t_control, u_plot[1], where="post", lw=2.0, color="tab:orange")
    axes[1, 1].set_title("Heading Command")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("heading [rad]")
    axes[1, 1].grid(True)

    axes[2, 0].step(t_control, power, where="post", lw=2.0, color="tab:red")
    axes[2, 0].axhline(scp.P_cons_max, color="black", ls="--", lw=1.0, label="P_cons_max")
    axes[2, 0].axhline(scp.model.power_generation_w, color="tab:blue", ls="--", lw=1.0, label="generation")
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
        axes[2, 1].set_xlabel("iteration")
        axes[2, 1].set_ylabel("objective")
        axes[2, 1].grid(True)
        defect_axis = axes[2, 1].twinx()
        defect_axis.plot(iterations, defects, marker="s", color="tab:purple", label="defect")
        defect_axis.set_ylabel("dynamics defect")
        axes[2, 1].set_title("SCP Iteration History")
    else:
        axes[2, 1].text(
            0.5,
            0.5,
            "SCP solve skipped;\nshowing nominal warm start",
            ha="center",
            va="center",
        )
        axes[2, 1].set_axis_off()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    try:
        terrain, raw_path, high_level_path, _, _ = build_planner_path()
    except ImportError as exc:
        raise SystemExit(str(exc)) from None

    scp = build_scp_from_path(high_level_path)
    x_nominal, u_nominal = scp.initialize_nominal_trajectory()

    x_star = None
    u_star = None
    if cp is None:
        print("cvxpy is not installed; plotting SCP nominal warm start only.")
    else:
        x_star, u_star = scp.scp_algorithm()

    plot_results(terrain, raw_path, high_level_path, scp, x_nominal, u_nominal, x_star, u_star, OUTPUT_PATH)
    x_plot = x_star if x_star is not None else x_nominal
    u_plot = u_star if u_star is not None else u_nominal

    print(f"Saved planner + SCP plot: {OUTPUT_PATH}")
    print(f"Planner raw waypoints: {len(raw_path)}")
    print(f"SCP high-level path waypoints: {len(high_level_path)}")
    print(f"SCP horizon steps: {scp.horizon_steps}")
    print(f"SCP final time: {scp.final_time_s:.1f} s")
    print(f"Trajectory length: {path_length(x_plot[:2].T):.1f} m")
    print(f"Final battery: {x_plot[2, -1]:.2f} J")
    print(f"Objective: {scp.nonlinear_objective(x_plot, u_plot):.6f}")


if __name__ == "__main__":
    main()
