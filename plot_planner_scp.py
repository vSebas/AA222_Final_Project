from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
from optimizer.scp_jax import SCP
from path_planner.planner import build_planner_solution

OUTPUT_PATH = PROJECT_DIR / "planner_scp_solution.png"

def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)))


def resample_path(points: np.ndarray, n_samples: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=float)
    if len(points) == 1:
        return np.repeat(points[:, :2], n_samples, axis=0)
    seg = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] <= 0.0:
        return np.repeat(points[:1, :2], n_samples, axis=0)
    s_eval = np.linspace(0.0, s[-1], n_samples)
    x = np.interp(s_eval, s, points[:, 0])
    y = np.interp(s_eval, s, points[:, 1])
    return np.column_stack((x, y))

def build_planner_path(n_waypoints: int = 40):
    return build_planner_solution(
        start_req=(-13_000.0, 0.0),
        goal_req=(13_000.0, 0.0),
        w_risk=10.0,
        n_resample=n_waypoints,
        smoothing_passes=3,
    )

def build_scp_from_path(path_points, max_horizon_steps: int = 100) -> SCP:
    points = np.asarray(path_points, dtype=float)
    nominal_speed = 0.4
    T_f = path_length(points) / nominal_speed       # final time
    # print("Number of points in path: ", len(points))

    # horizon_steps = min(max_horizon_steps, max(2, len(points) - 1))
    horizon_steps = max_horizon_steps
    dt = T_f / horizon_steps

    return SCP(
        dt=dt,
        N=horizon_steps,
        high_level_path=[point.tolist() for point in points],
    )

def power_trace(scp: SCP, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.array([scp.model.P(x[k], u[k]) for k in range(u.shape[0])], dtype=float)


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
    axes[0, 0].plot(x_plot[:, 0] / 1e3, x_plot[:, 1] / 1e3, color="cyan", lw=2.0, label="SCP trajectory")
    axes[0, 0].set_title("Planner Path and SCP Trajectory")
    axes[0, 0].set_xlabel("x [km]")
    axes[0, 0].set_ylabel("y [km]")
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(t_state, x_plot[:, 2], color="tab:green", lw=2.0)
    axes[0, 1].axhline(scp.E_min, color="black", ls="--", lw=1.0, label="E_min")
    axes[0, 1].set_title("Battery Energy")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("E [J]")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].step(t_control, u_plot[:, 0], where="post", lw=2.0)
    axes[1, 0].axhline(scp.v_max, color="black", ls="--", lw=1.0, label="v_max")
    axes[1, 0].set_title("Speed Command")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("speed [m/s]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].step(t_control, u_plot[:, 1], where="post", lw=2.0, color="tab:orange")
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
        iterations = [entry["iteration"] for entry in scp.history if "objective" in entry]
        objectives = [entry["objective"] for entry in scp.history if "objective" in entry]
        axes[2, 1].plot(iterations, objectives, marker="o", label="J")
        axes[2, 1].set_xlabel("iteration")
        axes[2, 1].set_ylabel("J")
        axes[2, 1].grid(True)
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

    high_level_path = np.asarray(high_level_path, dtype=float)
    scp = build_scp_from_path(high_level_path)

    x0 = np.array([high_level_path[0, 0], high_level_path[0, 1], scp.model.battery_charge_j, 0.0, 0.0, 0.0], dtype=float)
    x_goal = np.array([high_level_path[-1, 0], high_level_path[-1, 1], scp.model.min_battery_charge_j, 0.0, 0.0, 0.0], dtype=float)
    x_init = np.zeros((scp.N + 1, scp.n_state), dtype=float)
    u_init = np.zeros((scp.N, scp.m_control), dtype=float)
    path_xy = resample_path(high_level_path, scp.N + 1)
    path_delta = np.diff(path_xy, axis=0)
    path_dist = np.linalg.norm(path_delta, axis=1)
    path_heading = np.unwrap(np.arctan2(path_delta[:, 1], path_delta[:, 0]))
    path_speed = np.clip(path_dist / max(scp.dt, 1.0e-6), 0.0, scp.v_max)
    x_init[:, :2] = path_xy
    x_init[:, 2] = np.linspace(x0[2], x_goal[2], scp.N + 1)
    x_init[:-1, 3] = path_speed
    x_init[:-1, 4] = path_heading
    x_init[:-1, 5] = np.concatenate(([0.0], np.diff(path_heading) / max(scp.dt, 1.0e-6)))
    x_init[-1, 3] = x_init[-2, 3]
    x_init[-1, 4] = x_init[-2, 4]
    x_init[-1, 5] = x_init[-2, 5]
    for k in range(scp.N):
        u_init[k] = [path_speed[min(k, len(path_speed) - 1)], path_heading[min(k, len(path_heading) - 1)]]
    x_star, u_star = scp.solve_scp(x0, x_goal, scp.N, scp.eps, x_init=x_init, u_init=u_init)

    plot_results(terrain, raw_path, high_level_path, scp, x_init, u_init, x_star, u_star, OUTPUT_PATH)
    x_plot = x_star
    u_plot = u_star

    print(f"Saved planner + SCP plot: {OUTPUT_PATH}")
    print(f"Planner raw waypoints: {len(raw_path)}")
    print(f"SCP high-level path waypoints: {len(high_level_path)}")
    print(f"SCP horizon steps: {scp.horizon_steps}")
    print(f"SCP final time: {scp.final_time_s:.1f} s")
    print(f"Trajectory length: {path_length(x_plot):.1f} m")
    print(f"Final battery: {x_plot[-1, 2]:.2f} J")

if __name__ == "__main__":
    main()
