from __future__ import annotations

import os
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
LOG_PATH = PROJECT_DIR / "planner_scp_solution.log"
HISTORY_PATH = PROJECT_DIR / "planner_scp_history.csv"
SEGMENT_WAYPOINTS = 5
SEGMENT_DT_S = 200.0
MIN_SEGMENT_STEPS = 30
SLOW_SPEED_MPS = 0.2

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

def build_scp_from_path(path_points, dt_s: float = SEGMENT_DT_S) -> SCP:
    points = np.asarray(path_points, dtype=float)
    length = path_length(points)
    vmax = 0.45
    if length > 0.0:
        Tf_min = length / vmax
        Tf_max = length / SLOW_SPEED_MPS
    else:
        Tf_min = dt_s
        Tf_max = 2.0 * dt_s
    Tf_guess = 0.5 * (Tf_min + Tf_max)
    horizon_steps = max(MIN_SEGMENT_STEPS, int(np.ceil(Tf_guess / dt_s)))
    dt = Tf_guess / horizon_steps if Tf_guess > 0.0 else dt_s
    return SCP(
        dt=dt,
        N=horizon_steps,
        Tf_min=Tf_min,
        Tf_max=Tf_max,
        rho_Tf=max(dt_s, 0.25 * (Tf_max - Tf_min)),
        high_level_path=[point.tolist() for point in points],
    )

def power_trace(scp: SCP, x: np.ndarray, u: np.ndarray, dt_values: np.ndarray | None = None) -> np.ndarray:
    if dt_values is None:
        dt_values = np.full(u.shape[0], scp.dt)
    return np.array([scp.model.P(x[k], u[k], dt_values[k]) for k in range(u.shape[0])], dtype=float)


def nonlinear_rollout(scp: SCP, x0: np.ndarray, u: np.ndarray, dt_values: np.ndarray | None = None) -> np.ndarray:
    if dt_values is None:
        dt_values = np.full(u.shape[0], scp.dt)
    x = np.zeros((u.shape[0] + 1, scp.n_state), dtype=float)
    x[0] = x0
    for k in range(u.shape[0]):
        x[k + 1] = scp.model.F(x[k], u[k], dt_values[k])
    return x


def dynamics_defect(scp: SCP, x: np.ndarray, u: np.ndarray, dt_values: np.ndarray | None = None) -> float:
    if dt_values is None:
        dt_values = np.full(u.shape[0], scp.dt)
    defects = [x[k + 1] - scp.model.F(x[k], u[k], dt_values[k]) for k in range(u.shape[0])]
    return 0.0 if not defects else float(np.max(np.linalg.norm(defects, ord=np.inf, axis=1)))


def build_warm_start(
    scp: SCP,
    path_points: np.ndarray,
    x0: np.ndarray | None = None,
    terminal_stop: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    path_xy = resample_path(path_points, scp.N + 1)
    path_delta = np.diff(path_xy, axis=0)
    path_dist = np.linalg.norm(path_delta, axis=1)
    path_heading = np.unwrap(np.arctan2(path_delta[:, 1], path_delta[:, 0]))
    path_speed = np.clip(path_dist / max(scp.dt, 1.0e-6), 0.0, scp.v_max)

    x_init = np.zeros((scp.N + 1, scp.n_state), dtype=float)
    u_init = np.zeros((scp.N, scp.m_control), dtype=float)
    u_init[:, 0] = path_speed
    u_init[:, 1] = path_heading
    if x0 is None:
        x0 = np.array([path_xy[0, 0], path_xy[0, 1], scp.model.battery_charge_j, 0.0, path_heading[0], 0.0], dtype=float)
    x_init[0] = x0
    u_init[0, 0] = x0[3]
    if terminal_stop:
        u_init[-1, 0] = 0.0
    for k in range(scp.N):
        x_init[k + 1] = scp.model.F(x_init[k], u_init[k])
    return x_init, u_init


def path_chunks(points: np.ndarray, chunk_waypoints: int = SEGMENT_WAYPOINTS):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return
    step = max(1, chunk_waypoints - 1)
    for start in range(0, len(points) - 1, step):
        end = min(start + chunk_waypoints, len(points))
        chunk = points[start:end]
        if len(chunk) >= 2:
            yield chunk
        if end == len(points):
            break


def solve_chunked_path(path_points: np.ndarray):
    x_parts: list[np.ndarray] = []
    u_parts: list[np.ndarray] = []
    dt_parts: list[np.ndarray] = []
    scps: list[SCP] = []
    x_prev_final: np.ndarray | None = None

    chunks = list(path_chunks(path_points))
    for i, chunk in enumerate(chunks, start=1):
        terminal_stop = i == len(chunks)
        scp = build_scp_from_path(chunk)
        x_init, u_init = build_warm_start(scp, chunk, x0=x_prev_final, terminal_stop=terminal_stop)

        start_state = x_init[0].copy()
        goal_position = chunk[-1, :2]
        print(
            f"chunk {i}: length={path_length(chunk):.1f} m, "
            f"N={scp.N}, dt_guess={scp.dt:.2f} s, "
            f"Tf_bounds=[{scp.Tf_min:.1f}, {scp.Tf_max:.1f}] s"
        )
        x_star, u_star = scp.solve_scp(
            start_state,
            goal_position,
            scp.N,
            scp.eps,
            x_init=x_init,
            u_init=u_init,
            terminal_stop=terminal_stop,
        )

        if x_parts:
            x_parts.append(x_star[1:])
        else:
            x_parts.append(x_star)
        u_parts.append(u_star)
        dt_parts.append(np.full(u_star.shape[0], scp.final_time_s / scp.N))
        scps.append(scp)
        x_prev_final = x_star[-1].copy()

    if not x_parts:
        raise RuntimeError("No path chunks were generated.")
    return np.vstack(x_parts), np.vstack(u_parts), np.concatenate(dt_parts), scps


def write_run_logs(
    log_path: Path,
    history_path: Path,
    raw_path,
    high_level_path: np.ndarray,
    x_star: np.ndarray,
    u_star: np.ndarray,
    dt_values: np.ndarray,
    scps: list[SCP],
) -> None:
    total_time = float(np.sum(dt_values))
    total_distance = path_length(x_star)
    defect = dynamics_defect(scps[0], x_star, u_star, dt_values)
    powers = power_trace(scps[0], x_star, u_star, dt_values)
    energy_used = float(np.sum(powers * dt_values))

    with log_path.open("w", encoding="utf-8") as f:
        f.write("Planner + SCP run summary\n")
        f.write(f"raw_waypoints={len(raw_path)}\n")
        f.write(f"high_level_waypoints={len(high_level_path)}\n")
        f.write(f"chunks={len(scps)}\n")
        f.write(f"total_horizon_steps={u_star.shape[0]}\n")
        f.write(f"total_time_s={total_time:.6f}\n")
        f.write(f"trajectory_length_m={total_distance:.6f}\n")
        f.write(f"initial_battery_j={x_star[0, 2]:.6f}\n")
        f.write(f"final_battery_j={x_star[-1, 2]:.6f}\n")
        f.write(f"energy_used_power_integral_j={energy_used:.6f}\n")
        f.write(f"nonlinear_dynamics_defect={defect:.6e}\n")
        f.write(f"max_power_w={float(np.max(powers)):.6f}\n")
        f.write(f"mean_power_w={float(np.mean(powers)):.6f}\n")
        f.write("\nChunks\n")
        for i, scp in enumerate(scps, start=1):
            accepted = sum(1 for entry in scp.history if entry.get("accepted"))
            rejected = sum(1 for entry in scp.history if not entry.get("accepted", True))
            f.write(
                f"chunk={i}, N={scp.N}, dt={scp.final_time_s / scp.N:.6f}, "
                f"final_time_s={scp.final_time_s:.6f}, "
                f"Tf_min={scp.Tf_min:.6f}, Tf_max={scp.Tf_max:.6f}, "
                f"iterations={len(scp.history)}, accepted={accepted}, rejected={rejected}\n"
            )

    fields = [
        "global_iteration",
        "chunk",
        "local_iteration",
        "accepted",
        "objective",
        "candidate_objective",
        "merit",
        "candidate_merit",
        "delta_merit",
        "defect",
        "candidate_defect",
        "virtual",
        "candidate_virtual",
        "final_time_s",
        "candidate_final_time_s",
        "dt",
        "candidate_dt",
        "rho_state_max",
        "rho_u",
        "rho_Tf",
    ]
    with history_path.open("w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        global_iter = 0
        for chunk_id, scp in enumerate(scps, start=1):
            for local_iter, entry in enumerate(scp.history):
                row = {
                    "global_iteration": global_iter,
                    "chunk": chunk_id,
                    "local_iteration": local_iter,
                    **entry,
                }
                f.write(",".join(str(row.get(field, "")) for field in fields) + "\n")
                global_iter += 1


def plot_results(
    terrain,
    raw_path,
    smooth_path,
    scp: SCP,
    x_nominal: np.ndarray,
    u_nominal: np.ndarray,
    x_star: np.ndarray | None,
    u_star: np.ndarray | None,
    dt_values: np.ndarray,
    output_path: Path,
) -> None:
    raw = np.asarray(raw_path, dtype=float)
    smooth = np.asarray(smooth_path, dtype=float)
    x_plot = x_star if x_star is not None else x_nominal
    u_plot = u_star if u_star is not None else u_nominal
    t_control = np.concatenate(([0.0], np.cumsum(dt_values[:-1])))
    t_state = np.concatenate(([0.0], np.cumsum(dt_values)))
    power = power_trace(scp, x_plot, u_plot, dt_values)
    x_rollout = nonlinear_rollout(scp, x_plot[0], u_plot, dt_values)

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

    axes[0, 1].plot(t_state, x_plot[:, 2], color="tab:green", lw=2.0, label="SCP state")
    axes[0, 1].plot(t_state, x_rollout[:, 2], color="tab:olive", ls="--", lw=1.5, label="nonlinear rollout")
    axes[0, 1].axhline(scp.E_min, color="black", ls="--", lw=1.0, label="E_min")
    axes[0, 1].set_title("Battery Energy")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("E [J]")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].plot(t_state, x_plot[:, 3], color="tab:purple", lw=2.0, label="state speed")
    axes[1, 0].step(t_control, u_plot[:, 0], where="post", lw=1.4, color="tab:blue", alpha=0.75, label="speed command")
    axes[1, 0].axhline(scp.v_max, color="black", ls="--", lw=1.0, label="v_max")
    axes[1, 0].set_title("Speed")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("speed [m/s]")
    axes[1, 0].grid(True)
    axes[1, 0].legend()

    axes[1, 1].plot(t_state, x_plot[:, 4], color="tab:brown", lw=2.0, label="state heading")
    axes[1, 1].step(t_control, u_plot[:, 1], where="post", lw=1.4, color="tab:orange", alpha=0.75, label="heading command")
    axes[1, 1].set_title("Heading")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("heading [rad]")
    axes[1, 1].grid(True)
    axes[1, 1].legend()

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
        merits = [entry.get("merit", np.nan) for entry in scp.history if "objective" in entry]
        axes[2, 1].plot(iterations, objectives, marker="o", label="true objective")
        axes[2, 1].plot(iterations, merits, marker="x", label="merit")
        rejected = [
            entry
            for entry in scp.history
            if "objective" in entry and not entry.get("accepted", True)
        ]
        if rejected:
            axes[2, 1].scatter(
                [entry["iteration"] for entry in rejected],
                [entry.get("candidate_merit", entry["merit"]) for entry in rejected],
                marker="v",
                color="tab:red",
                label="rejected",
            )
        axes[2, 1].set_xlabel("iteration")
        axes[2, 1].set_ylabel("value")
        axes[2, 1].grid(True)
        axes[2, 1].set_title("SCP Iteration History")
        axes[2, 1].legend()
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
    x_star, u_star, dt_values, scps = solve_chunked_path(high_level_path)
    x_init, u_init = x_star, u_star
    scp = scps[0]
    scp.final_time_s = sum(local_scp.final_time_s for local_scp in scps)
    scp.N = u_star.shape[0]
    history = []
    global_iter = 0
    for chunk_id, local_scp in enumerate(scps, start=1):
        for local_iter, entry in enumerate(local_scp.history):
            if "objective" not in entry:
                continue
            row = dict(entry)
            row["chunk"] = chunk_id
            row["local_iteration"] = local_iter
            row["iteration"] = global_iter
            history.append(row)
            global_iter += 1
    scp.history = history

    plot_results(terrain, raw_path, high_level_path, scp, x_init, u_init, x_star, u_star, dt_values, OUTPUT_PATH)
    x_plot = x_star
    u_plot = u_star

    print(f"Saved planner + SCP plot: {OUTPUT_PATH}")
    print(f"Planner raw waypoints: {len(raw_path)}")
    print(f"SCP high-level path waypoints: {len(high_level_path)}")
    print(f"SCP chunks: {len(scps)}")
    print(f"SCP total horizon steps: {u_star.shape[0]}")
    print(f"SCP final time: {scp.final_time_s:.1f} s")
    print(f"Trajectory length: {path_length(x_plot):.1f} m")
    print(f"Final battery: {x_plot[-1, 2]:.2f} J")
    print(f"Nonlinear dynamics defect: {dynamics_defect(scp, x_plot, u_plot, dt_values):.3e}")
    write_run_logs(LOG_PATH, HISTORY_PATH, raw_path, high_level_path, x_star, u_star, dt_values, scps)
    print(f"Saved run log: {LOG_PATH}")
    print(f"Saved history log: {HISTORY_PATH}")

if __name__ == "__main__":
    main()
