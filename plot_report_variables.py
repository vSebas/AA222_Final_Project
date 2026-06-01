from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_PATH = PROJECT_DIR / "planner_scp_trajectory.npz"
HISTORY_PATH = PROJECT_DIR / "planner_scp_history.csv"
OUT_DIR = PROJECT_DIR / "report_plots"


def style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(path)


def main() -> None:
    data = np.load(CACHE_PATH)
    raw = data["raw_path"]
    high = data["high_level_path"]
    x = data["x_star"]
    u = data["u_star"]
    t_state_h = data["t_state"] / 3600.0
    t_control_h = data["t_control"] / 3600.0
    power_w = data["power_w"]
    battery_mj = data["battery_state_j"] / 1.0e6
    rollout_battery_mj = data["battery_rollout_j"] / 1.0e6
    cumulative_energy_mj = data["cumulative_energy_j"] / 1.0e6

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(raw[:, 0] / 1e3, raw[:, 1] / 1e3, color="0.70", lw=1.0, label="A* raw")
    ax.plot(high[:, 0] / 1e3, high[:, 1] / 1e3, color="tab:green", lw=2.0, label="smoothed path")
    ax.plot(x[:, 0] / 1e3, x[:, 1] / 1e3, color="tab:blue", lw=2.0, label="SCP trajectory")
    ax.axis("equal")
    style_axes(ax, "east position [km]", "north position [km]", "Planner Path and Rover Trajectory")
    ax.legend()
    save(fig, "trajectory.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(t_state_h, battery_mj, color="tab:green", lw=2.2, label="SCP state")
    ax.plot(t_state_h, rollout_battery_mj, color="tab:olive", ls="--", lw=1.6, label="nonlinear rollout")
    style_axes(ax, "mission time [h]", "battery energy remaining [MJ]", "Battery Energy Remaining")
    ax.legend()
    save(fig, "battery_energy_hours.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(t_state_h, x[:, 3], color="tab:purple", lw=2.0, label="state speed")
    ax.step(t_control_h, u[:, 0], where="post", color="tab:blue", lw=1.3, alpha=0.8, label="speed command")
    style_axes(ax, "mission time [h]", "speed [m/s]", "Rover Speed")
    ax.legend()
    save(fig, "speed_hours.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.step(t_control_h, power_w, where="post", color="tab:red", lw=2.0)
    style_axes(ax, "mission time [h]", "gross power demand [W]", "Gross Power Demand")
    save(fig, "power_hours.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(t_state_h, cumulative_energy_mj, color="tab:cyan", lw=2.2)
    style_axes(ax, "mission time [h]", "gross energy consumed [MJ]", "Cumulative Gross Energy Consumed")
    save(fig, "cumulative_energy_hours.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(t_state_h, x[:, 4], color="tab:brown", lw=2.0, label="state heading")
    ax.step(t_control_h, u[:, 1], where="post", color="tab:orange", lw=1.2, alpha=0.75, label="heading command")
    style_axes(ax, "mission time [h]", "heading [rad]", "Heading Profile")
    ax.legend()
    save(fig, "heading_hours.png")

    if HISTORY_PATH.exists():
        with HISTORY_PATH.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        by_chunk: dict[int, list[tuple[int, float]]] = {}
        for row in rows:
            chunk = int(row["chunk"])
            local_iter = int(row["local_iteration"])
            candidate_obj = float(row["candidate_objective"])
            by_chunk.setdefault(chunk, []).append((local_iter, candidate_obj))

        fig, ax = plt.subplots(figsize=(6.8, 4.3))
        for chunk, values in sorted(by_chunk.items()):
            values.sort()
            iters = np.array([v[0] for v in values], dtype=float)
            objs = np.array([v[1] for v in values], dtype=float)
            if objs[0] == 0.0:
                continue
            ax.plot(iters, objs / objs[0], lw=1.1, alpha=0.55, label=f"chunk {chunk}")

        style_axes(
            ax,
            "local SCP iteration",
            "candidate objective / initial candidate objective",
            "Normalized Candidate Objective History",
        )
        ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
        ax.legend(ncol=2, fontsize=7)
        save(fig, "objective_history.png")


if __name__ == "__main__":
    main()
