"""Battery-depletion example for the rover dynamic model.

Set MODE to "straight" or "circle". The rover executes that command until
the stored battery energy is depleted by net power draw:

    battery_dot = power_generation_w - power_consumption_w.

This is a simple open-loop simulation, not a planner. It repeatedly:

1. Converts the commanded projected-frame velocity into SE(2) state rates.
2. Computes acceleration from the constant-speed circular-motion equations.
3. Uses RoverModel.power_breakdown(...) to compute instantaneous load.
4. Calls RoverModel.step(dt), which integrates pose and battery energy with RK4.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rover_dynamic_model import RoverModel


MODE = "circle"
PLOT_PATH = "rover_battery_simulation.png"


def command_for_mode(mode: str) -> np.ndarray:
    """Return a constant projected-frame command [v_Bpi, omega_Bpi]."""

    if mode == "straight":
        # Positive forward speed with zero yaw rate gives a straight path.
        return np.array([0.45, 0.0])
    if mode == "circle":
        # Constant forward speed and constant yaw rate gives a circular arc.
        return np.array([0.45, 0.18])
    raise ValueError(f"unknown mode {mode!r}; expected 'straight' or 'circle'")


def plot_simulation(
    times: np.ndarray,
    states: np.ndarray,
    battery_trace: np.ndarray,
    power_trace: np.ndarray,
    power_generation_w: float,
    output_path: str,
) -> None:
    """Save pose, battery, and power traces from the simulation."""

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    axes[0, 0].plot(states[:, 0], states[:, 1], linewidth=2.0)
    axes[0, 0].scatter(states[0, 0], states[0, 1], label="start", s=35)
    axes[0, 0].scatter(states[-1, 0], states[-1, 1], label="end", s=35)
    axes[0, 0].set_title("XY Path")
    axes[0, 0].set_xlabel("x_G [m]")
    axes[0, 0].set_ylabel("y_G [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(times, states[:, 0], label="x_G")
    axes[0, 1].plot(times, states[:, 1], label="y_G")
    axes[0, 1].plot(times, states[:, 2], label="psi_G")
    axes[0, 1].set_title("Pose Evolution")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("state")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(times, battery_trace, color="tab:green", linewidth=2.0)
    axes[1, 0].set_title("Battery Energy")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("energy [J]")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, power_trace, label="consumption", color="tab:red")
    axes[1, 1].axhline(power_generation_w, label="generation", color="tab:blue", linestyle="--")
    axes[1, 1].set_title("Power")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("power [W]")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(f"Rover Battery Simulation ({MODE})")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    # Integration settings. max_time is a guard so the script still exits if
    # the selected power inputs are high enough to sustain the rover forever.
    dt = 0.1
    max_time = 2_000.0
    control = command_for_mode(MODE)

    # These parameters are intentionally chosen so load is above constant
    # generation; otherwise the battery would not run out in this example.
    model = RoverModel(
        battery_charge_j=20_000.0,
        power_generation_w=65.0,
        v_command_mps=control[0],
        omega_command_radps=control[1],
        phi=np.deg2rad(4.0),
        xi=np.deg2rad(3.0),
    )

    # Store traces for summary statistics at the end.
    times = []
    states = []
    battery_trace = []
    power_trace = []

    steps = int(max_time / dt)
    for step in range(steps):
        times.append(model.time_s)
        states.append(model.pose.copy())
        battery_trace.append(model.battery_charge_j)
        power_trace.append(model.power_consumption_w)

        if model.battery_charge_j <= 0.0:
            break

        model.step(dt)
        if model.battery_charge_j <= 0.0:
            times.append(model.time_s)
            states.append(model.pose.copy())
            battery_trace.append(model.battery_charge_j)
            power_trace.append(model.power_consumption_w)
            break

    times = np.array(times)
    states = np.array(states)
    battery_trace = np.array(battery_trace)
    power_trace = np.array(power_trace)

    # Approximate path length from logged positions.
    distance = float(np.sum(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1))) if len(states) > 1 else 0.0
    print(f"Mode: {MODE}")
    print(f"Run time: {times[-1]:.1f} s")
    print(f"Distance traveled: {distance:.2f} m")
    print(f"Final pose [x, y, psi]: {states[-1]}")
    print(f"Battery remaining: {model.battery_charge_j:.2f} J")
    print(f"Average total power: {np.mean(power_trace):.2f} W")
    print(f"Peak total power: {np.max(power_trace):.2f} W")
    print(f"Generation power: {model.power_generation_w:.2f} W")
    print(f"Initial battery: {model.battery_charge_j if len(battery_trace) == 0 else battery_trace[0]:.2f} J")
    print(f"Final logged battery: {battery_trace[-1]:.2f} J")

    plot_simulation(times, states, battery_trace, power_trace, model.power_generation_w, PLOT_PATH)
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
