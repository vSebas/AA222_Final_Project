"""Battery-depletion example for the rover dynamic model.

Set MODE to "straight" or "circle". The rover executes that command until
the stored battery energy is depleted by net power draw:

    battery_dot = p_available - p_total.

This is a simple open-loop simulation, not a planner. It repeatedly:

1. Converts the commanded projected-frame velocity into SE(2) state rates.
2. Estimates acceleration from the previous step.
3. Uses RoverModel.power_breakdown(...) to compute instantaneous load.
4. Drains the battery only when load exceeds RTG + solar input.
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
    available_power: float,
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

    axes[1, 1].plot(times, power_trace, label="total load", color="tab:red")
    axes[1, 1].axhline(available_power, label="RTG + solar", color="tab:blue", linestyle="--")
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

    # Stored battery energy is separate from hybrid generation. The rover can
    # keep driving while battery_energy_j > 0 even if p_total > p_available.
    battery_capacity_j = 20_000.0
    battery_energy_j = battery_capacity_j

    # These parameters are intentionally chosen so p_total is usually above
    # p_available; otherwise the battery would not run out in this example.
    model = RoverModel(
        mass=84.0,
        inertia_z=7.111679166666666,
        gravity=1.62,
        # Flat no-grade EMRS breadboard-scaled lunar resistance from the NASA
        # terramechanics estimate: compression + rolling + bulldozing.
        c0=63.54416662174218,
        p_base=100.0,
        p_rtg=45.0,
        p_solar=20.0,
        phi=np.deg2rad(4.0),
        xi=np.deg2rad(3.0),
    )

    control = command_for_mode(MODE)

    # Projected SE(2) state: [x_G, y_G, psi_G].
    state = np.array([0.0, 0.0, 0.0])

    # Used for finite-difference acceleration and angular acceleration. The
    # first step includes the acceleration needed to move from rest to command.
    previous_velocity = np.array([0.0, 0.0])
    previous_omega = 0.0

    # Store traces for summary statistics at the end.
    times = []
    states = []
    battery_trace = []
    power_trace = []

    steps = int(max_time / dt)
    for step in range(steps):
        time = step * dt

        # The dynamic model power equations need x_dot, y_dot, x_ddot, y_ddot,
        # omega, and omega_dot in the projected frame.
        derivative = model.se2_kinematics(state, control)
        velocity = derivative[:2]
        acceleration = (velocity - previous_velocity) / dt
        omega = derivative[2]
        omega_dot = (omega - previous_omega) / dt

        powers = model.power_breakdown(
            x_dot=velocity[0],
            y_dot=velocity[1],
            x_ddot=acceleration[0],
            y_ddot=acceleration[1],
            psi=state[2],
            omega=omega,
            omega_dot=omega_dot,
        )
        total_power = float(powers["p_total"])

        # Positive net_battery_power means the rover needs extra power from the
        # battery. Negative values mean RTG + solar fully cover the load, so the
        # battery is not drained in this simple example.
        net_battery_power = total_power - model.p_available

        times.append(time)
        states.append(state.copy())
        battery_trace.append(battery_energy_j)
        power_trace.append(total_power)

        battery_energy_j -= max(net_battery_power, 0.0) * dt
        if battery_energy_j <= 0.0:
            break

        # Advance the rover pose after logging the current-step state.
        state = model.euler_step(state, control, dt)
        previous_velocity = velocity
        previous_omega = omega

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
    print(f"Battery remaining: {max(battery_energy_j, 0.0):.2f} J")
    print(f"Average total power: {np.mean(power_trace):.2f} W")
    print(f"Peak total power: {np.max(power_trace):.2f} W")
    print(f"Hybrid input power: {model.p_available:.2f} W")
    print(f"Initial battery: {battery_capacity_j:.2f} J")
    print(f"Final logged battery: {battery_trace[-1]:.2f} J")

    plot_simulation(times, states, battery_trace, power_trace, model.p_available, PLOT_PATH)
    print(f"Saved plot: {PLOT_PATH}")


if __name__ == "__main__":
    main()
