from __future__ import annotations

import numpy as np

from scp import SCP, cp


def assert_close(actual, expected, name, atol=1.0e-8):
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{name} mismatch:\nactual={actual}\nexpected={expected}")


def build_test_scp():
    return SCP(
        dt=0.1,
        final_time_s=1.0,
        ct=2.0,
        ce=1.0,
        high_level_path=[
            [0.0, 0.0],
            [0.2, 0.0],
            [0.4, 0.1],
        ],
    )


def test_model_limits_are_exposed():
    scp = build_test_scp()
    assert scp.v_max == scp.model.max_speed_mps
    assert scp.E_min == scp.model.min_battery_charge_j
    assert scp.E_max == scp.model.max_battery_charge_j
    assert scp.P_cons_max == scp.model.max_power_consumption_w


def test_horizon_and_state_augmentation():
    scp = build_test_scp()
    assert scp.horizon_steps == 10
    assert_close(
        scp.nominal_state_from_waypoint([1.0, 2.0]),
        [1.0, 2.0, 20_000.0, 0.0, 0.0, 0.0],
        "2D nominal state",
    )
    assert_close(
        scp.nominal_state_from_waypoint([1.0, 2.0, 3.0]),
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        "3D nominal state",
    )


def test_nominal_trajectory_shapes():
    scp = build_test_scp()
    x_bar, u_bar = scp.initialize_nominal_trajectory()
    assert x_bar.shape == (scp.n_states, scp.horizon_steps + 1)
    assert u_bar.shape == (scp.n_inputs, scp.horizon_steps)
    assert scp.nu_k.shape == (scp.n_states, scp.horizon_steps)
    assert_close(x_bar[:2, 0], [0.0, 0.0], "initial position")
    assert_close(x_bar[:2, -1], [0.4, 0.1], "terminal nominal position")


def test_linearizations_match_nominal_value():
    scp = build_test_scp()
    x_bar, u_bar = scp.initialize_nominal_trajectory()
    x0 = x_bar[:, 0]
    u0 = u_bar[:, 0]

    A, B, c = scp.linearize_dynamics(x0, u0)
    f0 = scp.model.F(x0, u0, scp.dt)
    assert A.shape == (scp.n_states, scp.n_states)
    assert B.shape == (scp.n_states, scp.n_inputs)
    assert c.shape == (scp.n_states,)
    assert_close(A @ x0 + B @ u0 + c, f0, "linearized dynamics at nominal")

    p_x, p_u, p_c = scp.linearize_power(x0, u0)
    p0 = scp.model.P(x0, u0, scp.dt)
    assert p_x.shape == (scp.n_states,)
    assert p_u.shape == (scp.n_inputs,)
    assert_close(p_x @ x0 + p_u @ u0 + p_c, p0, "linearized power at nominal")


def test_nonlinear_helpers():
    scp = build_test_scp()
    x_bar, u_bar = scp.initialize_nominal_trajectory()
    rollout = scp.nonlinear_rollout(x_bar[:, 0], u_bar)
    assert rollout.shape == x_bar.shape
    assert_close(scp.nonlinear_dynamics_defect(rollout, u_bar), 0.0, "rollout defect")

    expected_cost = scp.ct * scp.final_time_s
    for k in range(scp.horizon_steps):
        expected_cost += scp.ce * scp.model.P(x_bar[:, k], u_bar[:, k], scp.dt) * scp.dt
    assert_close(scp.nonlinear_objective(x_bar, u_bar), expected_cost, "nonlinear objective")


def test_convex_subproblem_if_cvxpy_is_available():
    if cp is None:
        print("Skipping convex subproblem solve because cvxpy is not installed.")
        return

    scp = build_test_scp()
    scp.initialize_nominal_trajectory()
    x_star, u_star, nu_star = scp.convex_ocp()
    assert x_star.shape == (scp.n_states, scp.horizon_steps + 1)
    assert u_star.shape == (scp.n_inputs, scp.horizon_steps)
    assert nu_star.shape == (scp.n_states, scp.horizon_steps)


def main():
    test_model_limits_are_exposed()
    test_horizon_and_state_augmentation()
    test_nominal_trajectory_shapes()
    test_linearizations_match_nominal_value()
    test_nonlinear_helpers()
    test_convex_subproblem_if_cvxpy_is_available()
    print("All SCP tests passed.")


if __name__ == "__main__":
    main()
