#!/usr/bin/env python3
"""
emrs_resistive_forces.py

Estimate lunar rover resistive forces using an EMRS-inspired rover geometry and
NASA LTV terramechanics formulas, then optionally fit Hu-style persistent
resistance coefficients:

    P_res(v) = (C0 + C1 |v| + C2 v^2) v

Default preset:
    emrs_breadboard_scaled_lunar

Rationale:
    - EMRS breadboard mass/body/wheel spacing are directly reported in the
      EMRS breadboard field-test paper: mass 84 kg, body 890 x 230 x 370 mm,
      wheel spacing 980 x 830 mm. The paper states the breadboard is scaled
      1:2 from the flight-model preliminary design.
    - The main EMRS design paper reports flight-model wheel dimensions
      D = 612 mm, b = 216 mm. For the 1:2 breadboard preset, the script uses
      D = 306 mm, b = 108 mm.
    - Soil/resistance formulas are from NASA's LTV terramechanics white paper.

Outputs:
    1. CSV table over speed/slope samples with resistance components.
    2. JSON summary with rover/soil params and Hu-style coefficients.

Important modeling note:
    The terramechanics resistance terms used here are mostly quasi-static and
    therefore mainly identify C0 in Hu's polynomial force approximation. C1 and
    C2 require measured speed-dependent power data, motor maps, slip models, or
    user-provided viscous/quadratic priors.

Example:
    python emrs_resistive_forces.py \
        --preset emrs_breadboard_scaled_lunar \
        --speed-max 0.15 \
        --slope-deg-list 0 5 10 15 \
        --output-prefix emrs_lunar

    python emrs_resistive_forces.py \
        --preset emrs_flight_geometry_lunar \
        --mass-kg 250 \
        --speed-max 0.5 \
        --slope-deg-list 0 10 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class LunarSoilParams:
    """Typical lunar soil parameters from NASA LTV terramechanics white paper."""

    n: float = 1.0                      # exponent of sinkage [-]
    kc: float = 1400.0                  # cohesive modulus [N/m^2]
    kphi: float = 820_000.0             # frictional modulus [N/m^3]
    phi_deg: float = 35.0               # internal friction angle [deg], typical 30-40
    cohesion: float = 170.0             # c [N/m^2]
    gamma: float = 2470.0               # soil weight density [N/m^3]
    K_slip: float = 0.018               # soil slip coefficient [m]

    # NASA table defaults; can be recomputed from phi.
    Nq: float = 32.23
    Nc: float = 48.09
    Ngamma: float = 33.27
    Kc_deformation: float = 33.37
    Kgamma_deformation: float = 72.77


@dataclass
class RoverParams:
    """Reduced rover geometry needed by the resistance model."""

    name: str
    mass_kg: float
    body_length_m: float
    body_width_m: float
    body_height_m: float
    wheelbase_m: float
    track_width_m: float
    wheel_diameter_m: float
    wheel_width_m: float
    n_wheels: int = 4
    n_front_wheels: int = 2
    n_drive_wheels: int = 4
    rolling_friction_coeff: float = 0.03
    lunar_gravity_mps2: float = 1.62
    wheel_mass_kg: float = 7.0
    max_wheel_torque_Nm: float = 80.0
    max_speed_mps: float = 0.8333333333  # 3 km/h
    notes: str = ""


@dataclass
class ResistanceComponents:
    slope_deg: float
    speed_mps: float
    normal_total_N: float
    normal_per_wheel_N: float
    sinkage_m: float
    contact_length_m: float
    contact_area_per_wheel_m2: float
    compression_per_wheel_N: float
    compression_total_N: float
    rolling_total_N: float
    bulldozing_per_front_wheel_N: float
    bulldozing_total_N: float
    gravity_grade_N: float
    viscous_speed_N: float
    quadratic_speed_N: float
    total_no_grade_N: float
    total_with_grade_N: float
    power_no_grade_W: float
    power_with_grade_W: float
    resistance_torque_per_drive_wheel_Nm: float
    max_motor_force_total_N: float
    smooth_traction_per_wheel_N: float
    smooth_traction_total_N: float
    grouser_traction_per_wheel_N: float
    grouser_traction_total_N: float
    traction_margin_smooth_N: float
    traction_margin_grouser_N: float


# -----------------------------------------------------------------------------
# Presets
# -----------------------------------------------------------------------------

def preset_rover(name: str, mass_override: Optional[float] = None) -> RoverParams:
    """Return an EMRS-inspired rover preset.

    Presets:
      - emrs_breadboard_scaled_lunar:
          Uses reported breadboard mass/body/spacing and 1:2 scaled EMRS wheels.
          This is the most internally consistent default because the mass is known.
      - emrs_flight_geometry_lunar:
          Uses flight-model geometry/wheel dimensions. Flight mass is not clearly
          specified in the papers, so --mass-kg should be supplied.
    """
    key = name.lower().strip()

    if key == "emrs_breadboard_scaled_lunar":
        return RoverParams(
            name="emrs_breadboard_scaled_lunar",
            mass_kg=84.0 if mass_override is None else mass_override,
            body_length_m=0.890,
            body_width_m=0.230,
            body_height_m=0.370,
            wheelbase_m=0.980,
            track_width_m=0.830,
            # EMRS FM wheel D,b are 0.612,0.216 m; breadboard is reported as 1:2 scale.
            wheel_diameter_m=0.306,
            wheel_width_m=0.108,
            wheel_mass_kg=7.0 / 8.0,  # rough geometric scale placeholder; not used in resistance.
            max_wheel_torque_Nm=80.0 / 8.0,
            max_speed_mps=0.15,
            notes=(
                "EMRS breadboard: mass/body/spacing from 2024 breadboard paper. "
                "Wheel D,b are 1:2 scaled from EMRS flight-model wheel dimensions. "
                "Torque is also scaled heuristically; override if needed."
            ),
        )

    if key == "emrs_flight_geometry_lunar":
        m = 84.0 if mass_override is None else mass_override
        notes = (
            "EMRS flight-model geometry/wheels. Flight mass was not cleanly available "
            "from the papers used here; default mass=84 kg is only a placeholder. "
            "Pass --mass-kg for a proper flight-like study."
        )
        return RoverParams(
            name="emrs_flight_geometry_lunar",
            mass_kg=m,
            body_length_m=2.366,
            body_width_m=1.525,
            body_height_m=1.000,
            wheelbase_m=1.775,
            track_width_m=1.284,
            wheel_diameter_m=0.612,
            wheel_width_m=0.216,
            wheel_mass_kg=7.0,
            max_wheel_torque_Nm=80.0,
            max_speed_mps=0.8333333333,
            notes=notes,
        )

    raise ValueError(
        f"Unknown preset '{name}'. Use 'emrs_breadboard_scaled_lunar' or "
        "'emrs_flight_geometry_lunar'."
    )


# -----------------------------------------------------------------------------
# Utility math
# -----------------------------------------------------------------------------

def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def positive_part(x: float) -> float:
    return max(0.0, x)


def recompute_soil_factors(soil: LunarSoilParams) -> LunarSoilParams:
    """Recompute Nq, Nc, Ngamma, Kc, Kgamma from phi.

    Uses the formulas listed in the NASA LTV terramechanics white paper.
    """
    phi = deg2rad(soil.phi_deg)
    tan_phi = math.tan(phi)
    if abs(tan_phi) < 1e-12:
        raise ValueError("phi is too close to zero for bearing-factor formulas")

    Nq = math.exp((1.5 * math.pi - phi) * tan_phi) / (
        2.0 * math.cos(math.pi / 4.0 + phi / 2.0) ** 2
    )
    Nc = (Nq - 1.0) / tan_phi
    Ngamma = 2.0 * (Nq + 1.0) * tan_phi / (1.0 + 0.4 * math.sin(4.0 * phi))
    Kc = (Nc - tan_phi) * math.cos(phi) ** 2
    Kgamma = ((2.0 * Ngamma / tan_phi) + 1.0) * math.cos(phi) ** 2

    soil.Nq = Nq
    soil.Nc = Nc
    soil.Ngamma = Ngamma
    soil.Kc_deformation = Kc
    soil.Kgamma_deformation = Kgamma
    return soil


# -----------------------------------------------------------------------------
# NASA LTV terramechanics formulas
# -----------------------------------------------------------------------------

def wheel_sinkage_m(W_per_wheel_N: float, rover: RoverParams, soil: LunarSoilParams) -> float:
    """Wheel sinkage, NASA LTV Eq. (8).

    W_per_wheel_N is the terrain-normal load carried by one wheel.
    """
    if W_per_wheel_N <= 0.0:
        return 0.0
    n = soil.n
    D = rover.wheel_diameter_m
    b = rover.wheel_width_m
    denom = (soil.kc + b * soil.kphi) * math.sqrt(D)
    arg = (3.0 / (3.0 - n)) * W_per_wheel_N / denom
    return positive_part(arg) ** (2.0 / (2.0 * n + 1.0))


def compression_resistance_per_wheel_N(
    W_per_wheel_N: float, rover: RoverParams, soil: LunarSoilParams
) -> float:
    """Compression resistance for one wheel, NASA LTV Eq. (9).

    For n=1 this is equivalent to NASA's simplified Eq. (10).
    """
    if W_per_wheel_N <= 0.0:
        return 0.0
    n = soil.n
    D = rover.wheel_diameter_m
    b = rover.wheel_width_m
    term1 = 1.0 / (n + 1.0)
    term2 = ((3.0 / (3.0 - n)) * W_per_wheel_N / math.sqrt(D)) ** (
        2.0 * (n + 1.0) / (2.0 * n + 1.0)
    )
    term3 = (1.0 / (soil.kc + b * soil.kphi)) ** (1.0 / (2.0 * n + 1.0))
    return term1 * term2 * term3


def rolling_resistance_total_N(total_normal_N: float, rover: RoverParams) -> float:
    """Rolling/internal resistance, NASA LTV Eq. (15): Rr = Wv cf."""
    return positive_part(total_normal_N) * rover.rolling_friction_coeff


def gravity_grade_force_N(rover: RoverParams, slope_deg: float) -> float:
    """Gravitational grade force, NASA LTV Eq. (16): Rg = Wv sin(theta).

    Positive is uphill resistance. Negative is downhill assistance.
    """
    return rover.mass_kg * rover.lunar_gravity_mps2 * math.sin(deg2rad(slope_deg))


def contact_length_m(sinkage_m: float, rover: RoverParams) -> float:
    """Wheel-soil contact length, NASA LTV Eq. (24)."""
    z = max(0.0, sinkage_m)
    D = rover.wheel_diameter_m
    if z <= 0.0:
        return 0.0
    arg = clamp(1.0 - 2.0 * z / D, -1.0, 1.0)
    return 0.5 * D * math.acos(arg)


def bulldozing_resistance_per_front_wheel_N(
    sinkage_m: float, rover: RoverParams, soil: LunarSoilParams
) -> float:
    """Bulldozing resistance for one front wheel, NASA LTV Eq. (17)-(21)."""
    z = max(0.0, sinkage_m)
    if z <= 0.0:
        return 0.0

    D = rover.wheel_diameter_m
    b = rover.wheel_width_m
    phi = deg2rad(soil.phi_deg)

    alpha_arg = clamp(1.0 - 2.0 * z / D, -1.0, 1.0)
    alpha = math.acos(alpha_arg)
    if abs(math.sin(alpha)) < 1e-12 or abs(math.cos(phi)) < 1e-12:
        return 0.0

    l0 = z * math.tan(math.pi / 4.0 - phi / 2.0) ** 2

    term_a = (b * math.sin(alpha + phi)) / (2.0 * math.sin(alpha) * math.cos(phi))
    term_b = 2.0 * z * soil.cohesion * soil.Kc_deformation + soil.gamma * z**2 * soil.Kgamma_deformation
    term_c = (l0**3 / 3.0) * soil.gamma * (math.pi / 2.0 - phi)
    term_d = soil.cohesion * l0**2 * (1.0 + math.tan(math.pi / 4.0 + phi / 2.0))
    return term_a * term_b + term_c + term_d


def smooth_traction_per_wheel_N(
    W_per_wheel_N: float,
    contact_area_m2: float,
    contact_length: float,
    slip: float,
    soil: LunarSoilParams,
) -> float:
    """Smooth-wheel tractive force, NASA LTV Eq. (23).

    slip s should be in (0, 1] for driving slip. At s=0, this returns the ideal
    static shear capacity Ac + W tan(phi).
    """
    phi = deg2rad(soil.phi_deg)
    ideal = contact_area_m2 * soil.cohesion + W_per_wheel_N * math.tan(phi)
    s = max(0.0, slip)
    l = max(0.0, contact_length)
    if s <= 1e-9 or l <= 1e-9:
        return ideal
    factor = 1.0 - (soil.K_slip / (s * l)) * (1.0 - math.exp(-s * l / soil.K_slip))
    return ideal * clamp(factor, 0.0, 1.0)


def grouser_traction_per_wheel_N(
    W_per_wheel_N: float,
    contact_area_m2: float,
    contact_length: float,
    slip: float,
    soil: LunarSoilParams,
    rover: RoverParams,
    grouser_height_m: float,
    n_grousers_total: int,
) -> float:
    """Grouser-wheel tractive force, NASA LTV Eq. (26)-(27)."""
    if grouser_height_m <= 0.0 or n_grousers_total <= 0:
        return smooth_traction_per_wheel_N(W_per_wheel_N, contact_area_m2, contact_length, slip, soil)

    phi = deg2rad(soil.phi_deg)
    b = rover.wheel_width_m
    D = rover.wheel_diameter_m
    hg = grouser_height_m
    l = max(0.0, contact_length)
    A = max(0.0, contact_area_m2)
    s = max(0.0, slip)

    Ng_contact = n_grousers_total * l / (math.pi * D)
    cohesion_term = A * soil.cohesion * (1.0 + (2.0 * hg / b) * Ng_contact)
    friction_term = W_per_wheel_N * math.tan(phi) * (1.0 + 0.64 * (hg / b) * math.atan(b / hg))
    ideal = cohesion_term + friction_term

    if s <= 1e-9 or l <= 1e-9:
        return ideal
    factor = 1.0 - (soil.K_slip / (s * l)) * (1.0 - math.exp(-s * l / soil.K_slip))
    return ideal * clamp(factor, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Combined resistance model
# -----------------------------------------------------------------------------

def resistance_components(
    rover: RoverParams,
    soil: LunarSoilParams,
    speed_mps: float,
    slope_deg: float,
    include_bulldozing: bool = True,
    include_grade_in_total: bool = True,
    viscous_force_per_mps: float = 0.0,
    quadratic_force_per_mps2: float = 0.0,
    slip: float = 0.2,
    grouser_height_m: float = 0.0,
    n_grousers_total: int = 0,
) -> ResistanceComponents:
    """Compute all resistance components for one speed/slope sample."""
    slope_rad = deg2rad(slope_deg)
    total_normal = rover.mass_kg * rover.lunar_gravity_mps2 * math.cos(slope_rad)
    total_normal = max(0.0, total_normal)
    W_per = total_normal / rover.n_wheels

    z = wheel_sinkage_m(W_per, rover, soil)
    l_contact = contact_length_m(z, rover)
    A_contact = rover.wheel_width_m * l_contact

    Rc_per = compression_resistance_per_wheel_N(W_per, rover, soil)
    Rc_total = rover.n_wheels * Rc_per

    Rr_total = rolling_resistance_total_N(total_normal, rover)

    Rb_per = bulldozing_resistance_per_front_wheel_N(z, rover, soil) if include_bulldozing else 0.0
    Rb_total = rover.n_front_wheels * Rb_per

    Rg = gravity_grade_force_N(rover, slope_deg)

    F_visc = viscous_force_per_mps * abs(speed_mps)
    F_quad = quadratic_force_per_mps2 * speed_mps**2

    F_no_grade = Rc_total + Rr_total + Rb_total + F_visc + F_quad
    F_with_grade = F_no_grade + (Rg if include_grade_in_total else 0.0)

    # Do not let total driving resistance become negative on steep downhill; braking/regeneration
    # should be modeled separately if needed.
    F_with_grade_for_power = max(0.0, F_with_grade)
    P_no_grade = max(0.0, F_no_grade * speed_mps)
    P_with_grade = max(0.0, F_with_grade_for_power * speed_mps)

    r_wheel = 0.5 * rover.wheel_diameter_m
    torque_per_drive_wheel = (F_with_grade_for_power / max(1, rover.n_drive_wheels)) * r_wheel
    max_motor_force_total = rover.n_drive_wheels * rover.max_wheel_torque_Nm / max(r_wheel, 1e-9)

    H_smooth_per = smooth_traction_per_wheel_N(W_per, A_contact, l_contact, slip, soil)
    H_grouser_per = grouser_traction_per_wheel_N(
        W_per, A_contact, l_contact, slip, soil, rover, grouser_height_m, n_grousers_total
    )
    H_smooth_total = rover.n_drive_wheels * H_smooth_per
    H_grouser_total = rover.n_drive_wheels * H_grouser_per

    return ResistanceComponents(
        slope_deg=slope_deg,
        speed_mps=speed_mps,
        normal_total_N=total_normal,
        normal_per_wheel_N=W_per,
        sinkage_m=z,
        contact_length_m=l_contact,
        contact_area_per_wheel_m2=A_contact,
        compression_per_wheel_N=Rc_per,
        compression_total_N=Rc_total,
        rolling_total_N=Rr_total,
        bulldozing_per_front_wheel_N=Rb_per,
        bulldozing_total_N=Rb_total,
        gravity_grade_N=Rg,
        viscous_speed_N=F_visc,
        quadratic_speed_N=F_quad,
        total_no_grade_N=F_no_grade,
        total_with_grade_N=F_with_grade,
        power_no_grade_W=P_no_grade,
        power_with_grade_W=P_with_grade,
        resistance_torque_per_drive_wheel_Nm=torque_per_drive_wheel,
        max_motor_force_total_N=max_motor_force_total,
        smooth_traction_per_wheel_N=H_smooth_per,
        smooth_traction_total_N=H_smooth_total,
        grouser_traction_per_wheel_N=H_grouser_per,
        grouser_traction_total_N=H_grouser_total,
        traction_margin_smooth_N=H_smooth_total - F_with_grade_for_power,
        traction_margin_grouser_N=H_grouser_total - F_with_grade_for_power,
    )


# -----------------------------------------------------------------------------
# Hu coefficient fitting
# -----------------------------------------------------------------------------

def fit_nonnegative_hu_force_coeffs(v: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Fit F ~= C0 + C1 |v| + C2 v^2 with C >= 0.

    Active-set enumeration avoids scipy dependency for this 3-variable NNLS.
    """
    v = np.asarray(v, dtype=float)
    F = np.asarray(F, dtype=float)
    A_full = np.column_stack([np.ones_like(v), np.abs(v), v**2])

    best_c = None
    best_sse = math.inf
    indices = [0, 1, 2]
    for r in range(1, 4):
        for active in combinations(indices, r):
            A = A_full[:, active]
            c_active, *_ = np.linalg.lstsq(A, F, rcond=None)
            if np.all(c_active >= -1e-10):
                c = np.zeros(3)
                c[list(active)] = np.maximum(c_active, 0.0)
                resid = A_full @ c - F
                sse = float(np.sum(resid**2))
                if sse < best_sse:
                    best_sse = sse
                    best_c = c

    if best_c is None:
        best_c = np.zeros(3)

    resid = A_full @ best_c - F
    rmse = float(np.sqrt(np.mean(resid**2)))
    max_abs = float(np.max(np.abs(resid)))
    return best_c, rmse, max_abs


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def write_csv(path: Path, rows: List[ResistanceComponents]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def maybe_plot(path: Path, rows: List[ResistanceComponents]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Could not import matplotlib; skipping plot. Error: {exc}")
        return

    # Plot force vs speed grouped by slope.
    slopes = sorted(set(r.slope_deg for r in rows))
    plt.figure()
    for slope in slopes:
        subset = [r for r in rows if r.slope_deg == slope]
        subset.sort(key=lambda r: r.speed_mps)
        plt.plot(
            [r.speed_mps for r in subset],
            [r.total_with_grade_N for r in subset],
            marker="o",
            label=f"{slope:g} deg",
        )
    plt.xlabel("speed [m/s]")
    plt.ylabel("total resistance with grade [N]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute EMRS-inspired lunar rover resistive forces and fit Hu-style resistance coefficients."
    )
    p.add_argument("--preset", default="emrs_breadboard_scaled_lunar",
                   choices=["emrs_breadboard_scaled_lunar", "emrs_flight_geometry_lunar"])
    p.add_argument("--mass-kg", type=float, default=None,
                   help="Override rover mass. Strongly recommended for emrs_flight_geometry_lunar.")
    p.add_argument("--output-prefix", type=str, default="emrs_resistive_forces")
    p.add_argument("--make-plot", action="store_true")

    # Rover overrides.
    p.add_argument("--wheel-diameter-m", type=float, default=None)
    p.add_argument("--wheel-width-m", type=float, default=None)
    p.add_argument("--cf", type=float, default=None, help="Rolling/internal friction coefficient.")
    p.add_argument("--max-wheel-torque-Nm", type=float, default=None)
    p.add_argument("--n-wheels", type=int, default=None)
    p.add_argument("--n-front-wheels", type=int, default=None)
    p.add_argument("--n-drive-wheels", type=int, default=None)

    # Soil overrides.
    p.add_argument("--soil-n", type=float, default=1.0)
    p.add_argument("--soil-kc", type=float, default=1400.0)
    p.add_argument("--soil-kphi", type=float, default=820_000.0)
    p.add_argument("--soil-phi-deg", type=float, default=35.0)
    p.add_argument("--soil-cohesion", type=float, default=170.0)
    p.add_argument("--soil-gamma", type=float, default=2470.0)
    p.add_argument("--soil-K-slip", type=float, default=0.018)
    p.add_argument("--recompute-soil-factors", action="store_true")

    # Evaluation grid.
    p.add_argument("--speed-min", type=float, default=0.01)
    p.add_argument("--speed-max", type=float, default=None,
                   help="Default uses preset max_speed_mps.")
    p.add_argument("--n-speed-samples", type=int, default=20)
    p.add_argument("--slope-deg-list", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0])

    # Model flags.
    p.add_argument("--no-bulldozing", action="store_true")
    p.add_argument("--exclude-grade-from-total", action="store_true",
                   help="Keep grade as a separate force; total_with_grade will exclude it.")
    p.add_argument("--viscous-force-per-mps", type=float, default=0.0,
                   help="Optional speed-dependent prior: F += kv |v|.")
    p.add_argument("--quadratic-force-per-mps2", type=float, default=0.0,
                   help="Optional speed-dependent prior: F += kq v^2.")

    # Traction check.
    p.add_argument("--slip", type=float, default=0.2,
                   help="Slip ratio used for traction-capacity estimates.")
    p.add_argument("--grouser-height-m", type=float, default=0.0,
                   help="Grouser height for traction-capacity estimate. 0 disables grouser benefit.")
    p.add_argument("--n-grousers-total", type=int, default=0,
                   help="Total number of grousers around the wheel for traction estimate.")

    # Fit choices.
    p.add_argument("--fit-slope-deg", type=float, default=0.0,
                   help="Which slope's samples to use for fitting C0,C1,C2.")
    p.add_argument("--fit-include-grade", action="store_true",
                   help="Fit C0,C1,C2 to total_with_grade instead of total_no_grade.")

    return p


def main() -> None:
    args = build_parser().parse_args()

    rover = preset_rover(args.preset, mass_override=args.mass_kg)
    if args.wheel_diameter_m is not None:
        rover.wheel_diameter_m = args.wheel_diameter_m
    if args.wheel_width_m is not None:
        rover.wheel_width_m = args.wheel_width_m
    if args.cf is not None:
        rover.rolling_friction_coeff = args.cf
    if args.max_wheel_torque_Nm is not None:
        rover.max_wheel_torque_Nm = args.max_wheel_torque_Nm
    if args.n_wheels is not None:
        rover.n_wheels = args.n_wheels
    if args.n_front_wheels is not None:
        rover.n_front_wheels = args.n_front_wheels
    if args.n_drive_wheels is not None:
        rover.n_drive_wheels = args.n_drive_wheels

    soil = LunarSoilParams(
        n=args.soil_n,
        kc=args.soil_kc,
        kphi=args.soil_kphi,
        phi_deg=args.soil_phi_deg,
        cohesion=args.soil_cohesion,
        gamma=args.soil_gamma,
        K_slip=args.soil_K_slip,
    )
    if args.recompute_soil_factors:
        soil = recompute_soil_factors(soil)

    speed_max = rover.max_speed_mps if args.speed_max is None else args.speed_max
    speeds = np.linspace(args.speed_min, speed_max, args.n_speed_samples)

    rows: List[ResistanceComponents] = []
    for slope in args.slope_deg_list:
        for v in speeds:
            rows.append(
                resistance_components(
                    rover=rover,
                    soil=soil,
                    speed_mps=float(v),
                    slope_deg=float(slope),
                    include_bulldozing=not args.no_bulldozing,
                    include_grade_in_total=not args.exclude_grade_from_total,
                    viscous_force_per_mps=args.viscous_force_per_mps,
                    quadratic_force_per_mps2=args.quadratic_force_per_mps2,
                    slip=args.slip,
                    grouser_height_m=args.grouser_height_m,
                    n_grousers_total=args.n_grousers_total,
                )
            )

    prefix = Path(args.output_prefix)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    plot_path = prefix.with_suffix(".png")

    write_csv(csv_path, rows)

    fit_rows = [r for r in rows if abs(r.slope_deg - args.fit_slope_deg) < 1e-9]
    if not fit_rows:
        # If exact slope missing, fit to first slope.
        first_slope = rows[0].slope_deg
        fit_rows = [r for r in rows if r.slope_deg == first_slope]
        fit_slope_used = first_slope
    else:
        fit_slope_used = args.fit_slope_deg
    fit_rows.sort(key=lambda r: r.speed_mps)

    v_fit = np.array([r.speed_mps for r in fit_rows])
    F_fit = np.array([r.total_with_grade_N if args.fit_include_grade else r.total_no_grade_N for r in fit_rows])
    coeffs, rmse, max_abs = fit_nonnegative_hu_force_coeffs(v_fit, F_fit)

    if args.make_plot:
        maybe_plot(plot_path, rows)

    summary = {
        "rover": asdict(rover),
        "soil": asdict(soil),
        "settings": {
            "include_bulldozing": not args.no_bulldozing,
            "include_grade_in_total": not args.exclude_grade_from_total,
            "viscous_force_per_mps": args.viscous_force_per_mps,
            "quadratic_force_per_mps2": args.quadratic_force_per_mps2,
            "slip_for_traction_check": args.slip,
            "grouser_height_m": args.grouser_height_m,
            "n_grousers_total": args.n_grousers_total,
            "fit_slope_deg_used": fit_slope_used,
            "fit_include_grade": args.fit_include_grade,
        },
        "hu_resistance_fit": {
            "force_model": "F_res(v) ~= C0 + C1*abs(v) + C2*v^2",
            "power_model": "P_res(v) = F_res(v)*v",
            "C0_N": float(coeffs[0]),
            "C1_N_per_mps": float(coeffs[1]),
            "C2_N_per_mps2": float(coeffs[2]),
            "rmse_force_N": rmse,
            "max_abs_force_error_N": max_abs,
            "identifiability_note": (
                "NASA terramechanics terms are mostly quasi-static. Without measured "
                "speed-dependent losses or user priors, the fit will mostly identify C0."
            ),
        },
        "representative_row_first": asdict(rows[0]),
        "representative_row_last": asdict(rows[-1]),
        "outputs": {
            "csv": str(csv_path.resolve()),
            "json": str(json_path.resolve()),
            "plot": str(plot_path.resolve()) if args.make_plot else None,
        },
    }

    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EMRS lunar resistive-force estimate ===")
    print(f"Preset: {rover.name}")
    print(f"Mass: {rover.mass_kg:.3g} kg")
    print(f"Wheel D x b: {rover.wheel_diameter_m:.4g} m x {rover.wheel_width_m:.4g} m")
    print(f"Rolling coefficient cf: {rover.rolling_friction_coeff:.4g}")
    print(f"Rows written: {len(rows)}")
    print("\nHu-style fit:")
    print(f"  C0 = {coeffs[0]:.8g} N")
    print(f"  C1 = {coeffs[1]:.8g} N/(m/s)")
    print(f"  C2 = {coeffs[2]:.8g} N/(m/s)^2")
    print(f"  RMSE = {rmse:.6g} N")
    print("\nFiles:")
    print(f"  CSV:  {csv_path.resolve()}")
    print(f"  JSON: {json_path.resolve()}")
    if args.make_plot:
        print(f"  Plot: {plot_path.resolve()}")
    print("\nNote: grade force can be kept separate from C0,C1,C2. For the Hu-style")
    print("model, I recommend fitting C0,C1,C2 to total_no_grade_N and handling")
    print("slope through the translational power term m*g*sin(theta)*v.")


if __name__ == "__main__":
    main()
