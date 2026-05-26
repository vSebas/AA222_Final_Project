# EMRS-Hu Hybrid Rover Dynamic and Energy Model

## 1. Modeling goal

This document defines a rover model suitable for minimum-energy lunar trajectory optimization, especially with successive convexification / sequential convex programming (SCP).

The chosen model is:

$$
\boxed{
\text{Hu-style SE(2) energy-constrained planning model}
+
\text{EMRS rover geometry and wheel parameters}
+
\text{NASA lunar terramechanics}
}
$$

The intent is not to build a high-fidelity multibody rover simulator. The intent is to define a tractable, physically motivated model that can be linearized and used inside an SCP loop.

---

## 2. Source motivation

### 2.1 Hu et al. model contribution

Hu et al., **"Energy-Constrained Navigation for Planetary Rovers under Hybrid RTG-Solar Power"**, propose an energy-aware rover trajectory optimization framework that includes:

- SE(2)-based rover trajectory planning.
- Physics-based translational power.
- Physics-based rotational power.
- Persistent resistive power.
- Baseline subsystem power.
- Cumulative energy constraints.
- Instantaneous power constraints.
- Hybrid RTG-solar power generation.

This motivates the overall structure:

$$
\dot{E}
=
P_{\mathrm{gen}} - P_{\mathrm{load}}
$$

with

$$
P_{\mathrm{load}}
=
P_{\mathrm{base}}
+
P_{\mathrm{trans}}
+
P_{\mathrm{rot}}
+
P_{\mathrm{res}}.
$$

Reference:

- Hu et al., 2025: <https://arxiv.org/abs/2509.15062>

---

### 2.2 EMRS contribution

The **European Moon Rover System (EMRS)** papers provide a better physical rover baseline than Hu's generic simulated rover because EMRS gives concrete geometry, wheel parameters, locomotion modes, and breadboard energy data.

The main EMRS design paper gives:

| Quantity | Value |
|---|---:|
| Wheel diameter | $D_w = 0.612~\mathrm{m}$ |
| Wheel width | $b_w = 0.216~\mathrm{m}$ |
| Wheel radial stiffness | $2500\text{--}6000~\mathrm{N/m}$ |
| Max speed | $3~\mathrm{km/h}=0.833~\mathrm{m/s}$ |
| Max torque per wheel | $80~\mathrm{Nm}$ |
| Wheel mass with in-hub motor | $m_w = 7~\mathrm{kg}$ |
| Deployed rover dimensions | $2.366 \times 1.525 \times 1.0~\mathrm{m}$ |
| Wheel spacing | $1.775~\mathrm{m}$ longitudinal, $1.284~\mathrm{m}$ transverse |

Reference:

- Luna et al., **"The European Moon Rover System: a modular multipurpose rover for future complex lunar missions"**, 2023: <https://arxiv.org/abs/2311.03136>

The EMRS modularity/test paper motivates using EMRS as a multipurpose lunar rover concept. It states that EMRS was designed for multiple lunar mission scenarios and tested in analogue facilities with lunar soil simulant, different locomotion modes, obstacles, and excavation tasks.

Reference:

- Luna et al., **"Modularity for lunar exploration: European Moon Rover System Pre-Phase A Design and Field Test Campaign Results"**, 2023: <https://arxiv.org/abs/2311.03098>

The EMRS breadboard paper gives measured test data useful for calibration:

| Quantity | Value |
|---|---:|
| Breadboard mass | $84~\mathrm{kg}$ |
| Body dimensions | $0.890 \times 0.230 \times 0.370~\mathrm{m}$ |
| Wheel spacing | $0.980 \times 0.830~\mathrm{m}$ |
| Ground clearance | $0.250~\mathrm{m}$ |
| Drive motors | $13~\mathrm{W}$ brushed DC motors |
| Steering motors | $16~\mathrm{W}$ brushed DC motors |
| Tested locomotion modes | Ackermann, skid, crab, point-turn |
| Energy metric | Cost of transport, $\epsilon = P/(mgv)$ |

Reference:

- Luna et al., **"Breadboarding the European Moon Rover System: discussion and results of the analogue field test campaign"**, 2024: <https://arxiv.org/abs/2411.13978>

---

### 2.3 NASA lunar terramechanics contribution

NASA's LTV terramechanics white paper gives lunar wheel-soil resistance models and typical lunar soil parameters. It decomposes wheel/vehicle resistance into:

$$
F_{\mathrm{terrain}}
=
R_c + R_r + R_b + R_g
$$

where:

- $R_c$: compression resistance.
- $R_r$: rolling/internal resistance.
- $R_b$: bulldozing resistance.
- $R_g$: gravitational/slope resistance.

It also provides typical lunar soil values:

| Symbol | Description | Lunar value |
|---|---|---:|
| $n$ | exponent of sinkage | $1.0$ |
| $k_c$ | cohesive modulus | $1400~\mathrm{N/m^2}$ |
| $k_\phi$ | frictional modulus | $820000~\mathrm{N/m^3}$ |
| $\phi$ | internal friction angle | $30^\circ\text{--}40^\circ$ |
| $c$ | soil cohesion | $170~\mathrm{N/m^2}$ |
| $\gamma$ | soil weight density | $2470~\mathrm{N/m^3}$ |
| $K$ | soil slip coefficient | $0.018~\mathrm{m}$ |
| $N_q$ | Terzaghi bearing factor | $32.23$ |
| $N_c$ | Terzaghi bearing factor | $48.09$ |
| $N_\gamma$ | Terzaghi bearing factor | $33.27$ |
| $K_c$ | cohesive deformation modulus | $33.37$ |
| $K_\gamma$ | density deformation modulus | $72.77$ |

Reference:

- Li and Bingham, **"NASA White Paper: Terramechanics for LTV Modeling and Simulation"**, 2022: <https://ntrs.nasa.gov/citations/20220010732>

---

## 3. Chosen rover abstraction

Use an **EMRS-inspired rover**, but reduced to a planar dynamic model.

The high-level physical interpretation is:

> The rover is an EMRS-like four-wheel lunar rover, but for trajectory optimization we use a reduced-order planar model with differential/skid-steer-equivalent motion.

This is appropriate because:

1. Hu's model is already an SE(2)-based planning model, not a full multibody rover simulator.
2. SCP benefits from low-dimensional differentiable dynamics.
3. The project objective is minimum-energy path/trajopt, not detailed steering-actuator design.
4. EMRS provides enough geometry and wheel information to parameterize mass, inertia, wheel-terrain resistance, velocity limits, and torque limits.

---

## 4. Differential-drive / skid-steer choice

### Recommendation

For the first project implementation, use a **differential-drive-equivalent / skid-steer-equivalent** rover model.

That means the optimization controls forward acceleration and yaw acceleration directly:

$$
u =
\begin{bmatrix}
a & \alpha
\end{bmatrix}^T
$$

rather than steering angles.

This is the simplest model consistent with Hu's SE(2) planning formulation. It also lets us set steering power to zero in the baseline:

$$
P_{\mathrm{steer}} = 0.
$$

This does **not** mean the physical EMRS rover lacks steering. It means we are modeling a restricted locomotion mode or an equivalent planar motion model.

### Why this is acceptable

The EMRS platform supports multiple locomotion modes: skid steering, Ackermann, crab, and point turn. For a first SCP implementation, modeling all steering angles and mode switches would add nonconvex hybrid dynamics. The optimization problem would become significantly more complicated.

A differential/skid-steer equivalent keeps the model tractable while retaining:

- slope effects,
- terrain resistance,
- yaw dynamics,
- energy consumption,
- velocity and torque constraints,
- battery-energy depletion.

### Limitation

The EMRS breadboard paper reports that point-turn and Ackermann-type steering can be more efficient than skid steering for some turning maneuvers. Therefore, ignoring steering and using skid/differential turning may overestimate turning losses or misrepresent large yaw maneuvers.

A good project structure is:

1. **Baseline:** differential-drive equivalent, $P_{\mathrm{steer}}=0$.
2. **Optional extension:** add steering power penalty or compare against point-turn/Ackermann approximations.

---

## 5. State, controls, and terrain quantities

### 5.1 State

Use:

$$
x =
\begin{bmatrix}
X & Y & \psi & v & \omega & E
\end{bmatrix}^T
$$

where:

| Symbol | Meaning |
|---|---|
| $X,Y$ | rover position in global/map frame |
| $\psi$ | rover heading |
| $v$ | forward body-frame speed |
| $\omega$ | yaw rate |
| $E$ | remaining stored energy / battery energy |

### 5.3 Terrain quantities

From a DEM or terrain map, define:

| Quantity | Meaning |
|---|---|
| $h(X,Y)$ | terrain elevation |
| $\nabla h(X,Y)$ | local terrain gradient |
| $\theta(X,Y,\psi)$ | slope angle along rover heading |
| $S(X,Y,t)$ | illumination/shadow indicator, optional |
| $\mu_{\mathrm{terrain}}(X,Y)$ | optional terrain-dependent friction/slip coefficient |

The along-track slope may be approximated as:

$$
\theta(X,Y,\psi)
\approx
\tan^{-1}
\left(
\nabla h(X,Y)^T
\begin{bmatrix}
\cos\psi\\
\sin\psi
\end{bmatrix}
\right).
$$

For small slopes:

$$
\sin\theta \approx
\nabla h(X,Y)^T
\begin{bmatrix}
\cos\psi\\
\sin\psi
\end{bmatrix}.
$$

---

## 6. Continuous-time dynamics

The proposed continuous-time model is:

$$
\dot{X} = v\cos\psi
$$

$$
\dot{Y} = v\sin\psi
$$

$$
\dot{\psi} = \omega
$$

$$
\dot{v} = a
$$

$$
\dot{\omega} = \alpha
$$

$$
\dot{E} =
-P_{\mathrm{load}}
$$

for the simplified battery-only version.

The full generation-aware version is:

$$
\dot{E}
=
P_{\mathrm{gen}}(X,Y,t)
-
P_{\mathrm{load}}.
$$

For the initial implementation, use the simplified form:

$$
\boxed{
\dot{E} =
-P_{\mathrm{load}}
}
$$

with finite initial energy:

$$
E(0)=E_0.
$$

This is equivalent to assuming a fixed stored energy budget over the planning horizon.

---

## 7. Power model

Use:

$$
P_{\mathrm{load}}
=
P_{\mathrm{base}}
+
P_{\mathrm{trans}}
+
P_{\mathrm{rot}}
+
P_{\mathrm{res}}
+
P_{\mathrm{steer}}.
$$

For the baseline model:

$$
\boxed{
P_{\mathrm{steer}} = 0
}
$$

but keep the steering formula available as an optional extension.

---

## 8. Baseline subsystem power

$$
P_{\mathrm{base}}
=
P_{\mathrm{avionics}}
+
P_{\mathrm{thermal}}
+
P_{\mathrm{comm}}
+
P_{\mathrm{sensing}}.
$$

For the first implementation, use a constant value:

$$
P_{\mathrm{base}} = \bar{P}_{\mathrm{base}}.
$$

This follows Hu's use of baseline subsystem load.

If the project horizon is short and the objective is traction/locomotion energy, one may initially set:

$$
P_{\mathrm{base}}=0
$$

to isolate mobility energy. But for a more realistic mission model, keep $P_{\mathrm{base}}>0$.

---

## 9. Translational power

The translational mechanical power is:

$$
P_{\mathrm{trans}}
=
\left[
\left(
ma + mg_\ell \sin\theta
\right)v
\right]_+
$$

where:

| Symbol | Meaning |
|---|---|
| $m$ | rover mass |
| $g_\ell$ | lunar gravity, $1.62~\mathrm{m/s^2}$ |
| $\theta$ | terrain slope along heading |
| $v$ | rover speed |
| $[z]_+$ | $\max(z,0)$ |

The positive-part operator prevents downhill gravity from becoming artificial energy generation unless regenerative braking is explicitly modeled.

For SCP, introduce an auxiliary variable:

$$
p_{\mathrm{trans}}
\ge
\left(
ma + mg_\ell \sin\theta
\right)v
$$

$$
p_{\mathrm{trans}}\ge 0
$$

and use:

$$
P_{\mathrm{trans}} = p_{\mathrm{trans}}.
$$

---

## 10. Rotational power

The rotational mechanical power is:

$$
P_{\mathrm{rot}}
=
\left[
I_z \alpha\omega
\right]_+.
$$

For SCP, use an auxiliary variable:

$$
p_{\mathrm{rot}}
\ge
I_z\alpha\omega
$$

$$
p_{\mathrm{rot}}\ge 0.
$$

Then:

$$
P_{\mathrm{rot}} = p_{\mathrm{rot}}.
$$

---

## 11. Terrain resistance power

### 11.1 Hu-style persistent resistance form

Hu's persistent-resistance model can be written as:

$$
P_{\mathrm{res}}
=
(C_0 + C_1|v| + C_2v^2)v.
$$

For forward-only motion, $v\ge 0$, this becomes:

$$
P_{\mathrm{res}}
=
C_0v + C_1v^2 + C_2v^3.
$$

The coefficients have units:

| Coefficient | Units | Interpretation |
|---|---|---|
| $C_0$ | $\mathrm{N}$ | constant terrain/rolling force |
| $C_1$ | $\mathrm{N\,s/m}$ | speed-proportional loss |
| $C_2$ | $\mathrm{N\,s^2/m^2}$ | quadratic force / cubic power loss |

### 11.2 NASA terramechanics resistance

Use NASA's terrain resistance decomposition:

$$
F_{\mathrm{terrain}}
=
R_c + R_r + R_b + R_g.
$$

The resistance power is:

$$
P_{\mathrm{res}}
=
F_{\mathrm{terrain}}v.
$$

#### Compression resistance

For lunar soil with $n=1$, NASA gives approximately:

$$
R_c
\approx
0.85854
\left(
\frac{W^4}
{(k_c + b k_\phi)D^2}
\right)^{1/3}
$$

where:

| Symbol | Meaning |
|---|---|
| $W$ | normal force on a wheel |
| $D$ | wheel diameter |
| $b$ | wheel width |
| $k_c$ | cohesive modulus |
| $k_\phi$ | frictional modulus |

For a four-wheel rover on locally level terrain:

$$
W \approx \frac{mg_\ell}{4}.
$$

On sloped terrain, use terrain-normal load if available.

#### Rolling/internal resistance

NASA uses:

$$
R_r = W_v c_f
$$

where:

| Symbol | Meaning |
|---|---|
| $W_v$ | vehicle weight on level surface |
| $c_f$ | rolling friction coefficient |

For the Moon:

$$
W_v = mg_\ell.
$$

#### Gravitational/slope resistance

NASA uses:

$$
R_g = W_v\sin\theta.
$$

This is already included in $P_{\mathrm{trans}}$ above through $mg_\ell \sin\theta$. Therefore, to avoid double counting, use one of these conventions:

**Convention A:**

$$
P_{\mathrm{trans}}
=
[mav]_+,
\qquad
P_{\mathrm{res}}
=
(R_c + R_r + R_b + R_g)v.
$$

**Convention B, recommended:**

$$
P_{\mathrm{trans}}
=
[(ma+mg_\ell\sin\theta)v]_+,
\qquad
P_{\mathrm{res}}
=
(R_c + R_r + R_b)v.
$$

Use Convention B.

#### Bulldozing resistance

NASA defines bulldozing resistance as a function of:

| Symbol | Meaning |
|---|---|
| $b$ | wheel width |
| $z$ | sinkage depth |
| $c$ | soil cohesion |
| $\gamma$ | soil weight density |
| $\phi$ | internal friction angle |
| $K_c, K_\gamma$ | soil deformation moduli |
| $D$ | wheel diameter |

For the first implementation, either:

1. compute $R_b$ from NASA's formula, or
2. absorb bulldozing into $C_0$ by fitting $C_0,C_1,C_2$.

For SCP, option 2 is simpler.

---

## 12. Estimating $C_0,C_1,C_2$

Two estimation routes are recommended.

### 12.1 Physics-derived fit from NASA terramechanics

Compute:

$$
F_{\mathrm{terrain},k}
=
R_{c,k}
+
R_{r,k}
+
R_{b,k}
$$

for representative terrain and speed points. Then fit:

$$
F_{\mathrm{terrain},k}
\approx
C_0 + C_1 v_k + C_2v_k^2.
$$

Solve:

$$
\min_{C_0,C_1,C_2\ge 0}
\sum_k
\left[
F_{\mathrm{terrain},k}
-
(C_0+C_1v_k+C_2v_k^2)
\right]^2.
$$

Then:

$$
P_{\mathrm{res}}
=
(C_0+C_1v+C_2v^2)v.
$$

### 12.2 Empirical fit from EMRS breadboard cost of transport

The EMRS breadboard paper reports cost of transport:

$$
\epsilon =
\frac{P}{mgv}.
$$

Thus, for each test point:

$$
P_k = \epsilon_k m g v_k.
$$

Then fit:

$$
P_k - P_{\mathrm{base},k} - P_{\mathrm{slope},k}
\approx
C_0v_k + C_1v_k^2 + C_2v_k^3.
$$

The EMRS paper gives flat-terrain cost-of-transport values at $3,6,8~\mathrm{cm/s}$, plus slope and excavation cases. Those can be used for calibration, but note that they were obtained on Earth in analogue lunar regolith. For lunar use, scale weight-dependent terms with $g_\ell$ and/or refit using the NASA lunar terramechanics model.

---

## 13. Steering power model

Even if the baseline ignores steering, include the optional model.

### 13.1 Full four-wheel steering form

Let $\delta_i$ be the steering angle of wheel $i$, and $\dot{\delta}_i$ the steering rate.

A simple steering actuator power model is:

$$
P_{\mathrm{steer}}
=
\sum_{i=1}^4
\left(
\frac{[\tau_{\delta,i}\dot{\delta}_i]_+}{\eta_\delta}
+
P_{\delta,\mathrm{hold}}\mathbf{1}_{|\dot{\delta}_i|>0}
\right)
$$

where:

| Symbol | Meaning |
|---|---|
| $\tau_{\delta,i}$ | steering torque for wheel $i$ |
| $\dot{\delta}_i$ | steering angular rate |
| $\eta_\delta$ | steering actuator efficiency |
| $P_{\delta,\mathrm{hold}}$ | overhead/holding/driver power during steering |
| $[z]_+$ | positive mechanical power |

If torque data are unavailable, use a linear surrogate:

$$
P_{\mathrm{steer}}
=
c_\delta
\sum_{i=1}^4
|\dot{\delta}_i|
+
P_{\delta,0}
\sum_{i=1}^4
\mathbf{1}_{|\dot{\delta}_i|>0}.
$$

For differentiable optimization, replace the indicator with a smooth approximation or simply omit it:

$$
P_{\mathrm{steer}}
\approx
c_\delta
\sum_{i=1}^4
|\dot{\delta}_i|.
$$

### 13.2 Reduced yaw-rate surrogate

If steering angles are not states, use:

$$
P_{\mathrm{steer}}
=
k_\omega |\omega|
+
k_\alpha |\alpha|.
$$

This penalizes turning and yaw acceleration without modeling individual wheel steering.

### 13.3 Differential-drive baseline

For the differential-drive/skid-steer-equivalent baseline:

$$
\delta_i = 0,
\qquad
\dot{\delta}_i = 0,
\qquad
P_{\mathrm{steer}} = 0.
$$

Turning energy is still represented through:

$$
P_{\mathrm{rot}}
=
[I_z\alpha\omega]_+
$$

and, if desired, an added yaw-resistance term:

$$
P_{\mathrm{yaw,res}}
=
C_\omega |\omega|
+
C_{\omega 2}\omega^2.
$$

---

## 14. Inertia model

Approximate the rover body as a rectangular box with center of mass at the geometric center.

Let:

| Symbol | Meaning |
|---|---|
| $L$ | body length |
| $W$ | body width |
| $H$ | body height |
| $m$ | total rover mass |
| $m_w$ | mass of each wheel |
| $m_b=m-4m_w$ | body mass excluding wheels |

The body yaw inertia is:

$$
I_{z,\mathrm{body}}
=
\frac{1}{12}m_b(L^2+W^2).
$$

Model each wheel as a point mass at:

$$
x_i = \pm \frac{L_{\mathrm{wb}}}{2},
\qquad
y_i = \pm \frac{W_{\mathrm{tr}}}{2}.
$$

Then:

$$
I_{z,\mathrm{wheels}}
=
\sum_{i=1}^4
m_w(x_i^2+y_i^2).
$$

Thus:

$$
I_z
=
I_{z,\mathrm{body}}
+
I_{z,\mathrm{wheels}}.
$$

### Recommended parameters

For flight-like EMRS geometry:

$$
L=2.366~\mathrm{m},
\qquad
W=1.525~\mathrm{m},
\qquad
H=1.0~\mathrm{m}
$$

$$
L_{\mathrm{wb}}=1.775~\mathrm{m},
\qquad
W_{\mathrm{tr}}=1.284~\mathrm{m}.
$$

Use:

$$
m_w=7~\mathrm{kg}.
$$

The accessible EMRS papers do not clearly give a final flight mass. Therefore:

- Use $m=84~\mathrm{kg}$ for breadboard-calibrated studies.
- Treat $m$ as a tunable parameter for flight-like studies.
- If carrying payload, define $m=m_{\mathrm{rover}}+m_{\mathrm{payload}}$.

---

## 15. Energy model options

### 15.1 Simplified model: stored-energy budget only

For the first implementation, use:

$$
\boxed{
\dot{E} = -P_{\mathrm{load}}
}
$$

with:

$$
E(0)=E_0,
\qquad
E(t)\ge E_{\min}.
$$

This is the cleanest version for minimum-energy trajectory optimization. It avoids the need to model solar incidence, shadows, RTG sizing, and time-varying illumination.

Recommended initial objective:

$$
\min
\int_0^T
P_{\mathrm{load}}(t)\,dt
$$

or discrete form:

$$
\min
\sum_{k=0}^{N-1}
P_{\mathrm{load},k}\Delta t.
$$

This is sufficient if the project goal is:

> Find a dynamically feasible low-energy trajectory over lunar terrain.

### 15.2 Full Hu-style generation-aware model

The full model is:

$$
\dot{E}
=
P_{\mathrm{gen}}(X,Y,t)
-
P_{\mathrm{load}}.
$$

Use:

$$
P_{\mathrm{gen}}
=
P_{\mathrm{RTG}}
+
P_{\mathrm{PV}}(X,Y,t).
$$

The RTG term is constant:

$$
P_{\mathrm{RTG}}=\bar{P}_{\mathrm{RTG}}.
$$

The PV term may be modeled as:

$$
P_{\mathrm{PV}}
=
\eta_{\mathrm{PV}}
A_{\mathrm{PV}}
I_\odot(t)
\max(0,\cos\beta)
S(X,Y,t)
$$

where:

| Symbol | Meaning |
|---|---|
| $\eta_{\mathrm{PV}}$ | PV efficiency |
| $A_{\mathrm{PV}}$ | solar panel area |
| $I_\odot(t)$ | solar irradiance |
| $\beta$ | sun-panel incidence angle |
| $S(X,Y,t)$ | binary or continuous illumination factor |

This should be treated as an extension. It is useful if the project later includes illumination-aware route planning near the lunar south pole.

### 15.3 Instantaneous power constraint

Even in the simplified battery-only model, one can optionally include:

$$
P_{\mathrm{load},k}
\le
P_{\max}.
$$

In the full generation-aware model:

$$
P_{\mathrm{load},k}
\le
P_{\mathrm{gen},k}
+
P_{\mathrm{batt,max}}.
$$

For the first version, keep only:

$$
E_k \ge E_{\min}
$$

and optionally:

$$
P_{\mathrm{load},k}\le P_{\max}.
$$

---

## 16. Constraints

### 16.1 Speed

From EMRS:

$$
0 \le v \le 0.833~\mathrm{m/s}.
$$

If using breadboard-calibrated data, typical test speeds were lower, around $0.03\text{--}0.08~\mathrm{m/s}$, while the paper mentions an expected flight-design average speed of $12.67~\mathrm{cm/s}$. A conservative planning bound is:

$$
0 \le v \le 0.15~\mathrm{m/s}
$$

for nominal traverse simulations.

### 16.2 Acceleration

Use a conservative acceleration bound:

$$
|a| \le a_{\max}.
$$

This can be estimated from wheel torque:

$$
F_{\max}
\approx
\frac{4\tau_{\max}}{r_w}
$$

$$
a_{\max}
\approx
\frac{F_{\max}-F_{\mathrm{terrain}}}{m}.
$$

With:

$$
\tau_{\max}=80~\mathrm{Nm},
\qquad
r_w=0.306~\mathrm{m}.
$$

This gives a large theoretical force. In practice, traction and slip will dominate, so cap $a_{\max}$ conservatively.

### 16.3 Yaw rate

Use:

$$
|\omega|\le \omega_{\max}.
$$

For differential-drive equivalent motion, an approximate kinematic bound is:

$$
|\omega|
\le
\frac{2v}{W_{\mathrm{tr}}}
$$

for skid-steer-like turning. For point turns, this bound can be relaxed if $v=0$.

### 16.4 Energy

$$
E_{\min}\le E_k \le E_{\max}
$$

with:

$$
E_0 \le E_{\max}.
$$

### 16.5 Terrain slope / hazard

Use:

$$
|\theta_k|\le \theta_{\max}.
$$

For initial implementation, choose:

$$
\theta_{\max}=15^\circ\text{--}25^\circ.
$$

EMRS test campaigns include ramp and slope traverses up to significant inclinations, but a conservative planning limit should be used unless slip/stability constraints are explicitly modeled.

---

## 17. Discrete-time SCP form

Using time step $\Delta t$, discretize:

$$
X_{k+1}
=
X_k + \Delta t\,v_k\cos\psi_k
$$

$$
Y_{k+1}
=
Y_k + \Delta t\,v_k\sin\psi_k
$$

$$
\psi_{k+1}
=
\psi_k + \Delta t\,\omega_k
$$

$$
v_{k+1}
=
v_k + \Delta t\,a_k
$$

$$
\omega_{k+1}
=
\omega_k + \Delta t\,\alpha_k
$$

$$
E_{k+1}
=
E_k
-
\Delta t\,P_{\mathrm{load},k}.
$$

The nonconvex terms are:

- $v\cos\psi$,
- $v\sin\psi$,
- $\sin\theta(X,Y,\psi)$,
- $a v$,
- $\alpha\omega$,
- $v^2,v^3$ if used directly,
- absolute values if $v$ is allowed to be negative.

For SCP, linearize nonconvex equality constraints around the previous trajectory and add trust regions:

$$
\|x_k-x_k^{\mathrm{ref}}\|_\infty \le \Delta_x
$$

$$
\|u_k-u_k^{\mathrm{ref}}\|_\infty \le \Delta_u.
$$

Add dynamics defect slacks:

$$
x_{k+1}
=
f_{\mathrm{lin}}(x_k,u_k)
+
s_k
$$

and penalize:

$$
\rho \sum_k \|s_k\|_1.
$$

This matches the intended SCvx/SCP implementation style.

---

## 18. Recommended final model for the current project

Use this as the first implementable model:

$$
\boxed{
x =
[X,Y,\psi,v,\omega,E]^T
}
$$

$$
\boxed{
u =
[a,\alpha]^T
}
$$

$$
\boxed{
\begin{aligned}
\dot{X} &= v\cos\psi \\
\dot{Y} &= v\sin\psi \\
\dot{\psi} &= \omega \\
\dot{v} &= a \\
\dot{\omega} &= \alpha \\
\dot{E} &= -P_{\mathrm{load}}
\end{aligned}
}
$$

with:

$$
\boxed{
P_{\mathrm{load}}
=
P_{\mathrm{base}}
+
[(ma+mg_\ell\sin\theta)v]_+
+
[I_z\alpha\omega]_+
+
(C_0+C_1v+C_2v^2)v
}
$$

and:

$$
\boxed{
P_{\mathrm{steer}}=0
}
$$

for the differential-drive/skid-steer-equivalent baseline.

Use EMRS values:

$$
D_w=0.612~\mathrm{m},
\qquad
b_w=0.216~\mathrm{m},
\qquad
\tau_{\max}=80~\mathrm{Nm},
\qquad
v_{\max}=0.833~\mathrm{m/s}.
$$

Use NASA lunar soil values to compute or fit $C_0,C_1,C_2$.

Use finite stored energy:

$$
E(0)=E_0,
\qquad
E_k\ge E_{\min}.
$$

Ignore generation in the first version:

$$
P_{\mathrm{gen}}=0.
$$

Include generation only in a later extension:

$$
P_{\mathrm{gen}}=P_{\mathrm{RTG}}+P_{\mathrm{PV}}(X,Y,t).
$$

---

## 19. Practical modeling decisions

| Question | Recommended choice |
|---|---|
| Rover baseline | EMRS-inspired rover |
| Dynamics | Planar SE(2), Hu-style |
| Locomotion | Differential/skid-steer equivalent |
| Steering power | Ignore in baseline, include formula as extension |
| Energy generation | Ignore initially; use stored energy $E_0$ |
| Resistance | NASA lunar terramechanics projected into Hu polynomial |
| Inertia | Rectangular body + point-mass wheels |
| COM | geometric center / COG |
| Optimization | full-space SCP/SCvx with trust regions and slacks |
| Calibration | EMRS breadboard CoT and/or NASA terramechanics |

---

## 20. Main assumptions

1. The rover is modeled in planar SE(2).
2. Roll, pitch, suspension dynamics, wheel sinkage dynamics, and detailed slip dynamics are not state variables.
3. Terrain affects the model through slope and resistance.
4. The rover center of mass is at the geometric center.
5. Inertia is approximated from simple shapes.
6. Steering power is ignored in the baseline differential-drive model.
7. Energy generation is ignored in the first implementation.
8. $C_0,C_1,C_2$ are not universal constants; they must be estimated from lunar terramechanics or EMRS-like energy data.
9. The model is intended for trajectory optimization, not hardware-level rover simulation.

---

## 21. Suggested wording for report/proposal

> We model an EMRS-inspired lunar rover using a reduced-order SE(2) dynamics model adapted from Hu et al.'s energy-constrained rover planning framework. EMRS is selected because its design papers provide wheel dimensions, wheel stiffness, torque limits, speed limits, locomotion modes, chassis dimensions, and breadboard energy data. Lunar wheel-soil resistance is modeled using NASA LTV terramechanics parameters, and the resulting terrain resistance is projected into Hu et al.'s polynomial persistent-resistance form. For the initial SCP implementation, we use a differential-drive/skid-steer-equivalent model and ignore steering actuator power, while retaining an optional steering power term for later comparison. We initially ignore PV/RTG generation and enforce a finite stored-energy budget through an energy state $E$, with $E(0)=E_0$ and $E_k\ge E_{\min}$. This provides a tractable model for minimum-energy trajectory optimization while preserving physically meaningful dependence on rover geometry, lunar gravity, terrain slope, and wheel-soil resistance.

---

## 22. References

1. Tianxin Hu, Weixiang Guo, Ruimeng Liu, Xinhang Xu, Rui Qian, Jinyu Chen, Shenghai Yuan, Lihua Xie. **Energy-Constrained Navigation for Planetary Rovers under Hybrid RTG-Solar Power**. arXiv:2509.15062, 2025. <https://arxiv.org/abs/2509.15062>

2. Cristina Luna et al. **The European Moon Rover System: a modular multipurpose rover for future complex lunar missions**. arXiv:2311.03136, 2023. <https://arxiv.org/abs/2311.03136>

3. Cristina Luna et al. **Modularity for lunar exploration: European Moon Rover System Pre-Phase A Design and Field Test Campaign Results**. arXiv:2311.03098, 2023. <https://arxiv.org/abs/2311.03098>

4. Cristina Luna et al. **Breadboarding the European Moon Rover System: discussion and results of the analogue field test campaign**. arXiv:2411.13978, 2024. <https://arxiv.org/abs/2411.13978>

5. Zu Qun Li and Lee K. Bingham. **NASA White Paper: Terramechanics for LTV Modeling and Simulation**. NASA Technical Reports Server, Document ID 20220010732, 2022. <https://ntrs.nasa.gov/citations/20220010732>
