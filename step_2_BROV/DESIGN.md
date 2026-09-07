# step_2 — Vehicle dynamics and learned low-level control (design notes)

An IsaacLab environment for the BlueROV2 Heavy (8 thrusters) with two purposes:
a **physics ground truth** that is validated axis by axis before any learning
result is trusted, and a **reproduction of Sim2Swim** (Fosso et al., SINTEF
Ocean, arXiv:2512.08656) — a hierarchical controller in which classical
line-of-sight guidance produces a body-frame velocity command and an RL policy
tracks it. This document records what was built, what was found while building
it, and what the current state is. Deployment code lives in the companion
repository `brov_ros2`.

## 1. Physics model

Shared with `step_3` through `robots/` (promoted out of this stage so that a
change to a coefficient reaches both stages at once).

| Component | Implementation | Source |
|---|---|---|
| Rigid-body hydrodynamics | Fossen 6-DOF: added mass, linear + quadratic damping, restoring forces | Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control*, 2011 |
| Coefficients | measured BlueROV2 Heavy values | von Benzon et al., JMSE 2022 |
| Thrusters | T200 PWM→thrust from manufacturer table, third-order actuator dynamics | Blue Robotics T200 data |
| Allocation | 6×8 matrix built at runtime from thruster positions/directions in `brov2_heavy.yaml` | — |
| Max wrench `F_max` | [85, 85, 120] N, [26, 14, 22] N·m | von Benzon et al., Table 4 |

Mass, inertia and collision come from the USD asset; buoyancy, centre of
buoyancy, hydrodynamic coefficients and thruster geometry from the YAML. The two
are kept separate on purpose: one is a CAD fact, the other is a set of measured
or estimated quantities that domain randomization is allowed to perturb.

### Validation before learning

`validate_physics.py` runs five open-loop tests on the end-to-end environment
(8-dim PWM action, no allocation). A learning run is only started once these pass.

| Test | Pass criterion |
|---|---|
| neutral buoyancy | \|Δz\| < 0.1 m over 10 s |
| straight line (each axis) | axis displacement > 0.05 m, lateral drift < 50 % of it |
| rotation | commanded axis rotates, others stay bounded |
| full 6-DOF | combined translation + rotation stays bounded |
| thruster model | +64.1 N forward / −51.5 N reverse at full PWM (measured values) |

## 2. Sim2Swim MDP

**Observation (16-D, error-only).** Quaternion error `q_e = q̄_d ⊗ q` (4), body
velocity error `v^b − v_d^b` (3), body angular rate `ω^b` (3), integral of the
velocity error (3), integral of the vector part of `q_e` (3). No absolute
position or velocity appears in the observation. This is what lets the same
policy serve any guidance layer — LOS in this stage, an NBV planner in `step_3`.

**Action (6-D wrench, [−1, 1]).** Scaled by `F_max`, allocated to eight
thrusters by the pseudo-inverse of the allocation matrix, converted to PWM
through the inverse thruster model.

**Reward (paper Eq. 5–8).**

```
r = w_q · exp(−‖q_e,vec‖²) + w_v · exp(−‖v_e‖²) + w_ω · exp(−‖ω‖²)
  + w_q · exp(−∠(q_d, q)) + w_a · exp(−‖a‖)
```

with `w_q = 0.4, w_v = 0.2, w_ω = 0.05` from the paper.

**Desired state during training.** The desired attitude `q_d(t)` is generated
from the Frenet–Serret frame of the paper's Eq. 9 curve; the desired body
velocity is a fixed-magnitude 0.5 m/s vector with a direction sampled per
episode. (An earlier implementation used the Eq. 9 curve as the *velocity*
command and kept the attitude fixed; policies trained on that MDP are retained
only as a baseline, not as a reproduction.)

**Domain randomization.**

| Parameter | Nominal | Range | Reason |
|---|---|---|---|
| displaced volume | 0.01467 m³ | ±10 % | spans the buoyancy sign change used in the paper's ballast trial |
| centre of buoyancy | (0, 0, 10) mm | uniform in a 15 mm sphere | not measured on the real vehicle |
| rotational added mass | [0.189, 0.135, 0.222] | ±40 % | 30–100 % uncertainty reported by von Benzon et al. |
| action delay | — | 40–80 ms | see §4 |
| observation staleness | — | 15 % of steps repeat the previous observation | see §4 |

Mass randomization is not implemented (requires direct PhysX mass manipulation);
the ballast trial is approximated by an equivalent volume reduction plus a
lateral centre-of-buoyancy offset.

**Training.** RSL-RL PPO, 2048 environments, 128-step rollouts, 50 iterations
(≈ 13 M transitions), ≈ 85 s wall time at ~150 k FPS. The original 2.4 k FPS
bottleneck was per-environment debug drawing that ran even headless.

## 3. Guidance and evaluation

`guidance/los_guidance.py` implements attitude-independent 3D line-of-sight
guidance for a fully actuated vehicle: the velocity command points from the
vehicle to a look-ahead point on the path, in the world frame. Heading modes:
align-with-path, always-upright, and re-sample a random attitude at every
waypoint (the paper's trial (c)).

`test_policy.py` reproduces the paper's three real-vehicle trials in simulation
— straight line, square with ballast, square with random attitude at waypoints —
and plots them in the paper's figure layout. Domain-randomized quantities are
pinned per scenario for reproducibility.

## 4. Findings

### 4.1 Action penalty sets steady-state tracking, and thereby loop gain

With the paper's `w_a = 0.3`, the trained policy tracked a 0.5 m/s command at
about 16 %. Lowering `w_a` to 0.017 restored tracking to 100 % at the same
setpoint. The mechanism is an equilibrium of the reward: at any steady velocity
error, the marginal reward from reducing the error is balanced against the
marginal penalty of the thrust needed to do so, and the paper's weighting puts
that equilibrium far from zero error.

The correction is not free. A policy that tracks properly has a higher loop
gain, and a simulator with no transport delay never charges for it. A deployed
loop does — which is where §4.2 begins.

### 4.2 Delay-aware training

On the real vehicle the well-tracking policy produced a 3-axis oscillation at
about 2 Hz. Loop dead time measured on the vehicle was ≈ 80 ms; a phase-budget
argument attributes the oscillation to a delay–saturation limit cycle (details
in `brov_ros2/docs`). Two training-side remedies were compared (`DELAY_TRAINING_PLAN.md`):

| Design | Change to the MDP | Result at 80 ms delay | Result at 0–20 ms delay |
|---|---|---|---|
| A — delay randomization | action delay U(40, 80) ms + observation staleness | \|ω\| p90 down 14×, saturation 0 %, tracking 100 % | conservative, stable |
| B — A + action history in the observation (Markov-restoring, after Katsikopoulos & Engelbrecht 2003) | same DR, 28-D observation | 2–12× lower RMSE than A | **fails**: 7.7–9.2 Hz chatter, 23–26 % saturation |

B's history term is a prediction that cancels actions "in flight". Trained only
on 40–80 ms, it always assumes 1–2 steps are in flight; when delay disappears
the cancellation has nothing to cancel and becomes the oscillation source. This
is the out-of-distribution failure described by Imai et al. (2021) and was
predictable from the training range excluding zero. Design A was carried to the
vehicle.

### 4.3 What reproducing a paper surfaces

Three things the paper does not state were needed to reproduce it: the
Frenet–Serret meaning of Eq. 9, the action-penalty equilibrium above, and the
delay budget of the deployed loop. None is a criticism of the paper; each is a
detail that a reproduction cannot proceed without.

## 5. Status

- Physics validated; Sim2Swim reproduction with corrected desired-state
  generation trained and evaluated in IsaacLab (0 / 0.1 / 0.5 m/s steady
  tracking, zero actuator-bound hits in steady state).
- Design-A policy deployed on a BlueROV2 Heavy in a test tank via `brov_ros2`.
  Actuator saturation fell from 58 % to 7 % (surge) and 67 % to 4 % (sway);
  divergence-free runs exceeded 150 s. Remaining: a residual 2 Hz component in
  surge; DVL dropouts, later traced to the DVL mounting position in discussion
  with the Sim2Swim authors.
- Underactuated (torpedo-type) vehicle support is planned but not started.

## References

- Fosso, Amundsen, Xanthidis, Ohrem. *Sim2Swim: Zero-Shot Velocity Control for Agile AUV Maneuvering in 3 Minutes.* arXiv:2512.08656, 2025.
- Fosso et al. *Learning to Swim.* arXiv:2410.00120, 2024.
- von Benzon et al. *An Open-Source Benchmark Simulator: Control of a BlueROV2 Underwater Robot.* JMSE 10(12):1898, 2022.
- Fossen. *Handbook of Marine Craft Hydrodynamics and Motion Control.* Wiley, 2011.
- Katsikopoulos & Engelbrecht. *Markov decision processes with delays and asynchronous cost collection.* IEEE TAC, 2003.
- Imai et al. *Vision-guided quadrupedal locomotion in the wild with multi-modal delay randomization.* 2021.
- Chu et al. *MarineGym.* IROS 2025, arXiv:2503.09203.
