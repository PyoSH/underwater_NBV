# step_3 — Integration: NBV planner on a real vehicle (design notes)

**Status: in progress. Simulation only.** This stage joins the viewpoint policy
of `step_1` with the vehicle model of `step_2` and prepares it for a tank
deployment. The working plan with dates and task queue is
`DEPLOY_3WEEK_PLAN.md`; this file records the design decisions.

## 1. Why deployment reshapes the problem

Three facts about a real tank force the design:

```
tank water quality unknown  +  target object not chosen by us  +  no ground-truth depth
  → the policy must generalize over object and water           (multi-object training + water DR)
  → reward and observation must be invariant to both            (normalization is the critical path)
  → the voxel observation must be built from estimated depth    (a sensor problem, not a planning one)
```

Reward normalization, object diversity and depth estimation are therefore one
track, not three features.

## 2. Substitutions relative to step_2

| Layer | step_2 | step_3 |
|---|---|---|
| Guidance | line-of-sight along a path | NBV policy emitting a target pose |
| Low-level control | learned velocity/attitude policy | classical 6-DOF PID dynamic positioning |
| Perception input | simulator ground-truth depth | estimated depth from the camera, scale-anchored by DVL |

The PID replaced an earlier velocity-cascade design: purely proportional control
leaves a steady-state error under constant disturbances (buoyancy trim,
centre-of-buoyancy offset), and a viewpoint command requires settling *at* the
pose, not near it.

## 3. Decisions taken

- **Reward normalization (implemented, verified 2026-09-03).** Per-voxel
  geometric normalization so that reward magnitude does not depend on object
  size or water type. An information-gain form (NBUV log-gain) was considered
  and deferred.
- **Geometry.** Camera range 1.0–2.5 m, 10 cm voxels, 20³ grid — resized for a
  tank and for the depth error budget of a monocular estimator.
- **Object diversity.** Google Scanned Objects converted to USD (≈ 790 usable
  after filtering flat objects); rendering moved to `TiledCamera` so that
  every environment can carry a different asset.
- **Depth for deployment.** A monocular depth network (TRIDENT) with a
  per-scene scale anchor from the DVL (altitude and ego-motion). Measured in
  simulation: two-thirds of the depth error is global scale/shift, which the
  anchor removes; the network's own uncertainty output was rejected as
  uninformative. The depth model is a swappable component, not a premise.
- **Position feedback in the tank.** DVL dead reckoning, no external tracking;
  missions are kept short so drift does not accumulate.
- **Sonar.** Not used for dense depth — tank reverberation is severe. Retained
  only as a possible scalar anchor.
- **Staging.** First retraining with fixed water and many objects; water DR is
  added afterwards, one variable at a time (the same discipline that separated
  plant from policy in `step_2`).

## 4. Pipeline

```
P0  assets · normalization · geometry                    done / in progress
P1  privileged teacher retraining (objects + normalization + geometry)
P2  gate: simulation acceptance → depth-estimator qualification
P3  (conditional) teacher–student distillation
P4  TorchScript export · ROS 2 nodes (depth+anchor / TSDF / policy) · Gazebo sim-to-sim
P5  tank: calibration → shakedown → scripted baselines → policy → offline coverage evaluation
```

The TSDF node must import the fusion code from `env_reward`, not copy it — a
lesson from a vendored-copy drift bug in `step_2`.

## 5. Known risks

- DVL dead-reckoning drift is the largest open risk; mitigated only by mission length.
- Monocular depth has not been measured in real tank water.
- Object diversity per training step is bounded by `min(num_envs, pool size)`.
