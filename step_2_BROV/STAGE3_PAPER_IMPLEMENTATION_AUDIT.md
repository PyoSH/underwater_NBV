# Stage 3 — Sim2Swim 논문 구현 감사와 시간 제한형 재학습 사양

작성 기준: 2026-08-17  
대조 논문: [Sim2Swim: Zero-Shot Velocity Control for Agile AUV Maneuvering in 3 Minutes, arXiv:2512.08656v2](https://arxiv.org/html/2512.08656)

## 1. 결론

현재 코드는 **Sim2Swim의 제어 구조를 상당 부분 올바르게 구현했지만, 논문 충실 재현이라고 부를 수는 없다.** 가장 큰 이유는 논문 §4.5의 desired-state 생성 의미가 반대로 구현됐기 때문이다.

- 논문의 Eq. 9 `v(t)=[a,b sin(ωt),c cos(ωt)]`는 **Frenet–Serret frame을 따라 시간변화하는 목표 자세 `q_d(t)`**를 만드는 궤적이다.
- 목표 body velocity `v_d^b`는 이 곡선과 별개로 episode마다 단위구에서 방향을 뽑으며, 크기는 정확히 `0.5 m/s`다.
- 현재 코드는 Eq. 9를 `v_d^b(t)` 자체로 사용하고, `q_d`는 episode 동안 고정한다 (`envs/vel_env.py:287-309,364-366`). 현재 명령 속력은 5 s 동안 `0.5831–0.6733 m/s`다.

따라서 기존 `model_299.pt`는 논문이 설명한 MDP가 아니라 **다른 desired-state MDP에서 학습한 policy**다. 이는 논문 정합 reference를 새로 만들어야 한다는 근거이지만, 이 불일치가 현재 jitter의 직접 원인이라는 증거는 아니다. Stage 2에서 GT feedback에도 `-1↔+1` pitch action jitter가 남은 결과는 새 deploy policy를 평가할 필요를 보이지만, actor·observation·action mapping·plant의 폐루프 원인은 별도 ablation으로 분리해야 한다.

다만 두 종류의 변경을 구분한다.

1. `paper_ref_v1`: 논문이 공개한 MDP를 가능한 범위에서 충실하게 구현한 reference.
2. `deploy_v2`: `paper_ref_v1`의 수정된 core에 0.1/0.5 m/s, hold, stop/restart, 180° 반전과 saturation 억제를 추가한 Case-A 배포용 policy.

`0.1 m/s curriculum`, action smoothness penalty, runtime observation parity는 논문에 쓰인 항목이 아니라 **현재 sim2sim/sim2real gap을 줄이기 위한 프로젝트 확장**이다. 이를 “논문 미구현분”으로 부르지 않는다.

### 1.1 Jitter에 대한 인과 경계

- **확인됨:** perfect-state에 가까운 Gazebo GT feedback에서도 steady straight 구간의 pitch action이 tick마다 `-1↔+1`로 왕복했다. 따라서 DVL/EKF 오차는 jitter의 필요조건이 아니다.
- **확인됨:** DVL-EKF는 같은 frozen actor에서 velocity/depth error와 force clamp를 추가로 악화했다. estimator는 증폭 요인이다.
- **아직 미확정:** Eq. 9 오구현, reward 민감도, observation/integral 계약, action frame 또는 Gazebo plant 중 무엇이 GT 폐루프 limit cycle의 주원인인지는 분리되지 않았다.
- **단독 원인 가능성이 낮음:** mass DR 누락과 Water Linked noise 모델은 GT-feedback nominal Gazebo에서도 발생한 steady jitter를 단독으로 설명하지 못한다.

따라서 “논문 정합 구현”과 “jitter 원인 수정”은 같은 작업이 아니다. 재학습 전 최소 인과 gate는 (1) 동일 GT trace의 Isaac/runtime observation·action golden replay, (2) 같은 Gazebo plant에서 deterministic model-based controller와 frozen RL의 비교다. model-based controller도 흔들리면 allocation/plant부터 수정하고, model-based는 안정하지만 RL만 흔들릴 때 학습 MDP·reward·policy를 수정한다.

## 2. 구현 상태 분류

### 2.1 잘 구현된 부분

| 논문 요소 | 현재 구현 | 판정 |
|---|---|---|
| 저수준 velocity/attitude policy와 외부 3D LOS 분리 | `BROVVelEnv`와 `LOSGuidance`를 분리하고 평가 시 guidance를 attach | 구조 일치 |
| 16-D observation | `[q_e(4),v_e^b(3),omega^b(3),z_v(3),z_q(3)]` | Eq. 3의 차원·순서 일치 |
| integral observation | velocity/quaternion-vector error를 policy dt로 적분 | 논문 핵심 기여 반영 |
| 6-D action | `[-1,1]` force/torque action을 `F_max`로 scale하고 별도 thrust allocation | Eq. 4의 구조 일치 |
| reward 식 | Eq. 5–8의 exponential 항과 `0.4/0.2/0.05/0.3` 가중치 | 공개 수식과 일치 |
| episode | 5 s | 논문과 일치 |
| 학습기 | Isaac Lab, RSL-RL PPO, 2-hidden-layer MLP | 공개 구조와 일치 |
| volume 및 CB–CM offset DR | uniform volume과 구 내부 균등 CB offset | 논문 DR의 일부 구현 |
| actuator nominal model | T200 deadband, 정/역 비대칭, force saturation, 50 ms lag | 논문보다 상세한 nominal plant |
| Fig. 4 시험 틀 | 직선 왕복, ballast square, random-attitude square | 시나리오 구조 구현 |

근거: `envs/vel_env.py:311-340`, `envs/vel_env_cfg.py:53-80`, `agents/rsl_rl_ppo_cfg.py:31-71`, `robots/dynamics/brov2/thruster.py:90-217`, `test_policy.py:11-26,124-196`.

### 2.2 부분적으로 구현된 부분

| 항목 | 현재 상태 | 부족한 부분 |
|---|---|---|
| desired states | Eq. 9 계수와 `ω` 값만 사용 | Eq. 9의 역할이 velocity로 뒤바뀌었고 Frenet–Serret `q_d(t)`가 없음 |
| 0.5 m/s command | 평균적으로 비슷한 크기의 곡선 명령 | 단위구 방향과 exact norm `0.5`가 아님 |
| DR | volume, CB offset, 회전 added mass | 논문이 명시한 mass uniform DR가 없음 |
| massively parallel training | CLI로 2048 가능, 기존 default/run은 512 | 논문은 2048; 기존 run은 512와 강제 debug draw 사용 |
| MLP | 2 hidden layers, `[64,64]`, ELU | 논문은 width와 activation을 공개하지 않아 exact 여부 확인 불가 |
| observation contract | 16-D 수식은 맞음 | Isaac과 `brov_ros2`의 quaternion hemisphere, integral clamp/dt/reset/stale 규칙이 다름 |
| action contract | 6-D scale 및 pseudo-inverse allocation | FLU/Z-up wrench와 SNAME/FRD `B` 사이 `T6`가 명시되지 않음 |
| ballast test | volume 감소와 lateral CoB offset | 실제 `+600 g` mass/inertia/CM 이동이 아님 |
| validation | Fig. 4 형태 plot과 평균값 | 자동 pass/fail, axis action, best-checkpoint selection, artifact manifest가 없음 |
| Water Linked sim path | no-GPS `VISION_POSITION_DELTA→EKF3→LOCAL_POSITION_NED` | 실제 mount, saved orientation, packet timing/noise/FOM/dropout은 미확인 |

### 2.3 미구현 또는 잘못 구현된 핵심

1. Eq. 9 Frenet–Serret time-varying desired attitude.
2. episode별 exact `||v_d^b||=0.5 m/s` unit-sphere sampling.
3. mass domain randomization.
4. Isaac/runtime observation golden contract.
5. Isaac/runtime action-frame 6-axis golden contract.
6. machine-readable validation 기반 checkpoint selection과 artifact lineage.

### 2.4 논문만으로 결정할 수 없는 부분

논문은 다음 값을 공개하지 않았다.

- 2-layer MLP의 hidden width와 activation
- PPO horizon, batch/minibatch와 대부분의 PPO hyperparameter
- mass/volume/CB randomization 범위
- `K`의 수치값과 정확한 control rate
- integral clamp/reset/stale 규칙
- Frenet–Serret 축 convention, random initial-condition 결합 수식
- 3D LOS의 세부 수식

따라서 이 값에는 “논문 exact”가 아니라 **paper-disclosed-contract에 대한 문서화된 해석**이라는 라벨을 사용한다.

현재 25 Hz에서 reward의 step 최대는 `1.35`, 5 s의 이론적 episode return 최대는 `168.75`다. 논문이 보고한 final mean reward `315`는 같은 scale로 비교할 수 없다. 이는 논문의 미공개 control rate 또는 집계 방식이 다르다는 정황이지, `315`를 이번 학습 gate로 맞춰야 한다는 뜻이 아니다.

## 3. 논문 정합 desired-state 최소 구현

논문이 정확한 frame 수식을 공개하지 않았으므로 아래를 명시적 구현 가정으로 둔다.

```text
v_curve(t) = [a, b sin(omega t), c cos(omega t)]
dv_curve(t) = [0, b omega cos(omega t), -c omega sin(omega t)]
T = normalize(v_curve)
B = normalize(T cross dv_curve)
N = B cross T
R_FS = [T N B]
R_d(t) = R_initial R_FS(0)^T R_FS(t)
```

별도로 episode reset에서:

```text
s ~ uniform direction on S^2
v_d^b = 0.5 s
```

필수 unit test:

- 모든 sample에서 `abs(||v_d^b||-0.5)<=1e-6`
- `q_d(0)`가 random initial target과 일치
- `q_d(t)`가 episode 중 실제로 변화
- 인접 quaternion은 `dot(q_d[k],q_d[k-1])>=0`로 연속 표현
- `R_FS` orthonormality와 determinant `+1`
- CPU/GPU batch 결과의 finite 여부와 fixed-seed 재현성

## 4. 재학습 전에 반드시 고칠 계약

### 4.1 Observation v2는 우선 16-D 유지

첫 candidate에서 19-D로 확장하지 않는다. 현재 runtime과 같은 16-D를 유지하면서 다음을 공용 수식으로 고정한다.

- shortest-rotation quaternion hemisphere
- integral clamp `[-5,5]`
- valid new sample에서만 적분
- reset/re-prepare/session change에서 integral reset
- stale/duplicate sample에서 integral freeze
- 학습의 nominal `dt=0.04 s`, runtime의 bounded source `dt`를 같은 state transition 함수로 검증

Isaac trace와 `brov_ros2` replay의 observation max error가 `1e-5`를 넘으면 학습을 시작하지 않는다.

### 4.2 Action frame

새 policy 계약은 다음으로 고정한다.

```text
policy action: FLU/Z-up [Fx,Fy,Fz,Mx,My,Mz]
  -> scale [85,85,120,26,14,22]
  -> T6=diag(1,-1,-1,1,-1,-1)
  -> SNAME/FRD allocation B+
  -> per-thruster force clamp
  -> T200 inverse/PWM
```

Isaac과 `brov_ros2`를 함께 바꾸고 `±surge/±sway/±heave/±roll/±pitch/±yaw` basis test를 통과시킨다. 기존 `model_299.pt`는 legacy runtime에서만 baseline으로 보존한다.

### 4.3 Mass DR

논문은 범위를 공개하지 않았다. 실제 장착 상태 질량 범위를 확보하기 전 최소 project assumption은 nominal `14.635 kg`의 `±5%`, 즉 `[13.90325,15.36675] kg`이다. 이는 논문의 600 g ballast가 약 5% mass 변화라는 점에 근거한 **프로젝트 선택값**으로 manifest에 기록한다. PhysX mass와 관성의 일관성을 unit test하고 기존 volume/CB DR와 함께 사용한다.

## 5. `paper_ref_v1`과 `deploy_v2`

### 5.1 `paper_ref_v1`

- 16-D corrected observation contract
- corrected explicit action frame
- analytic Frenet–Serret `q_d(t)`
- exact 0.5 m/s unit-sphere `v_d^b`
- 논문 Eq. 5–8 reward 그대로
- 5 s episode
- uniform mass/volume, sphere CB–CM offset
- 2048 environments, headless

목적은 논문 공개 계약을 구현한 immutable reference를 남기는 것이다. 이 profile의 성능이 Case A에 충분하다고 미리 가정하지 않는다.

### 5.2 `deploy_v2`

`paper_ref_v1` core는 유지하고, 다음만 추가한다.

- episode 중 한 번의 command transition을 2–3 s 구간에 배치
- balanced bins: hold `0`, low `0.10`, cruise `0.50 m/s`
- stop, restart, exact 180° reversal
- attitude command는 FS trajectory와 runtime형 step/rate-limited command를 혼합
- velocity reward를 tolerance 기준으로 정규화: 우선 `sigma_v=0.10 m/s`
- 최소 `delta-action` 및 per-thruster clamp/allocation-residual penalty
- PPO rollout `64→128` step: 5.12 s로 episode와 한 transient를 포함

15–30 s training episode, 19-D observation, recurrent policy와 대규모 reward sweep은 첫 candidate에 넣지 않는다. 60–120 s hold/왕복은 validation으로 먼저 확인하고 장기 integral failure가 있을 때만 긴 episode fine-tune을 한다.

## 6. 학습 속도 개선 — 완료

기존 `model_299.pt` run의 TensorBoard 실측:

| 항목 | 기존 |
|---|---:|
| env / rollout / iteration | 512 / 64 / 300 |
| policy transitions | 9,830,400 |
| 전체 wall time | 16,157.5 s = 4.49 h |
| collection time 중앙값 | 54.161 s/iteration |
| PPO learning time 마지막 | 0.041 s/iteration |
| FPS 중앙값 | 2,417.5 |

병목은 PPO가 아니라 headless에서도 매 step 실행된 debug draw였다. 다음 변경을 완료했다.

- `BROVVelEnvCfg.debug_vis=False`
- debug-draw extension과 resource lazy initialization
- `draw_lines()` 전체 guard
- render interval을 physics 100 Hz가 아니라 policy 25 Hz에 맞춤
- `test_policy.py`에서만 명시적 `--debug_vis`로 opt-in
- CLI seed를 learner와 environment 양쪽에 적용
- rollout horizon과 save interval CLI override 추가

수정 후 2-iteration capacity smoke:

| 설정 | 안정 iteration collection | FPS | 결과 |
|---|---:|---:|---|
| 512 env, 64 step | 0.653 s | 47,318 | PASS |
| 2048 env, 64 step | 0.675 s | 182,362 | PASS, OOM 없음 |

512 기준 collection은 약 **83배** 빨라졌다. 이는 아직 잘못된 기존 MDP의 capacity smoke이며 policy 품질 시험은 아니다.

추가로 `32 env × 8 step × 1 iteration`에서 새 `--seed`, `--num_steps_per_env`,
`--save_interval` 옵션을 함께 사용해 `model_0.pt`와 exported state dict 생성까지 확인했다.
임시 smoke artifact는 컨테이너 `/tmp`에만 두었으며 재학습 artifact로 사용하지 않는다.

구현 위치: `envs/vel_env_cfg.py:38-50`, `envs/vel_env.py:115-164`, `test_policy.py:56-60,310-315`, `train.py:37-55,70-92`.

## 7. 시간 제한형 실행 순서

### P0 — 완료

1. Stage 2 fresh GT/DVL-EKF full-cycle baseline 고정.
2. training visualization 완전 비활성 및 2048 capacity 확인.
3. seed/horizon/checkpoint CLI 준비.

### P1 — 코드 구현 후 학습 전 gate

1. desired-state generator 수정 및 unit test.
2. observation 16-D parity와 q/-q golden test.
3. explicit `T6`와 6-axis allocation golden test.
4. mass DR와 mass/inertia consistency test.
5. axis action, achieved wrench, clamp metric logger.
6. config/commit/seed/checkpoint SHA manifest.

하나라도 실패하면 학습을 시작하지 않는다.

### P2 — 3-iteration capacity smoke

시간 제한상 `paper_ref_v1`과 `deploy_v2`를 각각 장시간 학습하지 않는다. 공용 core를
한 번 구현한 뒤 `paper_ref_v1`은 unit/golden test와 3-iteration contract smoke로 보존하고,
실제 full candidate 예산은 Case-A command transition을 포함한 `deploy_v2`에만 사용한다.

```bash
docker exec -it -w /workspace/OceanRL_test/step_2_BROV isaac-lab-base \
  /isaac-sim/python.sh train.py \
  --headless \
  --num_envs 2048 \
  --num_steps_per_env 128 \
  --max_iterations 3 \
  --save_interval 1 \
  --seed 42 \
  --experiment_name stage3_deploy_v2_smoke
```

Smoke gate:

- NaN/Inf, non-timeout reset 0
- desired speed norm `0.5±1e-6`인 paper episodes
- `q_d` time variation과 quaternion continuity
- checkpoint save/reload/export 성공
- warm iteration collection time `<=2.0 s` 잠정 기준

### P3 — 첫 candidate

2048×128×50은 `13.11M` transitions로 기존 full run의 `9.83M`보다 많고, 현재 GPU capacity에 여유가 있다.

```bash
docker exec -it -w /workspace/OceanRL_test/step_2_BROV isaac-lab-base \
  /isaac-sim/python.sh train.py \
  --headless \
  --num_envs 2048 \
  --num_steps_per_env 128 \
  --max_iterations 50 \
  --save_interval 5 \
  --seed 42 \
  --experiment_name stage3_deploy_v2_seed42
```

50 iteration에서 validation이 계속 개선 중이면 100까지 resume한다. 첫 seed가 구조적으로 실패하면 seed를 추가하지 않고 config를 고친다. 경계값의 10% 이내에서만 stochastic variance가 의심될 때 seed 43을 한 번 추가한다.

## 8. checkpoint 선택과 승인 gate

Training reward가 아니라 다음 고정 validation score로 checkpoint를 선택한다.

### Isaac

- hold에서 축별 불필요한 action bias `abs(mean(action))<=0.05`
- 0.10/0.50 m/s straight steady velocity RMSE와 cross-speed
- 180° reversal 뒤 3 s 이내 재정착
- steady actor bound `<5%`, `-1↔+1` one-tick flip 0
- per-thruster clamp `<5%`
- 60 s에서 integral clamp dwell/growing oscillation 없음

### Gazebo fresh Case A

GT feedback 1회가 먼저 통과한 policy만 DVL-EKF 3회를 수행한다.

- exact waypoint RLE `[0,1,2,1]`
- outbound와 return 각각 vector RMSE: GT `<=0.08`, DVL-EKF `<=0.12 m/s`
- cross-speed RMS: GT `<=0.05`, DVL-EKF `<=0.08 m/s`
- depth RMS: GT `<=0.03`, DVL-EKF `<=0.10 m`
- whole-cycle actor bound: GT `<=10%`, DVL-EKF `<=20%`
- force clamp: GT `<=5%`, DVL-EKF `<=10%`
- fault, NaN, reset, collision 0

GT가 통과하고 DVL-EKF만 실패하면 추가 재학습을 중단하고 estimator/Baro/DVL feedback으로 돌아간다. GT부터 action jitter가 남으면 policy/action/plant 쪽 문제로 판정한다.

## 9. 이번 일정에서 미룰 항목

- 19-D observation 또는 recurrent policy
- 15–30 s training episode
- full Water Linked acoustic beam/range-mode/mount/noise/FOM/dropout model
- current/tether/single-thruster-failure DR
- network width/activation/PPO sweep
- AprilTag/localization 변경
- mid-training video와 GUI rendering
- Case B/C 최종 승인과 100-seed 통계
- Gazebo legacy plant에 맞춘 Isaac nominal 왜곡

실제 Water Linked settings와 raw packet을 확보하기 전 임의 sensor DR를 “sim2real parity”로 부르지 않는다. Stage 3의 첫 목표는 common-mode policy/action jitter를 제거하고, estimator-only residual은 Stage 2 진단과 분리하는 것이다.
