# BROV 정책 재학습 및 sim2sim 재검증 분석

작성 기준: 2026-08-16

sim2real bag 보완: 2026-08-17 — [`CASE_A_FAULT_20260814_170757_ANALYSIS.md`](CASE_A_FAULT_20260814_170757_ANALYSIS.md)

대상: `step_2_BROV` IsaacLab 학습, Edo Gazebo/ArduSub SITL, `brov_ros2` sim2real runtime

## 1. 결론

현재 배포 정책은 **실기체 승인 정책으로 간주하면 안 되며, 기존 문서에 기록된 미구현 항목을 보완해 재학습한다.** Isaac 0.10/0.60 frozen-policy 시험은 재학습 GO/NO-GO가 아니라 전후 비교 baseline이다. 근거는 다음과 같다.

1. `step_2_BROV` 문서는 mass DR과 주기적 mid-training validation이 미구현이라고 명시한다 (`CLAUDE.md:211-225`). 같은 절의 “본 규모 학습 미실행”은 사용자가 실제 학습을 수행한 현재 상태와 맞지 않는 과거 기록이므로 재학습 근거로 사용하지 않는다.
2. 학습 관측에는 quaternion hemisphere 고정과 적분 clamp가 없지만, 현재 `brov_ros2` runtime에는 둘 다 있다 (`envs/vel_env.py:299-311`, `brov_base/brov_base/observation.py:120-138`). 현재 정책은 학습 때와 다른 관측 분포로 실행된다.
3. 사용자 확인 기준 Isaac 학습 운용 속도는 약 0.58–0.63 m/s인 반면, `deploy/*` 기반 Gazebo와 수조 Case A는 0.10 m/s였다. 학습에는 0.10 m/s, stop·180° reversal과 장시간 운용이 없다 (`envs/vel_env_cfg.py:43-62`, `envs/vel_env.py:290-297`, `SIM2SWIM_DEMO.md:47-89`).
4. 학습의 6축 action과 SNAME/FRD allocation 사이 frame 변환이 명시되지 않았다. 같은 누락이 policy runtime에도 있어 end-to-end로 우연히 맞을 수 있지만, action label 자체는 검증된 계약이 아니다 (`envs/vel_env.py:101-143`, `brov_control/brov_control/policy_node.py:213-221`).
5. Edo Gazebo plant는 IsaacLab과 질량, 관성, 복원모멘트, added mass, damping, 추진기 곡선이 다르다. 다만 사용자가 `step_2_BROV/deploy/*` 기반 stack을 0.10 m/s로 실행해 sim2sim했으므로 “실행 경로가 없다”고 단정하지 않는다. 부족한 것은 그 실행의 launch argument, policy/config SHA, Gazebo GT·EKF trace와 재현 manifest다. Track A/B 결과는 별도의 고전 GNC 결과다.

단, **AprilTag one-shot pool localization 자체는 재학습 사유가 아니다.** Vision은 16-D 정책 입력을 바꾸지 않고 외부 `pool -> odom` 정렬과 waypoint 변환만 담당한다고 문서에 명시되어 있다 (`SIM2SWIM_DEMO.md:31-43`, `ARCHITECTURE.md:37-54`). 최신 Case A bag에서 vision-vs-aligned odometry residual이 증가했으므로 최종 full-stack safety gate에서는 보완해야 하지만, 현 최소 재학습의 blocker로 두지 않는다.

### 시간 제약을 반영한 결론

사용자 관측상 Gazebo와 실기 모두에서 자세는 대체로 추종했고, 실기 P2의 방향전환도 성공했다. 공통 실패는 0.10 m/s 선속도 제어였고 position 표류는 그 downstream 결과다. 실기 bag은 사용자가 이 문제를 보고 수동 종료한 `OPERATOR_ABORT(TRACKING_CONCERN)`으로 해석한다.

현 `policy.pt`는 zero-error synthetic observation에서도 sway action 약 `+0.232`를 내고, `v_e.x=-0.10 m/s`에서 surge/sway action이 약 `+0.207/+0.241`이다. 정책은 position을 직접 보지 않으므로, 새 position policy를 만들기보다 저속 velocity-policy OOD·bias·reward를 먼저 확인한다. 학습 reward도 quaternion 0.4 항이 두 번 들어가고 velocity는 0.2이므로, 자세 추종이 속도 추종보다 잘 보인 관측과 정합한다.

2026-08-17 `model_299.pt`를 Isaac 0.10 m/s에서 60 s 직접 실행한 결과 평균 velocity error 0.0074 m/s, 마지막 2 s 0.0024 m/s였고, 45.96 s의 180° 전환 뒤에도 지속 요동이 없었다. checkpoint actor와 deploy/sim2real TorchScript actor tensor도 exact equality였다. 따라서 저속 OOD 단독 원인 가설은 약해졌고, 다음 최우선은 같은 policy의 Gazebo GT-feedback/deploy-feedback A/B다. 이후 기존 P0/P1의 `observation parity → 저속·hold·stop·reversal command → velocity-priority reward → 15–30 s episode/horizon → action 계약 고정 → 자동 validation/artifact manifest → 재학습 → Gazebo Case A 3회`를 수행한다. 재학습은 정확도·robustness 보완 목표로 유지하지만 현재 요동을 덮는 수단으로 사용하지 않는다.

### 재학습과 시스템 변경 의사결정 요약

| 변경/문제 | 재학습 판단 | 필요한 조치 |
|---|---|---|
| AprilTag/pool one-shot 정렬만 추가 | 불필요 | localization 정확도와 reset/freshness는 시스템 시험으로 검증 |
| `q_e.w >= 0`, 적분 clamp·실측 `dt` | 필요 | 학습과 runtime의 관측 수식을 단일화한 뒤 새 정책 학습 |
| depth outer-loop와 terminal hold | 강하게 권고 | 같은 guidance를 양 simulator와 runtime에서 사용하고 해당 명령 분포로 학습/평가 |
| action↔wrench↔thruster frame/sign 수정 | 필요 | 기존 정책을 보존하려고 runtime만 고치지 말고 계약을 고정한 뒤 재학습 |
| 센서 지연·dropout, 추진기 지연/비대칭, mass·damping DR 추가 | 필요 | domain randomization 및 fault curriculum에 반영 |
| RCPassThru에서 custom ArduPilot mode로 변경 | 조건부 | 입력 단위/frame/rate/mixer가 동일하면 재검증, 달라지면 재학습까지 수행 |

## 2. 증거 범위와 해석 한계

이 문서는 다음 세 종류를 구분한다.

- **문서 사실:** 각 저장소의 Markdown에 명시된 상태, 운용 envelope, 알려진 제한
- **코드 대조:** 관측, frame, 모델 계수, timing 구현의 실제 차이
- **권고 기준:** 새 실험에서 사전에 고정할 gate와 수치. 기존 결과로 오해하지 않도록 별도로 표시한다.

`brov_ros2`의 Markdown 20개는 주로 runbook/architecture/roadmap이며, 정책 성공률, tracking RMSE, overshoot, settling time을 담은 정량 sim2real 결과 보고서는 없다. 따라서 문서만으로 실기 실패의 단일 원인을 확정할 수 없다. 가장 직접적인 정책 상태 문구는 포함 정책을 quaternion 및 depth-guidance 수정 후 실기체에서 재검증해야 한다는 것이다 (`docs/DEMO_RUNBOOK.md:113-114`).

과거 실기 시험 기록은 현재 checkout 파일이 아니라 `origin/main`의 Git object에 남아 있다. 이 기록에서는 다음을 직접 확인했다 (`origin/main:step_2_BROV/deploy/WORKLOG_2026-08-13.md:22-81,168-196,198-269`).

- 약 25 Hz 실 observation에서 정책 action은 `[surge +0.207, sway +0.308, heave +0.063, roll -0.028, pitch -0.484, yaw +0.055]`였다.
- 완전 무오차 합성 관측에서도 sway action 약 `+0.232`가 남았고, pitch 오차 ±5.14°에 pitch action 약 ±0.48이 발생했다.
- T1~T8 전기 채널은 direct였고 실제 운동의 T2/T3/T8 반전 마스크는 코드와 일치했다. 따라서 그 시험 범위에서 단순 채널 mapping은 주원인에서 제외됐다.
- 실제 `q_e=[-0.9988,-0.0077,+0.0458,+0.0148]`가 관측되어 quaternion hemisphere 불연속을 확인했다.
- 수직 추진기는 동작했지만 상승 후 하강했다. 기록은 원인을 수평 LOS가 depth error를 압도한 것과 terminal 목표속도를 0으로 고정한 구조로 분리했고, 이후 independent depth/terminal hold를 추가했다.

따라서 기존 실험은 “정책만 나빴다”가 아니라 **정책의 zero-state bias/과도 gain, 관측 불연속, guidance 분포, 실제 ballast·tether 영향**이 함께 있었음을 보여준다. 새 시험에서는 이 항목을 별도 trace로 측정해야 한다.

현재 policy artifact는 두 저장소에서 같은 파일이다.

```text
step_2_BROV/deploy/exported/policy.pt
brov_ros2-main/artifacts/policies/demo_policy/policy.pt
SHA-256 = 0d89f3270f46214f1569b7d48dcb5e25363b1d9b7353b82ced0fc67c0093a472
```

vehicle YAML도 동일 SHA를 사용한다. 이는 **배포 파일 동일성**은 증명하지만, 어느 checkpoint·학습 commit·seed·DR 설정으로 만들었는지는 증명하지 않는다. 현재 metadata에는 `(1,16) -> (1,6)`, action order, vehicle hash와 observation fix 후 재검증 필요만 기록되어 있다 (`artifacts/policies/demo_policy/metadata.yaml:1-22`).

또한 `DEMO_RUNBOOK.md:113-114`는 quaternion과 depth-guidance 수정이 metadata에 기록됐다고 표현하지만, 실제 metadata `notes`에는 quaternion/observation 재검증만 있고 guidance version이나 depth-hold 변경은 없다. 새 artifact manifest에는 guidance contract/version도 포함해야 한다.

Edo 저장소 단독에는 `policy.pt`, TorchScript, 16-D observation 또는 policy runner 통합 경로가 없지만, 실제 sim2sim은 `step_2_BROV/deploy/*` 기반 stack을 Gazebo와 연결해 0.10 m/s로 수행했다. **현 checkout의 핵심 부족분은 실행 경로의 부재가 아니라, 그 run을 동일하게 재현할 launch/config manifest와 GT·EKF trace의 부재**다. 다음 sim2sim에서는 실행 인자, policy/config SHA와 최소 trace를 남긴다.

## 3. 현재 전달 경로에서 끊어진 계약

```text
IsaacLab state/command
  -> observation -> policy action -> Z-up wrench 의도
  -> SNAME allocation -> T200 model -> Isaac plant
                         |
                         +-> exported policy.pt
                              -> brov_ros2 observation/guidance
                              -> SNAME allocation -> PWM -> ArduSub/실기체
                              -> Gazebo SITL adapter -> Gazebo plant
```

동일해야 하는 경계는 다섯 개다.

1. 상태와 목표에서 16-D 또는 새 observation을 만드는 수식
2. action 각 축의 frame, 부호, 단위와 scale
3. allocation 열 순서 및 추진기별 정/역 방향
4. PWM에서 실제 추력까지의 deadband, 비대칭, 포화, 지연
5. source timestamp, observation 주기, stale/dropout 처리

현재는 이 경계 중 어느 것도 Isaac–Gazebo–ROS 전체에 대한 golden-vector 시험으로 고정되어 있지 않다.

## 4. 핵심 gap 분석

### 전체 로드맵 P0 — 최종 일반화 전에 닫을 항목

| 항목 | 근거와 영향 | 수정 및 재확인 |
|---|---|---|
| Quaternion 표현 | 학습은 `q_e = conjugate(q_d) ⊗ q`를 그대로 쓰고 runtime은 `w>=0`로 canonicalize한다. 같은 자세 `q/-q`가 학습에서는 다른 숫자다. | 공용 순수 함수로 만들고 `q`와 `-q`가 같은 obs/action을 내는 golden test 추가 |
| 적분 상태 | 학습은 고정 0.04 s로 무제한 누적하고 runtime은 실측 `dt`와 ±5 clamp를 쓴다. 5 s 학습과 60 s 운용에서 상태 분포가 달라진다. | clamp, reset, stale 시 freeze, waypoint 전환 시 처리 규칙을 하나로 고정 |
| 관측의 정보 부족 | 현재 관측에는 `v_e`만 있고 `v_d` 또는 현재 `v`가 없다. `v=0,v_d=0`과 항력을 상쇄하며 `v=v_d!=0`인 상태가 같은 zero error로 보일 수 있다. | 권장 `observation_v2`에 `v_d_body` 3개를 추가. actuator memory가 중요하면 previous action/actuator state도 추가 |
| Action frame | `B`는 SNAME/FRD 위치·방향으로 만들지만 Z-up action을 변환하지 않고 넣는다. thruster output은 다시 Z-up으로 변환된다. runtime policy path도 동일하며 model-based path만 explicit 변환한다. | `wrench_zup -> T6 -> wrench_sname -> B+` 경계를 명시하고 6축 basis test 후 새 정책 학습 |
| Guidance 동등성 | 구형 Isaac LOS는 3D 벡터를 한 번에 정규화하고 terminal hold가 없다. runtime은 수평 lookahead가 depth error를 압도하지 않도록 독립 depth hold와 terminal outer-loop를 쓴다 (`guidance.py:559-581`). | runtime guidance를 simulator에서 그대로 재사용하거나 같은 golden vectors로 검증 |
| Edo 재현 경로 | 현재 Track B loop는 RL runtime이 아니며, policy artifact/관측 builder 통합도 없다. | 실제 `brov_ros2` launch/config/policy를 SITL에 연결하는 adapter를 버전 관리 |
| Edo frame·timing | Track B allocator는 Gazebo raw FLU/Z-up 형상에 NED wrench를 섞고 `Mz`만 경험적으로 반전한다. IMU와 fake DVL frame도 같은 축인지 입증되지 않았다. 명목 20 Hz loop는 로그상 약 3.5 Hz다. | DVL frame 및 6축 allocation을 먼저 수정하고, subprocess polling 대신 비동기 25 Hz runtime 사용 |

### P1 — 재학습 사양에 포함할 항목

| 항목 | 현재 상태 | 필요한 보완 |
|---|---|---|
| 명령 분포 | 학습은 `[0.5, 0.5 sin, 0.3 cos]` 중심이며 zero/저속/terminal hold가 없다 (`envs/vel_env.py:290-297`). | zero hold, 0.05/0.10 m/s, ramp, reversal, corner, depth step, 외란 복귀 포함 |
| 자세 분포 | 기존 평가 Case C는 roll/pitch ±90°, yaw ±180°의 즉시 변경이다. 실기 첫 시험은 ±15°/±30°, 10°/s slew다. | 보수 envelope부터 curriculum으로 확대하고 중간 episode setpoint 변경 포함 |
| episode | 학습 5 s, 실기 Case C 60 s, Case A는 수동 종료까지 왕복한다. | 5/30/60/120 s 혼합 rollout과 장시간 hold/anti-windup 평가 |
| Dynamics DR | volume ±10%, CoB 15 mm, 회전 added mass ±40%뿐이며 mass DR은 미구현이다 (`CLAUDE.md:142-148`). | mass/COM/inertia, 6축 added mass와 damping, current, tether, ballast 포함 |
| Actuator DR | nominal T200 모델만 사용하고 runtime Case C limiter는 학습에 없다. | gain 편차, deadband, 정/역 비대칭, time constant, battery, 추진기별 열화/고장, PWM/action limiter 포함 |
| Sensor/timing | YAML에 DVL 15 Hz·100 ms와 noise가 있으나 환경은 ground truth state를 직접 읽는다. | sample-and-hold, 비동기 IMU/DVL, bias/noise, jitter, latency, 0.2 s dropout과 장기 stale fault 포함 |
| 보상/종료 | action norm은 allocation 전 값을 보고, achieved wrench/saturation을 보지 않는다. 종료는 20 m bound와 timeout뿐이다. | tracking 외 saturation, slew, achieved-wrench residual, collision/depth/NaN/fault를 기록·반영 |
| Validation | 세 시나리오 플롯과 평균 속도오차만 있고 pass/fail 및 machine-readable summary가 없다. | 고정 validation set, DR holdout set, seed별 JSON/CSV와 자동 gate 추가 |

### P2 — 정책과 분리해 시스템 수준에서 검증할 항목

AprilTag/pool localization은 정책 입력 변경이 아니지만 다음 제한은 mission 결과를 오염시킬 수 있다.

- marker frame은 과거 가정보다 평면 내 180° 달랐고 현재 값으로 수정됐다 (`brov_perception/README.md:72-108`).
- camera extrinsic은 수중 hand-eye calibration이 아닌 nominal 값이며 neutral tilt에서만 유효하다 (`brov_perception/README.md:110-112`).
- ArUco 처리 backlog, decode-time timestamp, single marker ambiguity가 남아 있다 (`POOL_LOCALIZATION_RUNBOOK.md:394-421`).
- vision은 one-shot 이후 continuous fusion을 하지 않으므로 DVL 장기 drift를 고치지 않는다 (`SIM2SWIM_DEMO.md:440-456`).
- waypoint bounds는 선체 swept volume이나 tether geofence가 아니다.

이 항목 때문에 생긴 경로 오차를 저수준 policy의 학습 실패와 혼동하지 않아야 한다.

## 5. 후속 전체 로드맵 IsaacLab 재학습 사양

현 최소 재학습은 16-D를 유지한다. 아래 19-D·DR·장시간 사양은 최소 반복이 실패하거나 최종 일반화 단계에 들어갈 때 적용한다.

### 5.1 계약을 먼저 동결한다

권장 최소 관측은 다음 `brov_velocity_observation_v2` 19차원이다.

```text
[q_error_unique_wxyz(4), velocity_error_body_zup(3),
 angular_velocity_body_zup(3), velocity_error_integral_clamped(3),
 quaternion_vector_error_integral_clamped(3), desired_velocity_body_zup(3)]
```

- `q_error`는 항상 shortest-rotation hemisphere로 고정한다.
- 적분은 source timestamp의 유효한 새 sample에서만 진행한다.
- stale sample, fault, preview-only replay에서 적분과 waypoint 진행을 동결한다.
- reset, mission re-prepare, localization epoch/session 변경 시 적분을 초기화한다.
- limiter 또는 actuator memory가 성능에 중요하면 previous action 6개나 actuator state 8개를 추가하거나 recurrent policy를 쓴다. 이 경우 별도 schema version으로 올린다.
- 16-D를 유지하는 대안도 시험할 수 있지만, `v_e=0`의 명령 모호성은 구조적으로 남으므로 zero-state bias gate를 통과해야 한다.

Action 계약은 다음처럼 한 줄로 고정한다.

```text
policy action [-1,1], FLU/Z-up [Fx,Fy,Fz,Mx,My,Mz]
  -> scale [85,85,120,26,14,22]
  -> T6=diag(1,-1,-1,1,-1,-1)
  -> SNAME/FRD allocation B+
  -> per-thruster desired force [N]
  -> T200 inverse curve -> normalized PWM
```

frame 수정 전 정책과 수정 후 runtime을 섞어 쓰면 안 된다.

### 5.2 명령 curriculum

학습 batch에 다음을 모두 포함한다.

1. 완전 정지 및 terminal position/depth hold
2. 0.05, 0.10 m/s 저속과 기존 고속 명령
3. step, ramp, sine, 정/역전환, 정지 후 재출발
4. 수평 이동 중 depth 변경, 순수 상승/하강, depth 외란 후 복귀
5. Case A의 takeoff-then-align 및 loop
6. Case C의 roll/pitch ±15°, yaw ±30°, 10°/s slew, 1 s dwell
7. 점진적으로 넓힌 자세와 속도 envelope

### 5.3 DR와 현실적 I/O

- 물리: mass, COM/CoB, inertia, volume, 6축 added mass, linear/quadratic damping
- 환경: 정/측/수직 current, tether 방향·장력 근사, surface/bottom 근접, ballast
- 추진기: 위치/방향 오차, 개별 gain, forward/reverse 비대칭, deadband, 50 ms급 time constant, saturation, battery, 1개 추진기 열화/고장
- 센서: IMU/DVL bias·noise, DVL 15 Hz/100 ms, 비동기 timestamp, sample-and-hold, outlier/dropout
- 실행: 25 Hz nominal, jitter, observation-to-action latency, 0.2 s gap 회복, 장기 stale에서 neutral/fault

DR 범위는 임의 백분율로 끝내지 말고 Isaac/Gazebo 축별 plant identification과 실기 bag/bench 결과로 갱신한다. Gazebo legacy 모델은 nominal truth가 아니라 하나의 큰 OOD corner로 활용할 수 있다.

### 5.4 학습 및 validation

- 5 s 수렴 속도뿐 아니라 30~120 s 장기 rollout을 validation에 포함한다.
- reward component, 축별 RMSE, steady-state bias, action/PWM saturation, allocation residual을 매 iteration 기록한다.
- nominal validation과 학습 범위 밖 holdout DR validation을 분리한다.
- 최소 여러 seed의 독립 학습을 비교하고 best training reward가 아니라 validation gate로 checkpoint를 선택한다.
- export 시 checkpoint SHA, source commit, schema, normalization, DR config, seed, IsaacLab/RSL-RL 버전, TorchScript SHA를 함께 저장한다.

## 6. Gazebo sim2sim 재검증 gate

### Gate 0 — 순수 함수 및 golden-vector 동등성

- 같은 합성 상태/목표/timestamp sequence를 Isaac과 `brov_ros2` observation builder에 넣는다.
- 19개 또는 선택한 schema의 모든 성분이 float32 기준 `max_abs_error <= 1e-5`여야 한다.
- `q`와 `-q`, yaw ±π 통과, reset, clamp, stale freeze, waypoint 전환을 포함한다.
- 정책 파일 한 개에 같은 observation을 넣었을 때 action이 bitwise 또는 허용오차 내 동일해야 한다.

### Gate 1 — 6축/8추진기 mapping

- T1~T8 각각의 작은 정/역 impulse로 Gazebo의 `Δv_body`, `Δω_body`를 기록한다.
- surge/sway/heave/roll/pitch/yaw basis action을 각각 적용한다.
- 기대 부호와 다른 축이 하나라도 있으면 실패다. 특히 기존에 확인하지 않은 `Mx/My`를 포함한다.
- `B @ force`와 simulator에서 측정한 achieved wrench/가속도의 열 순서, cosine similarity, cross-axis coupling을 저장한다.

### Gate 2 — plant 동등성 또는 의도적 OOD 분리

현재 모델을 한 결과로 섞지 말고 두 profile로 분리한다.

- `gazebo_calibrated`: Isaac nominal과 mass/inertia/CoB/added mass/damping/T200 응답을 맞춘 회귀용 모델
- `gazebo_legacy_ood`: 기존 Edo 모델을 유지해 강한 model mismatch에서 robustness를 보는 모델

현재 주요 차이는 다음과 같다.

| 항목 | Edo Gazebo | IsaacLab |
|---|---:|---:|
| mass | 13.0 kg | 14.635 kg |
| inertia | `[0.26,0.23,0.37]` | `[0.289,0.329,0.337]` |
| CoB-COM z | 약 0.049 m | 0.010 m |
| added mass | 전 축 0 | `[6.36,7.12,18.68,0.189,0.135,0.222]` |
| linear damping | 전 축 0 | `[13.7,0,33,0,0.8,0]` |
| quadratic damping 절댓값 | `[58.42,55.137,124.818,4,4,4]` | `[141,217,190,1.19,0.47,1.5]` |
| thrust | PWM 1100/1500/1900 -> -50/0/+50 N 선형 | -51.5/+64.1 N 비대칭, deadband 0.075, tau 0.05 s |
| physics step | 1 ms | 10 ms |

근거: Edo `configs.yaml:1-44`, `model.sdf:52-79,440-529`; Isaac `robots/data/BROV2/brov2_heavy.yaml:3-55`.

Gazebo plugin을 바로 바꾸지 않는다면 sim2sim adapter가 T200 forward curve와 actuator lag로 desired N을 만든 뒤 Gazebo의 선형 `N <-> PWM`에 맞춰 변환해야 한다. 실기용 inverse PWM을 그대로 Gazebo에 보내면 서로 다른 plant를 시험하게 된다.

### Gate 3 — 정책 정적/동적 특성

- zero/hover observation의 6축 action bias
- velocity error ±0.05/±0.10/±0.30 m/s sweep
- roll/pitch/yaw error ±1/±2/±5° sweep의 부호, 연속성, 과도 gain
- 결합축 명령에서 saturation, deadband, achieved-wrench residual
- 60~120 s hold에서 적분 clamp 체류율과 drift

### Gate 4 — 실제 `brov_ros2` runtime으로 mission 회귀

Track B의 blocking CLI loop를 쓰지 않고, 실기와 같은 `brov_ros2` node, policy artifact, config, safety gate를 Gazebo SITL에 연결한다.

| 시험 | 고정 profile | 확인 항목 |
|---|---|---|
| Neutral hold | `v_d=0`, level, 60 s | action bias, drift, depth hold, integrator |
| Axis step | 6축 단독 저 authority | 부호, overshoot, settling, cross coupling |
| Case A short | 0.10 m/s, lookahead 0.40 m, reach 0.15 m | takeoff, depth, 직선 왕복 |
| Case A long | 0.8 m와 기존 2 m segment | 장기 drift, terminal/turnaround transient |
| Depth transition | 수평 이동 중 상·하향 목표 | independent depth hold와 heading 부호 |
| Case C v2 | 0.05 m/s, ±15°/±30°, 10°/s | 1 s dwell, 한 lap, limiter 개입 |
| DR corners | calibrated 및 legacy OOD, current/tether | seed별 success와 worst case |

Case C에는 문서의 exact envelope를 그대로 쓴다.

```text
action_abs_limit = [0.25,0.25,0.30,0.15,0.15,0.15]
pwm_abs_limit = 0.35
policy_pwm_slew = 0.40/s
attitude_tolerance = 10 deg
angular_speed_tolerance = 5 deg/s
dwell = 1.0 s
max_duration = 60 s, target = 1 lap
```

### Gate 5 — fault와 safety

- observation jitter/dropout/stale, DVL outlier/bottom-lock loss
- EKF unhealthy, localization epoch/session 변경, mission invalidation
- policy NaN/shape 오류, ROS node/container 종료, PWM publisher 중복
- estop, stop, communication loss, disarm 후 neutral 도달시간
- START 후 첫 PWM, 연속 PWM 간격과 0.25 s watchdog

Custom ArduPilot mode는 interface와 failsafe가 아직 전달되지 않은 roadmap 상태다. frame/unit/rate/mixer/timeout 계약과 SITL fault gate를 통과하기 전에는 검증된 RCPassThru 결과와 섞지 않는다 (`ACTUATION_BACKEND_ROADMAP.md:5-26,28-53,72-96`).

## 7. 승인 지표

### 문서에서 직접 고정되는 hard gate

- 정책 입력/출력 shape와 schema가 artifact metadata와 일치
- Case C action/PWM/slew 한도 위반 0회
- Case C 자세 10°, 각속도 5°/s 안에서 1 s dwell
- 25 Hz pipeline에서 연속 PWM 간격 0.25 s 초과 시 반드시 neutral/disarm fault
- NaN, 충돌, 수면/바닥 접촉, publisher 중복, 잘못된 축 부호 0회

### 새 실험 전에 확정할 제안 기준

기존 Markdown에는 성능 기준을 정할 정량 결과가 없으므로 아래 값은 **잠정 제안**이다. 먼저 bag baseline을 계산해 시험 책임자가 확정해야 한다.

- Case A: waypoint reach 0.15 m, 수평 leg의 takeoff·waypoint 전환 제외 depth RMSE 0.10 m 이하, cross-track RMSE 0.15 m 이하
- Case C: 60 s 안에 1 lap, 모든 waypoint dwell 승인
- tracking: 축별 RMSE뿐 아니라 p95/max, overshoot, settling time, steady-state bias를 모두 보고
- timing: observation/PWM rate, interval p50/p95/max, source age/skew, end-to-end latency를 보고
- actuator: action/PWM saturation과 slew limiter 개입률, deadband 체류율, allocation residual을 보고
- robustness: 고정 validation scenario에서 최소 100 seed를 실행하고 성공률 95% 이상을 잠정 목표로 사용

평균 하나로 승인하지 않는다. 각 실패 seed와 worst-case trace를 남긴다.

## 8. 최신 Case A 실기 bag 분석 결과

`brov_ros2/runtime/experiments`에는 세 bag이 있다.

```text
case_a_20260814_161032                 약 27.4 s
case_a_fault_20260814_165321           약 51.9 s
case_a_fault_20260814_170757           약 84.2 s
```

요청된 최신 bag `case_a_fault_20260814_170757`은 CDR을 직접 역직렬화해 분석했다. 상세 수식·타임라인·한계는 [`CASE_A_FAULT_20260814_170757_ANALYSIS.md`](CASE_A_FAULT_20260814_170757_ANALYSIS.md)에 있다.

| 항목 | 결과 |
|---|---:|
| 실제 resolved speed | 0.10 m/s |
| active control | 59.04 s |
| P0→P1 / P1→P2 | 도달 / 도달 |
| P2→P1 return | 명시적 stop 전 31.28 s 동안 미도달, 최소 0.495 m/종료 1.063 m |
| position-velocity 적분 residual | outbound 0.72 m, return 1.19 m |
| return 시작 q target step | 109.7° |
| 최소 한 action 축 exact ±1 | 26.22% |
| 최소 한 T200 force clamp | 11.19% |
| preview→sent | 값 손실 0 |
| sent→servo | configured reversal/channel과 99.79% exact match |
| 종료 | automatic fault가 아닌 명시적 ROS CLI stop |

strict post-initialization ArUco-vs-aligned-odometry residual은 position median 0.216 m/p95 0.904 m/max 1.486 m였다. localization epoch/session/alignment는 끝까지 valid였으므로 `valid=true`는 pool absolute error가 작다는 뜻이 아니다. 반면 external ground truth와 RPM/thrust feedback이 없으므로 vision과 odometry 중 어느 쪽이 더 정확한지, 실제 motor force가 맞았는지는 단정할 수 없다.

현 deadline에서 우선할 것은 다음 세 가지다.

1. frozen policy Isaac straight 0.10/0.60으로 변경 전 baseline을 남긴다.
2. 기존 P0/P1의 observation parity, command curriculum, velocity reward, episode/horizon, action 계약, validation/manifest를 구현한 뒤 재학습한다.
3. 새 policy를 Gazebo GT 기반 Case A로 검증하고, GT와 EKF가 다를 때만 estimator 문제를 별도로 수정한다.

q slew, vision residual/reacquisition, RPM/current, full tracking-failure monitor는 이 최소 반복 후의 시스템 강화 항목으로 남긴다.

나머지 두 bag은 동일 analyzer로 비교해 baseline repeatability와 변경 전후 차이를 확인해야 한다. 파일명만으로 success/fault를 분류하지 않는다.

## 9. 실행 순서

1. **완료:** `model_299.pt` Isaac straight 0.10/60 s가 지속 요동 없이 추종함을 확인했다.
2. `test_policy.py`에 시점 정렬된 6축 raw/applied action, requested wrench/thruster force, CSV/JSON과 action plot을 구현한다.
3. 같은 policy의 Gazebo GT-feedback/deploy-feedback 0.10 A/B로 feedback 문제와 plant/action 문제를 분리한다.
4. 확인된 deploy 오류를 수정한 뒤 observation/action 계약, command curriculum, reward, 15–30 s episode/horizon과 evaluator/manifest를 구현한다.
5. smoke retrain 뒤 Isaac 0.10/0.60, hold·stop·reversal로 checkpoint를 선택한다.
6. Gazebo Case A 0.10에서 `P2 도달/방향전환/P1 return`, 속도·cross-track·action cap을 3회 연속 통과한다.
7. 같은 artifact/config으로 실기 Case A 0.10을 재시도한다.

DVL sample-and-hold, 소규모 mass/damping/thruster-gain DR과 19-D observation은 최종 sim2real 후보 전에 검토한다. exact G0/G1/G2 전체, plant 전면 식별, localization/fault matrix, Case C와 100-seed 승인은 별도 backlog이다. Frame/sign 불일치는 재학습으로 덮지 않고 계약을 먼저 고정한다.

## 10. 주요 근거 파일

### `step_2_BROV`

- [`CASE_A_FAULT_20260814_170757_ANALYSIS.md`](CASE_A_FAULT_20260814_170757_ANALYSIS.md): 최신 sim2real bag 정량 분석
- [`../case_a_fault_20260814_170757/metadata.yaml`](../case_a_fault_20260814_170757/metadata.yaml): bag duration/topic/message count
- [`CLAUDE.md`](CLAUDE.md): MDP, 5 s episode, mass DR와 mid-training validation 미구현 기록; 본 규모 학습 상태 문구는 과거 기록
- [`envs/vel_env.py`](envs/vel_env.py): action-allocation, 관측, reward, reset/DR
- [`envs/vel_env_cfg.py`](envs/vel_env_cfg.py): 25 Hz, 명령, reward, DR 설정
- [`guidance/los_guidance.py`](guidance/los_guidance.py): 구형 3D LOS와 heading 부호 주의
- [`test_policy.py`](test_policy.py): 기존 세 평가 시나리오와 ballast 근사
- [`../robots/data/BROV2/brov2_heavy.yaml`](../robots/data/BROV2/brov2_heavy.yaml): Isaac nominal plant/actuator/sensor 값
- `origin/main:step_2_BROV/deploy/WORKLOG_2026-08-13.md`: 기존 실기 policy/observation/thruster/guidance 시험 기록 (`git show`로 조회)
- `origin/main:step_2_BROV/deploy/DEVELOPMENT_STATUS.md`: 실기 telemetry gap과 정책 출력 요약 (`git show`로 조회)

### `brov_ros2`

- [`docs/DEMO_RUNBOOK.md`](../../brov_ros2-main/docs/DEMO_RUNBOOK.md): 포함 정책 재검증 요구
- [`docs/SIM2SWIM_DEMO.md`](../../brov_ros2-main/docs/SIM2SWIM_DEMO.md): localization 분리, Case A/C, 안전 envelope, 한계
- [`docs/ARCHITECTURE.md`](../../brov_ros2-main/docs/ARCHITECTURE.md): frame 및 16-D contract
- [`docs/POOL_LOCALIZATION_RUNBOOK.md`](../../brov_ros2-main/docs/POOL_LOCALIZATION_RUNBOOK.md): vision/localization 한계
- [`docs/ACTUATION_BACKEND_ROADMAP.md`](../../brov_ros2-main/docs/ACTUATION_BACKEND_ROADMAP.md): backend 계약과 승인 순서
- [`brov_base/brov_base/observation.py`](../../brov_ros2-main/brov_base/brov_base/observation.py): 현재 runtime 관측
- [`brov_base/brov_base/guidance.py`](../../brov_ros2-main/brov_base/brov_base/guidance.py): depth/terminal hold 수정
- [`brov_control/brov_control/policy_node.py`](../../brov_ros2-main/brov_control/brov_control/policy_node.py): policy, allocation, limiter, PWM 경로
- [`artifacts/policies/demo_policy/metadata.yaml`](../../brov_ros2-main/artifacts/policies/demo_policy/metadata.yaml): 현재 artifact contract

### Edo Gazebo/SITL

- [`TRACK_B.md`](../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/TRACK_B.md): 기존 GNC 실패와 정량 결과/제한
- [`gnc_lab/track_b/provided/thrust_allocation.py`](../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/gnc_lab/track_b/provided/thrust_allocation.py): Gazebo allocation과 선형 thrust mapping
- [`gnc_lab/track_b/provided/dvl_sim_node.py`](../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/gnc_lab/track_b/provided/dvl_sim_node.py): fake DVL 및 blocking latency
- [`gnc_lab/track_b/run_mission.py`](../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/gnc_lab/track_b/run_mission.py): 명목 20 Hz loop
- [`SITL_Models/Gazebo/models/bluerov2_heavy/model.sdf`](../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/SITL_Models/Gazebo/models/bluerov2_heavy/model.sdf): frame, hydrodynamics, ArduPilot channel mapping
- [`SITL_Models/Gazebo/models/bluerov2_heavy/configs.yaml`](../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/SITL_Models/Gazebo/models/bluerov2_heavy/configs.yaml): mass, inertia, geometry
