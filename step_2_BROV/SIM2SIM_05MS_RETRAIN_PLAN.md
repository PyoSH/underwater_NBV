# 0.10→0.50 m/s Case A gap 분리와 최소 재학습 계획

작성 기준: 2026-08-16

sim2real bag 보완: 2026-08-17

사용자 실험 맥락 반영: 2026-08-17

`brov_ros2` direct SITL smoke 반영: 2026-08-17

0.50 m/s GT/EKF feedback A/B 반영: 2026-08-17

Stage 2 no-GPS DVL/EKF 수정 및 회귀 반영: 2026-08-17

관련 종합 감사: [`RETRAIN_SIM2SIM_REVIEW.md`](RETRAIN_SIM2SIM_REVIEW.md)

관련 실험 분석: [`CASE_A_FAULT_20260814_170757_ANALYSIS.md`](CASE_A_FAULT_20260814_170757_ANALYSIS.md)

Stage 2 결과: [`STAGE2_DVL_ONLY_GAP_RESULT.md`](STAGE2_DVL_ONLY_GAP_RESULT.md)

Stage 2 fresh Case-A A/B: [`STAGE2_CASE_A_GT_DVL_AB_RESULT.md`](STAGE2_CASE_A_GT_DVL_AB_RESULT.md)

Stage 3 논문 구현 감사와 시간 제한형 재학습 사양: [`STAGE3_PAPER_IMPLEMENTATION_AUDIT.md`](STAGE3_PAPER_IMPLEMENTATION_AUDIT.md)

Stage 3 재학습 결과: [`STAGE3_RETRAIN_RESULT.md`](STAGE3_RETRAIN_RESULT.md)

MK2 Gazebo 배포 및 rosbag 판정: [`MK2_SIM2SIM_DEPLOY_RESULT.md`](MK2_SIM2SIM_DEPLOY_RESULT.md)

## 0. 사용자가 확인한 맥락과 현재 최소 범위

이 문서의 현재 우선순위는 다음 실험 맥락을 기준으로 한다.

1. IsaacLab 정책은 사용자 확인 기준 약 `0.58–0.63 m/s`로 보였지만, 현재 Eq. 9 코드의 exact 5 s 속력 범위는 `0.5831–0.6733 m/s`다. 더 중요한 점은 논문에서 Eq. 9가 속도 명령이 아니라 Frenet–Serret 자세 궤적이라는 것이다.
2. `step_2_BROV/deploy/*`를 사용한 Gazebo sim2sim은 `0.10 m/s`로 실행됐고, 자세는 대체로 추종했지만 선속도와 위치를 제대로 추종하지 못했다.
3. 같은 계열의 runtime/policy를 쓴 sim2real Case A에서도 자세는 얼추 추종했고 P2 종점의 방향전환도 수행했다. 이후 선속도·위치 추종이 불량해 사용자가 수동으로 stop했다.

따라서 bag의 `P2→P1 미도달`을 **방향전환 실패**로 해석하지 않는다. 실험 outcome은 `OPERATOR_ABORT(TRACKING_CONCERN)`이며, 현재의 1차 문제는 Gazebo와 실기에서 공통으로 나타난 **0.10 m/s 선속도·위치 추종 불량**이다. `q_d` step, localization 재획득, full fault matrix는 정량 관측으로는 남기되 현 deadline의 선행 blocker로 두지 않는다.

저수준 policy는 position을 직접 관측하거나 제어하지 않고, LOS가 position error를 `v_d`로 바꾸면 policy가 body velocity를 추종한다. 따라서 현재 position 표류는 새 position policy가 필요하다는 뜻이 아니라 **velocity-policy 실패의 downstream 결과**로 먼저 다룬다. 현 `policy.pt`의 deterministic synthetic inference에서도 zero-error observation의 sway action은 약 `+0.232`이고, `v_e.x=-0.10 m/s`일 때 surge/sway action은 약 `+0.207/+0.241`이다. 현 scale/allocation으로 재구성하면 각각 약 19.7 N, 20.5 N의 횡력 명령이므로 저속 policy bias를 최우선 가설로 둘 근거가 있다. 또한 현 reward는 quaternion 관련 0.4 항을 두 번 사용하는 반면 velocity 항은 0.2에 그쳐, “자세는 대체로 추종하지만 속도는 불량”한 관측과 정합한다 (`envs/vel_env_cfg.py:64-71`, `envs/vel_env.py:322-328`).

### 3단계 실행 계획

| 단계 | 목적 | 상태와 종료 조건 |
|---|---|---|
| 1. feedback gap 분리 | frozen actor·guidance·actuation을 고정하고 Gazebo GT와 MAVLink/EKF feedback만 A/B | **완료.** GT translational tracking PASS, EKF FAIL과 kinematic inconsistency를 rosbag으로 확인 |
| 2. sim2sim/sim2real 공통 배포 gap 수정 | 실기에 없는 stock GPS 경로를 no-GPS DVL→VPD→EKF3로 교체하고 같은 frozen actor로 재검증 | **완료(진단/경로 교체).** pulse frame/sign/scale PASS. Fresh full-cycle A/B에서 GT도 action bound 59.3%가 남았고, DVL-EKF는 return velocity RMSE 0.0495→0.1312 m/s와 whole force clamp 4.8→18.9%로 추가 악화 |
| 3. 논문 정합 구현 후 재학습 | 확인된 runtime 계약 위에서 학습 명령·observation/action·reward·horizon·최소 DR를 고치고 새 artifact 생성 | **1차 구현·재학습·MK2 배포 완료, 후보 기각.** Isaac steady gate는 통과했지만 fresh Gazebo GT에서도 action cap 98.9%, force clamp 30.2%, outbound vector RMSE 0.355 m/s로 실패. real 배포 금지 |

이 순서는 `gap 수정`과 `재학습` 중 하나를 버리는 선택이 아니다. 2단계에서 stock GPS estimator가 along-speed를 악화한 사실과, no-GPS DVL로 교체한 뒤에도 cross-axis jitter·action saturation이 남은 사실을 각각 분리했다. 따라서 이후 재학습 효과를 estimator 수정으로 잘못 귀속하지 않을 수 있다. 기존 model_299 학습 코드는 논문의 시간변화 곡선 항을 body velocity template으로 사용하고 episode 안의 desired attitude는 정적으로 뒀다. 1차 `paper_ref_v1`/`deploy_v2` 구현에서 이 역할을 교정했지만, fresh Gazebo 실패를 통해 bounded-action, 실제 Case-A recovery horizon과 plant/observation 분포 보완이 추가로 필요함을 확인했다.

### 재학습 구현 목표 및 상태

재학습 필요성은 frozen policy의 0.10 m/s 시험 결과를 기다려 결정하는 사항이 아니다. 기존 문서와 코드에 이미 다음 gap이 기록돼 있고, Gazebo와 실기에서 같은 저속 추종 불량도 관측됐으므로 **재학습은 진행한다.** frozen policy 0.10/0.60 시험은 기존 정책의 기준선을 남기는 진단일 뿐 재학습 GO/NO-GO gate가 아니다.

#### [x] G0 — 현 frozen policy의 Isaac 0.10 m/s action 확인

목적은 policy weight, gain, frame, LOS 또는 plant를 바꾸기 전에, 학습 분포에 없던 `0.10 m/s` 명령에서 기존 policy가 실제로 어떤 6축 제어 action을 출력하는지 사용자가 IsaacLab에서 직접 확인하는 것이다.

시험 대상은 우선 다음 checkpoint로 고정한다.

```text
checkpoint = logs/sim2sim_repo/model_299.pt
SHA-256    = da2db94ac6f4fba184e91fb301e91ff3913a1b9b540ab034db437800b9dc563d
scenario   = straight_line, cruise_speed=0.10 m/s, duration=60 s
physics    = nominal fixed evaluation profile
```

2026-08-17 read-only tensor 비교에서 이 checkpoint의 actor weight/bias 6개와 `deploy/exported/policy.pt`의 모든 actor tensor가 exact equality(`max_abs_error=0`)임을 확인했다. 이 TorchScript는 `brov_ros2` demo artifact와도 같은 SHA-256 `0d89f3270f46214f1569b7d48dcb5e25363b1d9b7353b82ced0fc67c0093a472`다. 따라서 **Isaac와 sim2sim/sim2real은 같은 actor weight**를 사용한다. 다만 source commit, seed와 training config provenance는 metadata에 없으므로 G7은 여전히 필요하다.

현재 코드도 다음 명령으로 0.10 m/s LOS를 실행할 수 있다. GUI에서 직접 보려면 `--headless`를 붙이지 않는다.

```bash
cd /workspace/OceanRL_test/step_2_BROV
/isaac-sim/python.sh test_policy.py \
  --checkpoint logs/sim2sim_repo/model_299.pt \
  --test straight_line \
  --cruise_speed 0.1 \
  --duration 60
```

현재 plot에서 목표/실제 body velocity와 자세·경로는 볼 수 있지만, action은 `mean(abs(action))` 한 스칼라만 저장된다. 이 값으로는 surge/sway/heave/roll/pitch/yaw 중 어느 축이 bias·포화되는지 판단할 수 없다. 따라서 G0의 축별 정량 분석을 완료하려면 `test_policy.py`에 다음을 구현한다.

1. action을 만들 때 사용한 pre-step `v_d`, body velocity, velocity error, `q_d`, waypoint index와 observation timestamp를 같은 sample로 저장한다.
2. NN의 raw 6-D action과 `env._actions`의 clamp 후 6-D action을 `[surge,sway,heave,roll,pitch,yaw]` 순서로 25 Hz 저장한다.
3. clamp 후 action으로 계산한 6-D requested wrench와 8-thruster requested force를 저장하고 T200 `-51.5/+64.1 N` clamp 초과를 표시한다.
4. CSV 또는 NPZ raw trace, JSON summary와 6축 action plot을 저장한다. 파일명에는 `0p10`, checkpoint명/SHA와 run ID를 넣어 기존 0.5 plot을 덮어쓰지 않는다.
5. 축별 signed mean, RMS, p95 absolute, min/max, `|action|>=0.99` 비율·최장 연속시간과 requested-force clamp 비율을 출력한다.
6. 최초 transient, outbound steady, waypoint reversal 전후 ±2 s, return steady를 별도 집계한다. 5 m 경로에서 이상적인 첫 전환은 약 45 s이므로 60 s를 사용한다.

G0 완료 조건은 60 s의 1,500 policy step에 대해 위 trace와 summary가 생성되고, raw/applied action의 시점 정렬과 reset/waypoint-switch 표식이 확인되는 것이다. 이 결과는 기존 정책의 저속 OOD 동작을 기록하는 baseline이며 재학습 취소 조건이 아니다.

##### 2026-08-17 직접 실행 결과

동일 명령을 headless로 다시 실행했고 사용자 GUI 관측과 같은 결과를 얻었다.

| 항목 | 결과 |
|---|---:|
| policy steps / reset | 1,500 / 0 |
| waypoint 전환 | 1회, 45.96 s |
| 평균 속도 vector error | 0.0074 m/s |
| 마지막 2 s 평균 속도 error | 0.0024 m/s |
| X/Y/Z 위치 변화 범위 | 4.5374 / 0.0728 / 0.0498 m |
| 시간 평균 `mean(abs(action_6))` | 0.0238 |
| 최대 `mean(abs(action_6))` | 0.5296 |

결과 plot은 [`plots/policy_eval_straight_line.png`](plots/policy_eval_straight_line.png)이다. 약 45.96 s의 180° 전환에서 짧은 속도·자세 transient는 있지만 지속적인 요동은 없고 곧 다시 안정화된다. 현재 action 수치는 6축 평균 절댓값이므로 개별 축 포화로 해석하면 안 된다.

이 결과는 다음을 의미한다.

1. `0.10 m/s`가 학습 분포에 없다는 사실만으로 Isaac 정책이 요동한다는 가설은 기각한다.
2. nominal Isaac에서 180° waypoint reversal 자체도 지속 요동의 충분조건이 아니다.
3. 같은 actor weight가 Isaac에서는 안정적이고 Gazebo·실기에서만 요동하므로, 1차 원인 후보를 `brov_ros2` full-SITL의 MAVLink/EKF velocity observation·frame·timestamp와 Gazebo/실기 plant·actuator scale·lag 순으로 올린다.
4. 저속 curriculum 재학습은 정확도·robustness 보완 목표로 유지하지만, 현재 요동을 재학습만으로 덮지 않는다.

##### 2026-08-17 `brov_ros2` direct SITL integration smoke (`deploy/*` 미사용)

이번 시험은 `step_2_BROV/deploy/*`를 다시 실행한 것이 아니다. Edo `bluerov2_sitl:student-deploy` 컨테이너에 로컬 [`brov_ros2-main`](../../brov_ros2-main)을 bind mount하고, 컨테이너 안에서 Torch `2.7.0+cpu`를 설치한 뒤 8개 ROS 2 package를 `colcon build`했다. 기존 demo TorchScript의 `(1,16) -> (1,6)` inference, `rl_demo.launch.py`, MAVLink telemetry와 policy preview까지 shadow 경로가 동작했다.

실기 배선 보정을 SITL에 잘못 적용하지 않도록 다음 최소 adapter도 구현했다.

- 실기 기본 `real_brov2=[+,-,-,+,+,+,+,-]`는 변경하지 않았다.
- `edo_sitl_identity=[+,+,+,+,+,+,+,+]` profile을 추가하고 `udpin:` 연결에서만 허용했다.
- 별도 `vehicle_sitl.yaml`을 추가했으며 신규 profile/MAVLink 단위시험 25개와 `brov_base` 전체 92개가 통과했다.
- 현재 launch parameter가 YAML의 connection을 뒤에서 덮으므로, 재현 명령에는 `vehicle_config:=.../vehicle_sitl.yaml`과 `connection:=udpin:0.0.0.0:14552`를 **함께** 전달해야 한다. 실기 launch의 기본 connection은 바꾸지 않는다.

첫 arm 요청은 SITL `LOCAL_POSITION_NED` 약 4 Hz와 attitude 25 Hz 사이 source-time skew가 `180 ms > 150 ms`여서 안전하게 거부됐다. MAVProxy `set streamrate 25` 적용 뒤 active 구간에는 같은 gate warning이 없었고 모든 control/debug topic이 25 Hz로 유지됐다. 안전 threshold를 완화해 통과시킨 것이 아니다.

`mission_sim2swim_a.yaml`의 Case A `0.10 m/s`를 짧게 실행한 정량 결과는 다음과 같다. Raw policy action은 bag의 16-D observation을 동일 SHA의 TorchScript에 다시 넣어 복원했다. `/brov/action`은 이미 operational limit가 적용된 값이므로 두 값을 섞지 않는다.

| 항목 | 결과 |
|---|---:|
| build / shadow / MAVLink transport | PASS |
| CONTROL ACTIVE | 17.720 s |
| observation/action/sent PWM | 443/443/442 sample, 24.999 Hz, max gap 42.4 ms |
| 목표 / 실제 속력 평균 | 0.101 / 0.331 m/s |
| 실제 속력 p95 / max | 0.547 / 0.588 m/s |
| 목표방향 속도 평균 / 횡속도 RMS | 0.095 / 0.320 m/s |
| velocity error RMS / p95 | 0.347 / 0.513 m/s |
| active 수평 path / 순변위 | 5.206 / 1.359 m |
| active cross-track 범위 | -0.151~+0.373 m |
| raw action 중 한 축 이상 `abs(a)>=0.99` | 68.2% |
| raw sway / pitch / yaw `abs(a)>=0.99` | 40.0 / 14.2 / 55.3% |
| limited action 중 한 축 이상 operational cap | 99.3% |
| sent PWM `max(abs(pwm))` | 0.339 (`0.35` 절대 cap 미도달) |
| stop 뒤 첫 8채널 1500 us neutral | 40.4 ms |

Waypoint index는 끝까지 0이고 endpoint에 도달하지 않았으므로 이 시험은 180도 방향전환 실패가 아니다. 직선 leg에서 전진 평균은 목표와 비슷했지만 훨씬 큰 횡속도와 반복 correction 때문에 17.72 s 동안 수평 path가 5.21 m에 달했다. Identity profile의 sent PWM과 SITL `SERVO_OUTPUT_RAW`는 약 37 ms lag 정렬에서 평균 오차 1.13 us였고, 실기 reversal mask를 가정하면 28.94 us로 악화됐다. 따라서 이번 bag에서 **채널 reversal과 전송 주기는 정상 근거가 강하다.**

판정은 **build/shadow/transport PASS, 0.10 m/s tracking FAIL, 원인 분리 미완료**다. Active 전에 끝난 Gazebo odometry capture는 성능 근거로 쓰지 않으며, active 성능값은 MAVLink/EKF 기반 `brov_ros2` trace다. 또한 legacy Edo plant, T200 inverse-to-linear plugin mismatch와 Case-C action/PWM 제한 envelope를 사용했으므로 이 결과를 Isaac nominal parity로 해석하지 않는다. Active 구간에 source-skew gate drop은 없었기 때문에 gate dropout 하나만으로 실패를 설명할 수는 없지만, estimator·frame·action/plant 중 무엇이 주원인인지는 synchronized GT A/B 전까지 확정하지 않는다.

수동 stop/disarm과 neutral은 정상 완료됐지만 launch `Ctrl-C` 뒤 두 node에서 `rcl_shutdown already called` 예외가 남았다. 이는 tracking 실패 원인은 아니며, 다음 반복 전에 `rclpy.ok()` 확인과 예외 격리 cleanup으로 정리할 비차단 lifecycle TODO다.

산출물은 [`runtime/experiments/sim2sim_brov_ros2_0p1_smoke_20260817_021003`](../../brov_ros2-main/runtime/experiments/sim2sim_brov_ros2_0p1_smoke_20260817_021003)에 rosbag, Gazebo capture와 [`analysis_summary.json`](../../brov_ros2-main/runtime/experiments/sim2sim_brov_ros2_0p1_smoke_20260817_021003/analysis_summary.json)으로 보존했다.

#### [x] G0B — 같은 policy의 Gazebo GT/EKF feedback A/B 분리 (1단계 완료)

동일 TorchScript, LOS, action allocation, T200 inverse, SITL identity channel map과 표준 Case-A controller/safety envelope를 유지하고 **policy/guidance로 들어가는 feedback source만** 바꿨다. 0.5 m/s 정상상태를 180도 반전과 분리하기 위해 두 run 모두 `START-relative +0.20 m` 심도에 먼저 도달한 뒤 5 m 단방향 직선을 비행했다.

1. `gazebo_truth`: Gazebo pose/body twist를 ENU/FLU→NED/FRD로 변환해 guidance와 legacy 16-D observation을 구동
2. `mavlink_ekf`: 기존 ArduSub `LOCAL_POSITION_NED`/attitude 경로로 guidance와 observation을 구동

MAVLink health, arm/disarm, RC override와 실제 actuation 경로는 두 run 모두 그대로 유지했다. GT source가 stale/invalid이거나 publisher가 정확히 하나가 아니면 ARM/START 및 active output이 fail-closed되며 EKF로 자동 fallback하지 않는다. Gazebo의 `twist.angular`는 body gyro가 아니라 RPY 차분값이므로 policy angular rate는 연속 quaternion과 simulation source stamp로 계산했다.

구현 위치는 다음과 같다.

- `brov_base/brov_base/gazebo_truth.py`: timestamp·frame 검증과 ENU/FLU→NED/FRD 변환
- `brov_base/brov_base/obs_node.py`: immutable `feedback_source`, source별 sample key/dt, arm/authority/stale gate와 동기 진단 topic
- `brov_control/brov_control/policy_node.py`: raw action, requested/limited thrust·wrench·PWM 진단
- `brov_bringup/launch/sim2sim_0p5_ab.launch.py`: 항상 같은 Gazebo odometry bridge를 띄우고 source 하나만 선택
- `brov_bringup/config/mission_sim2sim_straight_0p5.yaml`: `takeoff_then_align`, 0.50 m/s, 5 m, non-loop profile

격리 build 후 수정 범위 테스트는 `brov_base 108`, `brov_control 17`, `brov_bringup 33`, 합계 158개가 모두 통과했다. 전체 workspace의 291개 중 남은 4개 실패는 이번 변경과 무관한 기존 AprilTag survey/OpenCV API 및 viz survey 불일치다.

두 rosbag은 모든 feedback·observation·action·allocation·PWM·servo topic을 같은 schema로 기록했다. 아래 정상상태는 수평 waypoint 진입 후 첫 1 s와 terminal hold를 제외했다.

| 0.50 m/s 단방향 지표 | GT feedback | MAVLink/EKF feedback |
|---|---:|---:|
| takeoff+5 m mission 완료 시각 | 13.480 s | 13.641 s |
| 실제 GT 목표방향 속도 평균 | **0.468 m/s** | **0.436 m/s** |
| 실제 GT vector velocity RMSE | **0.0588 m/s** | **0.1089 m/s** |
| 실제 GT cross-speed RMS | 0.0404 m/s | 0.0811 m/s |
| 실제 GT cross-track RMS | 0.0022 m | 0.0121 m |
| 실제 GT depth error RMS | 0.0038 m | 0.0101 m |
| controller가 본 목표방향 속도 | 0.468 m/s | 0.478 m/s |
| selected feedback↔GT velocity RMSE / p95 | 0 / 0 | **0.0627 / 0.0977 m/s** |
| horizontal position drift RMS / p95 / end | 0 / 0 / 0 | **0.0701 / 0.1146 / 0.0845 m** |
| horizontal `Δp−∫vdt` norm | 0.0204 m | **0.3272 m** |
| 최적 constant-lag / 보정 뒤 RMSE | 0 ms / 0.0001 m/s | **-10 ms / 0.0595 m/s** |
| 한 축 이상 `abs(action)>=0.99` | **58.2%** | **64.1%** |
| pitch `abs(action)>=0.99` | 55.5% | 54.5% |
| 연속 bound 최장시간 | 0.160 s | 0.203 s |
| requested thruster force clamp | 5.94% | 12.12% |
| action rate / max gap | 25.000 Hz / 43.3 ms | 24.953 Hz / 80.0 ms |

판정은 다음과 같다.

1. GT feedback에서는 translational gate `v_parallel=0.45–0.55 m/s`, vector RMSE `<=0.08 m/s`를 모두 통과했다. EKF feedback은 두 gate를 모두 실패했다.
2. EKF feedback은 실제 0.436 m/s를 약 0.478 m/s로 보았고, 같은 수평 trace에서 position과 적분 velocity의 closure가 약 0.020 m에서 0.327 m로 붕괴했다. 따라서 **MAVLink/EKF velocity·position feedback 계약이 Gazebo의 선속도 gap을 인과적으로 악화한다.** 재학습으로 먼저 덮을 문제가 아니다.
3. 25 Hz 자체는 유지됐고, ±250 ms constant-lag sweep의 최적값은 -10 ms였지만 velocity RMSE는 약 5%만 줄었다. 단순 주기나 상수 지연이 주원인은 아니므로 2단계에서는 `LOCAL_POSITION_NED` position/velocity frame·scale와 DVL/EKF update/source timestamp를 직접 분리한다.
4. 반면 GT feedback도 pitch 중심 actor-bound 점유가 58.2%이고 raw policy output과 `/brov/action`은 exact하게 같았다. 즉 runtime limiter가 잘라 만든 현상이 아니며 전체 deployment acceptance는 아직 실패다. Estimator 수정 뒤에도 이 bound 점유가 남으면 action frame/allocation/plant와 현재 정책 학습의 구조적 문제를 다음 원인으로 다룬다.

두 run은 같은 장시간 실행 중인 uniform Gazebo world에서 순서대로 수행했고 각 START에서 상대 origin/heading과 policy integrator를 reset했다. 병진 위치와 yaw는 이 profile에 대해 대칭이지만 ArduSub estimator process를 hard reset하지 않았다는 순서 효과는 남는다. 따라서 2단계 수정 전후 비교는 동일 순서 또는 fresh SITL로 한 차례 재확인한다.

재현 분석기는 [`analyze_brov_stage1_ab.py`](analyze_brov_stage1_ab.py), machine-readable 결과는 [`STAGE1_SIM2SIM_0P5_AB_RESULT.json`](STAGE1_SIM2SIM_0P5_AB_RESULT.json)에 보존했다. 원본 bag은 각각 [`stage1_gt_feedback_0p5_20260817_040227`](../../brov_ros2-main/runtime/experiments/stage1_gt_feedback_0p5_20260817_040227)과 [`stage1_ekf_feedback_0p5_20260817_040600`](../../brov_ros2-main/runtime/experiments/stage1_ekf_feedback_0p5_20260817_040600)이다.

#### [x] G0C — no-GPS DVL/EKF 수정과 frozen-policy 회귀 (2단계 완료)

실기에는 GPS가 없으므로 stock Gazebo의 5 Hz synthetic GPS EKF를 더 튜닝하지 않았다. Gazebo body velocity를 15 Hz DVL로 sampling하고 100 ms delay/noise/FOM을 적용한 뒤 Water Linked 계열과 같은 `VISION_POSITION_DELTA`로 EKF3 ExternalNav에 넣었다. `GPS1_TYPE=0`, `GPS2_TYPE=0`, `SIM_GPS_DISABLE=1`이며 DataFlash의 GPS/GPA message count도 0이다.

정책 없는 pulse에서 DVL-EKF velocity RMSE는 0.011 m/s, XY determinant는 0.993으로 frame/sign/scale가 통과했다. 그러나 injected VPD delta의 수평 누적오차가 약 0.009 m인 데 비해 XKF full horizontal closure는 0.077 m여서 작은 fusion bias가 남았다. delay/noise를 제거한 A0 oracle에서도 0.073 m이므로 100 ms와 noise가 주원인은 아니다.

같은 frozen policy의 0.5 m/s/5 m fresh run은 11.603 s에 mission을 완료했고 physical `v_parallel=0.469 m/s`로 stock GPS-EKF의 0.436 m/s를 회복했다. 하지만 vector RMSE 0.113 m/s, cross-speed RMS 0.102 m/s, any-axis action bound 67.9%와 requested-force clamp 16.5%가 남았다. 즉 estimator 수정은 along-speed gap을 줄였지만 jitter를 제거하지 못했다. 다음 작업은 GPS/EKF 추가 튜닝이 아니라 G1–G7의 paper-aligned policy 재학습이다. 상세 수치·재현 파일·artifact는 [`STAGE2_DVL_ONLY_GAP_RESULT.md`](STAGE2_DVL_ONLY_GAP_RESULT.md)에 있다.

#### [x] G0D — fresh full-cycle Case-A GT/DVL-EKF A/B (3단계 직전 확인 완료)

Gazebo, ArduSub EEPROM/EKF와 policy integrator를 매 run fresh start하고 동일 physical START barrier를 사용했다. 시작 상태 차이는 position 1.96 mm, velocity `7.8e-8 m/s`, attitude `0.000184°`, DVL sequence 0이었다. 두 run 모두 `P0→P1 takeoff → 2 m outbound → 180° 반전 → 2 m return`의 waypoint RLE `[0,1,2,1]`을 fault 없이 완주했다.

GT feedback은 straight steady에서 outbound/return vector velocity RMSE `0.0756/0.0495 m/s`, depth RMS `0.0086/0.0068 m`로 physical tracking을 크게 회복했다. 그러나 pitch action은 여전히 `-1↔+1`로 왕복했고 whole-cycle actor bound 점유가 59.3%였다. 따라서 GT만 사용해도 Isaac nominal처럼 control output이 완전히 매끈해지는 것은 아니다.

Water Linked code-default-aligned ideal VPD feedback은 같은 시작 상태에서 outbound/return velocity RMSE를 `0.0993/0.1312 m/s`, depth RMS를 `0.1177/0.1048 m`, whole-cycle force clamp를 18.9%로 악화했다. 특히 180° turn window의 cross-speed RMS는 GT `0.0576` 대비 `0.1238 m/s`였다. 결론은 **공통 frozen-policy/action jitter가 있고, DVL-EKF/Baro를 포함한 `LOCAL_POSITION_NED` feedback이 이를 추가 증폭한다**는 것이다.

외부 supervisor의 `2→1` 감지 뒤 control inactive까지 GT/DVL-EKF 각각 0.758/0.437 s가 걸려 exact one-lap 0.25 s shutdown gate는 실패했다. 성능 metric은 return edge에서 잘랐지만, 새 validation harness는 같은 tick에 finite-lap complete/output-close를 구현해야 한다.

이 A/B는 estimator 인과 분리를 위한 direct-relative Case-A motion contract다. production의 camera/pool-localization/resolved-mission orchestration은 포함하지 않았다. DVL profile도 `DVL_DOWN`, `VISION_POSITION_DELTA`, rangefinder topology는 맞췄지만 noise/delay/dropout/FOM=0, `VISO_POS=0`인 lower bound이며 실제 Water Linked saved settings와 mount/packet 분포는 미확보다. 상세 결과와 원본 bag/DataFlash는 [`STAGE2_CASE_A_GT_DVL_AB_RESULT.md`](STAGE2_CASE_A_GT_DVL_AB_RESULT.md)에 있다.

#### [x] G0E — deploy_v2 재학습, MK2 ROS 배포 및 fresh Gazebo 판정

논문 공개 계약과 프로젝트 deploy 확장을 분리해 `deploy_v2`를 headless로
2,048 env, 128-step rollout, 50 iteration 재학습했다. Isaac steady 0.5 m/s는
`v_parallel=0.482 m/s`, vector RMSE `0.0206 m/s`, action/force clamp 0%로
통과했다. 기존 model_299와 섞이지 않도록 metadata-bound MK2 artifact,
`policy_node_mk2`, 명시적 T6, 별도 launch/mission을 구현했다.

동일 MK2 artifact로 fresh direct-relative 2 m Case-A-shaped cycle을 GT와
Water Linked-aligned DVL-EKF feedback에서 각각 수행했다. 두 arm 모두
`[0,1,2,1]`을 완주했고 policy replay 최대오차 `4.77e-7`, T6-wrench 최대오차
`3.81e-6`으로 export/runtime 계약은 통과했다. 그러나 GT arm조차 whole-cycle
action cap 98.9%, force clamp 30.2%, outbound vector RMSE 0.355 m/s로 실패했다.
DVL-EKF는 force clamp 47.5%, outbound RMSE 0.471 m/s로 추가 악화했다.

따라서 model_49는 실기 승격 없이 격리한다. 원인은 T6/ROS inference가 아니라
Gazebo observation에서 unbounded actor가 unit 범위를 벗어난 비율이 GT 99.58%,
DVL 89.78%인 policy-distribution/closed-loop gap이다. 다음 후보는 bounded-action
학습 계약, 15--20 s Case-A/recovery curriculum 및 측정 기반 최소 plant DR를
반영한 뒤 fresh GT gate부터 다시 통과해야 한다. 상세 결과는
[`MK2_SIM2SIM_DEPLOY_RESULT.md`](MK2_SIM2SIM_DEPLOY_RESULT.md)에 있다.

#### 1차 재학습 구현 상태와 후속 보완

| ID | 구현 목표 | 현재 미구현 또는 불일치 | 완료 조건 |
|---|---|---|---|
| [x] G1 | Observation parity | hemisphere, 적분 clamp/reset 및 중복 sample freeze를 공용 contract로 구현하고 테스트했다. | MK2 runtime metadata가 `brov_velocity_observation_v2`를 고정하며 rosbag replay exact |
| [x] G2 | Action frame/sign 계약 | Isaac/runtime에 explicit `T6`를 적용하고 legacy no-T6를 별도 SHA-bound 계약으로 보존했다. | 6축 basis, startup negative tests 및 실제 bag T6-wrench residual `3.81e-6` 통과 |
| [x] G3 | Paper desired state + deploy command curriculum | `paper_ref_v1`의 Eq. 9 Frenet-Serret `q_d(t)`/exact 0.5 sphere와 별도 `deploy_v2` 0/0.1/0.5, hold/stop/restart/reversal을 구현했다. | pure Torch tests와 1차 재학습 완료 |
| [~] G4 | Velocity-priority reward/metric | velocity precision, delta-action 및 force-clamp penalty를 구현했지만 penalty가 clipped action만 보아 unbounded actor overflow를 놓쳤다. | bounded policy distribution 또는 pre-clamp overflow penalty/log를 추가하고 GT steady cap <1% |
| [~] G5 | Episode/PPO horizon | 5 s 안 1 transition, rollout 128 step은 구현했다. Gazebo Case-A에서는 장기 recovery/반전 분포가 부족했다. | 15--20 s takeoff/straight/reversal/return curriculum로 최소 fine-tune |
| [~] G6 | Validation/checkpoint 선정 | Isaac JSON/plot 및 Gazebo GT/DVL rosbag gate를 구현했다. 1차 후보는 Isaac PASS/Gazebo FAIL로 기각했다. | fresh GT 3회 선행 gate와 validation 기반 checkpoint 자동 선택 |
| [x] G7 | Artifact lineage | checkpoint/policy/vehicle SHA, seed, profile, observation/action/T6 계약을 metadata에 저장하고 MK2 startup에서 fail-closed 검증한다. | legacy와 MK2 artifact/node/launch 완전 분리 |

`CLAUDE.md:214,223-224`에 명시된 미구현 항목도 이 분류에 반영했다. 그중 **mass DR**과 **주기적 mid-training validation**은 실제 미구현이다. 다만 W&B logger 자체와 nominal T200 deadband·정/역 비대칭·saturation·50 ms lag는 이미 구현돼 있으므로, 남은 것은 각각 자동 validation과 actuator **randomization**이다. mid-training video는 재학습 필수 기능이 아니다. 같은 문서의 “본 규모 학습 미실행”은 사용자 설명 및 현재 `model_299.pt`와 맞지 않는 과거 상태이므로 TODO로 사용하지 않는다.

#### 최종 sim2real 후보 전에 권장하지만 첫 smoke retrain을 막지 않는 항목

- [ ] **G8 — 최소 reality model:** Stage 2 SITL runtime에는 no-GPS DVL의 15 Hz/100 ms stress profile과 Water Linked code-default-aligned ideal lower bound를 구현했다. 그러나 Isaac training observation은 여전히 ground truth를 직접 읽으므로 같은 sensor topology와 소규모 mass/COM·damping·thruster-gain DR를 학습에 구현해야 한다. 현재 DVL rate/noise/delay/mount는 실측값이 아니며, 실제 settings/raw packet을 확보한 항목만 최종 DR 범위로 승인한다.
- [ ] **G9 — 19-D observation 후보:** 구조적 feedforward 모호성을 없애도록 `v_desired_body(3)`를 추가한다. 시간상 첫 16-D parity 재학습과 비교할 수 있으나, zero-state bias가 남으면 19-D를 최종 후보로 올린다.

#### 이번 일정에서 제외

mid-training video, AprilTag continuous fusion/reacquisition, full localization/fault matrix, current/tether·single-thruster failure DR, plant 전면 식별, Case C/square, 100-seed 승인은 이번 저속 Case A 복구 뒤로 미룬다. Position learner는 추가하지 않는다.

### 가장 짧은 실행 순서

1. **완료:** 현 `model_299.pt`의 Isaac `straight 0.10`, 60 s baseline에서 지속 요동이 없음을 확인했다.
2. **완료(실패 기준선):** `brov_ros2` direct full-SITL Case A에서 straight leg 0.10 m/s 안정 추종이 실패함을 25 Hz bag으로 확인했다.
3. **완료:** G0B의 synchronized Gazebo GT source/logger를 구현하고 0.50 m/s GT/EKF feedback A/B를 실행했다. GT translational tracking은 통과했고 EKF tracking은 실패했다.
4. **완료:** 실기에 없는 stock GPS를 제거하고 no-GPS DVL 15 Hz→`VISION_POSITION_DELTA`→EKF3 ExternalNav 경로를 구현했다. 정책 없는 A1 pulse에서 velocity RMSE 0.011 m/s와 정상 frame/sign/scale를 확인했다. 10 s horizontal closure는 약 0.05 m 경계이며 장기/수직 closure TODO는 별도로 남긴다.
5. **완료:** 수정한 DVL-only EKF 경로에서 같은 frozen-policy 0.50 m/s/5 m profile을 fresh SITL로 재검증했다. 실제 `v_parallel=0.469 m/s`와 mission 완료는 통과했지만 vector RMSE 0.113 m/s, cross-speed 0.102 m/s, action bound 67.9%로 jitter/saturation은 실패했다.
6. **완료:** 동일 fresh START barrier의 2 m full-cycle Case-A를 GT/DVL-EKF feedback으로 각각 실행했다. GT도 actor bound 59.3%가 남았고 DVL-EKF는 return velocity·depth·force clamp를 추가 악화해 복합 원인을 확인했다.
7. **완료(후보 기각):** G1–G7의 1차 구현과 headless `deploy_v2` 재학습, metadata-bound MK2 ROS 배포를 완료했다. Isaac steady gate는 통과했지만 fresh Gazebo GT에서도 action cap 98.9%와 vector RMSE 0.355 m/s로 실패했다.
8. **다음:** T6/inference는 동결하고 bounded-action/pre-clamp overflow 계약, 15--20 s Case-A recovery curriculum, 측정 기반 최소 plant DR만 보완한다. `3 iter smoke -> 20 iter 조기 gate -> 통과 시 50 iter` 순서로 재학습한다.
9. 새 후보가 fresh Gazebo GT Case A를 3회 통과한 뒤에만 DVL-EKF 3회를 수행한다. 양쪽을 통과하기 전에는 sim2real Case A를 재시도하지 않는다.

최소 통과 기준은 Gazebo 0.10 straight 정상상태 `v_parallel` 평균 0.08–0.12 m/s, p95 0.15 m/s 이하, max 0.20 m/s 이하, cross-track RMS 0.15 m 이하이다. estimator 자체의 선행 gate는 policy-free axis pulse의 EKF-vs-GT 속도 RMSE 0.03 m/s 이하이며 Stage 2에서 통과했다. 최종 closed-loop에서도 0.03 m/s 이하를 목표로 하지만, 현 actor가 만드는 큰 cross-axis/각운동과 coupled estimator error를 분리하기 위해 이 값은 새 policy의 validation gate로 다시 검사한다. Gazebo Case A는 `P2 도달`, `방향전환`, `P1 return 도달`을 분리 기록하고 3회 연속 통과한 뒤 sim2real을 재시도한다.

아래 세부 절은 구현 근거와 후속 승인 항목을 보존한 상세 로드맵이다. 상단의 필수·권장·제외 분류가 현재 일정의 우선순위다.

## 1. 후속 로드맵의 목표를 두 단계로 분리한다

### 후속 단계 A — 기존 정책을 그대로 둔 0.5 m/s sim2sim

목표는 다음과 같다.

> 기존 frozen policy의 weight를 변경하지 않고 IsaacLab `test_policy.py`의 0.5 m/s LOS 검증을 Gazebo에 재현한 뒤, 불안정 원인이 policy가 아니라 observation, frame, actuator, plant, timing 또는 guidance 차이인지 분리한다.

이 단계에서는 임의 gain, action sign, policy 출력 bias를 튜닝해서 성공시키면 안 된다. simulator adapter와 계약 오류는 고칠 수 있지만, 바꾼 계약은 이름과 결과를 별도로 남긴다.

### 후속 단계 B — 앞선 요소를 보완한 새 정책 확장 학습

단계 A에서 만든 25 Hz runtime, calibrated Gazebo plant, trace, 자동 metric과 failure 분류를 그대로 validation 기반으로 사용한다. 관측·action frame·guidance·DR·reward를 수정한 뒤 새 schema와 새 artifact version으로 학습한다.

두 단계를 섞으면 기존 정책의 문제와 simulator 이식 문제를 구분할 수 없다.

## 2. 0.5 m/s의 정확한 의미

`0.5 m/s`는 현재 코드에서 **Isaac 학습 속력과 완전히 동일한 값이 아니라 Isaac LOS 검증 속력**이다. 사용자가 확인한 실제 학습 운용 범위는 약 0.58–0.63 m/s이다.

- 학습 명령은 `q_cmd ⊗ [0.5, 0.5 sin(0.2t), 0.3 cos(0.2t)]`이다 (`envs/vel_env.py:290-297`).
- source 수식으로 계산한 5 s episode 목표 norm의 이론 범위는 `sqrt(0.34 + 0.16 sin²(0.2t))`, 약 0.583–0.672 m/s이다. 이 수식상 상한과 사용자가 확인한 0.58–0.63 m/s 운용 범위는 구분해 기록한다.
- `0.5`는 학습 템플릿의 surge 성분이다.
- LOS 검증은 벡터를 정규화한 뒤 정확히 `||v_d||=0.5 m/s`로 만든다 (`guidance/los_guidance.py:131-136`, `test_policy.py:52-55`).

따라서 단계 A의 공식 이름은 `Isaac LOS validation parity at 0.5 m/s`로 두는 것이 정확하다.

## 3. 단계 A의 첫 blocker: 0.5 연구 profile과 실기 저속 profile 분리

현재 `brov_ros2`에는 서로 모순되는 값이 있다.

| 근거 | Case A 속도 |
|---|---:|
| `docs/SIM2SWIM_DEMO.md:47-55` | 0.10 m/s |
| `mission_sim2swim_a.yaml:9-16` | 0.10 m/s |
| `test_sim2swim_contract.py:71-84` | 0.10 m/s를 assert |
| `mission_manager_sim2swim_a.yaml:28-34` 현재 값 | `0.50 # 0.10` |
| `brov_base/config/safety.yaml:35` 현재 상한 | `0.60 # 0.30` |

현재 config는 문서·test와 불일치하며, 일반 safety 상한도 0.5를 허용한다. 이 상태로는 sim 전용 0.5가 실기 Case A에 유입될 수 있고 contract test도 논리적으로 실패한다.

최소 세 profile을 별도 파일과 별도 test로 고정해야 한다.

1. `sim2sim_los_0p5`: 연구용, SITL localhost만 허용
2. `sim2swim_pool_a_0p1`: 실기 Case A, 0.10/0.40/0.15
3. `sim2swim_pool_c_0p05`: 실기 Case C, 0.05/0.15/0.08와 제한된 action/PWM

주석으로 `0.50 # 0.10`을 토글하거나 하나의 mission contract를 공유하지 않는다. sim profile은 별도 launch, 별도 safety YAML, `use_sim_time`, localhost SITL 연결을 요구하고 물리 autopilot 주소에서는 fail-closed하도록 설계한다.

## 4. Case A sim2real bag이 바꾼 우선순위

[`case_a_fault_20260814_170757`](../case_a_fault_20260814_170757)은 84.247 s/45,529 message의 실제 실험 기록이다. `/brov/mission/resolved`가 증명하는 실행 contract는 `0.10 m/s`, lookahead 0.40 m, reach 0.15 m, `takeoff_then_align`, loop true다. 즉 이 bag은 0.5 m/s 성능 증거가 아니라 **0.5로 올리기 전 이미 0.1에서 해결해야 하는 regression**이다.

### 확정된 실험 증거

| 항목 | bag 결과 |
|---|---:|
| active control | 59.040 s |
| takeoff P0→P1 | active state transition 기준 9.76 s에 도달 |
| outbound P1→P2 | active state transition 기준 18.00 s에 도달 |
| return P2→P1 | 명시적 stop 전 31.28 s 동안 미도달, 최소 0.495 m/종료 1.063 m |
| 목표/EKF-reported 속력 | 평균 약 0.100/0.151 m/s, reported max 0.463 m/s |
| velocity vector error | RMS 0.173 m/s, p95 0.314 m/s |
| position-velocity consistency | outbound 0.721 m, return 1.194 m residual |
| idx 1→2 command step | `v_d` 0.173 m/s, `q_d` 109.7° |
| 최대 자세오차 | 114.4°; 방향전환은 수행된 후순위 transition metric |
| 최소 한 action 축 exact ±1 | active sample 26.22% |
| 최소 한 T200 force clamp | active sample 11.19% |
| preview→sent | 1,456 sample 값 차이 0 |
| sent→servo | configured channel/reversal과 99.79% exact vector match |
| 종료 원인 | software fault가 아니라 명시적 `ros2cli` Trigger stop |

`LOCAL_POSITION_NED`의 동일 message에서 얻은 position 변위와 velocity 적분이 서로 맞지 않았다. 동시에 one-shot alignment는 계속 valid였지만 엄격한 초기화 이후 time-interpolated ArUco-vs-aligned-odometry position residual은 median 0.216 m, p95 0.904 m, max 1.486 m였다. 이는 estimator와 vision 중 어느 한쪽을 ground truth로 확정하는 증거는 아니지만, 초기 정렬을 공유한 뒤 갱신 경로가 분리된 두 state가 크게 달라도 제어 중 감시되지 않았다는 증거다.

localization은 `UNINITIALIZED→COLLECTING→INITIALIZED` 뒤 `INVALID` 전환 없이 끝까지 valid였고 aligned odometry 1,902개는 같은 epoch/session/alignment를 유지했다. 현재 validity는 vision residual이나 position/velocity consistency를 포함하지 않는다.

반대로 기록된 configured logical command 경로에서 preview→sent 값은 일치하고 sent→servo는 configured channel/reversal mask와 일치했다. 이는 실제 ESC 회전, thrust 방향·크기나 기계적 응답을 입증하지 않으므로 impulse/RPM/current/force 검증은 여전히 필요하다.

`control_active=false` 이후 action/preview는 각 264개 계산됐고 `258/264=97.73%`가 `|action|>=0.99`였지만 sent PWM은 0개였다. 이는 output gate가 작동했다는 증거이며, 포화 지표의 분모를 `control_active && output_enabled`로 한정해야 한다.

### 후속 상세 로드맵 P0

1. SITL ground truth로 DVL→EKF→`LOCAL_POSITION_NED` position/velocity frame·scale·timestamp를 검증한다.
2. 1/5/10 s `Delta p - integral(v dt)` monitor를 source time 기준으로 구현하고 reset·known correction window와 EKF correction term을 기록한다. 10 s residual 0.10 m는 G2 truth 및 restrained real test로 교정하기 전 provisional regression gate로만 둔다.
3. exact Case A P0→P1→P2→P1을 Gazebo G0/G1/G2에 구현한다.
4. model-based controller가 exact Case A return까지 통과하는지 먼저 확인한다.
5. total speed norm cap, braking, low-speed dwell과 quaternion slew를 구현한다.
6. aligned odometry와 quality-approved vision의 online residual을 기록하고 지속 불일치 정책을 정한다.
7. target progress, overspeed, attitude error와 saturation dwell을 `TRACKING_FAILURE`로 판정하는 runtime gate를 추가한다.
8. 위 항목을 통과한 뒤 frozen policy 0.5 Isaac parity로 진행한다.

position/velocity 불일치와 품질 미확인 vision outlier를 그대로 DR에 넣어 policy가 보상하게 해서는 안 된다. estimator의 유효 범위를 먼저 확정하고 그 안의 noise/delay만 학습에 사용한다.

## 5. 후속 단계 A-1: Isaac 0.5 기준선을 다시 생성한다

현재 최우선은 G0의 checkpoint 기반 0.10 m/s action trace다. 이 절의 배포 TorchScript 0.5 reference는 G0 0.10 실행과 동일 logger로 0.60 비교를 끝낸 뒤 수행하는 후속 artifact-parity 작업이다.

현재 플롯은 qualitative evidence일 뿐, 어떤 checkpoint가 현재 배포 `policy.pt`가 되었는지 metadata에 없다. 먼저 **동일 artifact**를 양 simulator에서 사용한다.

```text
policy SHA-256
0d89f3270f46214f1569b7d48dcb5e25363b1d9b7353b82ced0fc67c0093a472
```

필수 작업:

1. `test_policy.py`가 checkpoint뿐 아니라 배포 TorchScript를 직접 실행하도록 한다.
2. 현재 `policy.pt`로 Isaac 60 s reference를 다시 생성한다.
3. source commit, vehicle YAML SHA, policy SHA, seed, simulator version을 manifest에 저장한다.
4. 플롯 외에 매 40 ms raw trace와 JSON summary를 저장한다.

Isaac exact reference는 다음과 같다.

| 항목 | 값 |
|---|---|
| physics/policy rate | 100 Hz / 25 Hz |
| path | `(0,0,5) <-> (5,0,5)`, 5 m 왕복 |
| 시작 Z | waypoint Z와 동일한 5 m |
| speed | 0.5 m/s |
| lookahead | 1.0 m |
| reach | 0.5 m |
| heading | `align` |
| loop/duration | true / 60 s |
| action envelope | 학습과 같은 full `[-1,1]` |
| nominal plant | Isaac nominal, 평가 DR 고정 |

근거: `test_policy.py:52-55,95-100,107,124-148,310-340`, `guidance/los_guidance.py:47-60`.

첫 실험은 5 m 왕복 외에 충분히 긴 **단방향 직선**도 포함한다. 단방향이 통과하고 왕복만 실패하면 저수준 0.5 m/s 추종이 아니라 waypoint의 180° 명령 반전이 원인이다.

## 6. 단계 A-2: frozen policy 호환 계약과 현재 runtime 계약을 분리한다

기존 policy가 실제로 학습한 계약은 현재 `brov_ros2`와 다르다.

### 진단용 `legacy_exact_0p5`

- 16-D: `[q_e, v_e, omega, z_v, z_q]`
- quaternion `w>=0` canonicalization 없음
- 고정 `dt=0.04 s`
- 적분 clamp 없음, environment reset에서만 0
- waypoint 전환에서 적분 유지
- Z-up action을 별도 `T6` 변환 없이 SNAME `B+`에 전달
- 기존 단순 LOS, depth/terminal outer-loop 없음

이는 **격리된 simulator에서 기존 policy가 본 수치 경로를 재현하기 위한 compatibility mode**다. 현재 실기 runtime을 이 방식으로 되돌리라는 의미가 아니다.

### 배포 후보 `runtime_fixed_0p5`

- quaternion `w>=0`
- 실측 `dt`, 적분 ±5 clamp, stale 시 freeze
- current `start_heading` frame
- independent depth hold와 terminal hold
- frozen policy가 학습한 realized wrench를 보존하는 versioned legacy-action adapter
- 25 Hz 비동기 runtime과 watchdog

이 profile은 observation/guidance/runtime 수정의 영향만 보기 위한 것이다. 올바른 explicit `T6` action frame은 기존 policy에 임의 적용하지 않고 단계 B의 새 policy와 함께 도입한다. legacy quaternion도 Gazebo/MAVLink가 같은 자세의 부호를 임의로 뒤집지 않도록 직전 sample과 dot product가 양수가 되는 연속 표현을 사용하되, 이를 `w>=0` profile과 혼동하지 않는다.

두 결과를 비교한다.

- legacy와 current 모두 통과: 기존 policy를 sim2sim 기준선으로 사용할 수 있음
- legacy만 통과: runtime fix로 observation distribution이 바뀐 것이므로 기존 policy는 배포 후보가 아니며 재학습 필요
- 둘 다 실패: policy 또는 아직 맞지 않은 plant/action/guidance를 추가 분리

기존 action 경로는 `B`가 SNAME/FRD인데 `tau_cmd` 앞에 Z-up→SNAME 변환이 없다 (`envs/vel_env.py:101-143`). 작은 action에서 Isaac realized wrench는 대략 다음 부호를 갖는다.

```text
tau_zup_realized ~= diag(1,-1,-1,1,-1,-1) * (Fmax * action)
```

frozen-policy parity에서는 이 end-to-end 의미를 복제한다. 올바른 explicit `T6` action 계약은 단계 B의 새 정책에 적용한다. runtime 한쪽만 고치면 기존 policy가 학습한 sign을 깨뜨린다.

## 7. 단계 A-3: Gazebo integration을 세 층으로 나눈다

현재 Edo checkout에는 TorchScript/16-D observation/policy runner가 없고 Track B는 별도 고전 GNC loop다. Track B의 `gz topic` subprocess loop는 명목 20 Hz지만 로그상 약 3.51 Hz이므로 25 Hz policy 검증에 사용하면 안 된다.

### G0 — oracle-state/controller 분리

```text
Gazebo ground truth -> exact frame adapter -> observation builder
-> frozen policy -> actuator adapter -> Gazebo plant
```

- DVL/EKF/ArduSub estimator를 배제한다.
- 먼저 model-based controller로 같은 0.5 LOS/plant/actuator 경로가 안정한지 확인한다.
- model-based도 실패하면 policy가 아니라 frame, actuator, plant 또는 guidance 문제다.

### G1 — oracle-state + 실제 ArduSub actuation

- observation은 ground truth를 사용한다.
- PWM은 실제 `brov_ros2` RCPassThru/MAVLink/ArduPilotPlugin 경로로 보낸다.
- G0는 통과하고 G1이 실패하면 channel order, reversal, mixer, PWM mapping 문제다.

### G2 — full SITL runtime

- 실제 `brov_ros2` MAVLink telemetry, DVL/EKF 상태, observation node, policy node를 사용한다.
- Track B fake DVL/navigation 코드는 policy 입력에 사용하지 않는다.
- Gazebo truth와 `LOCAL_POSITION_NED` position/velocity를 동시에 저장하고, 각 state source의 frame·scale·source timestamp를 검증한다.
- 1/5/10 s `Delta p - integral(v dt)`와 estimator reset/innovation을 online metric으로 발행한다.
- G1은 통과하고 G2가 실패하면 state estimator, timestamp, frame, delay/dropout 문제다.

각 층에서 같은 policy SHA와 같은 mission/profile을 쓴다.

현재 G2의 direct mount/build, shadow, MAVLink/RCPassThru active 연결과 G1 성격의 GT-feedback/G2 EKF-feedback A/B까지 완료했다. GT-feedback translational tracking은 통과하고 EKF-feedback은 실패했으며 EKF `Delta p - integral(v dt)` residual도 크게 증가했다. 따라서 feedback 경로가 인과적 악화 요인임은 확인됐지만, 양쪽에 공통인 pitch action 포화 때문에 estimator를 유일 원인으로 단정하지 않는다.

full SITL은 Edo Track B가 아니라 `brov_bringup/rl_demo.launch.py`를 진입점으로 사용한다. 첫 0.5 기준선에는 pool localization이 필요하지 않으므로 camera/AprilTag를 강제하는 `sim2swim_demo.launch.py`보다 relative `start_heading` mission이 적합하다. QGC와 BROV용 UDP 포트를 분리하고, Edo extras interface·Track B controller·BROV gateway가 같은 endpoint/PWM을 동시에 소유하지 않게 한다.

추가 구현 blocker도 있다.

- **부분 완료:** 실기/SITL reversal profile 분리와 8-channel PWM→SERVO identity 전달은 검증됐다. 물리 6축 wrench 부호·gain은 T1~T8 impulse/basis test로 아직 고정해야 한다.
- **완료:** policy node가 raw NN/policy-limited action, desired/achieved wrench, requested/clamped force와 raw inverse PWM을 발행하며, obs node가 selected/GT source와 timestamp/age를 같은 bag에 기록한다.
- exact Case A P0→P1→P2→P1 scenario와 one-shot localization/marker dropout scenario를 구현하고, truth와 aligned odometry를 별도 trace로 남긴다.
- policy load 시 파일 존재만 보지 말고 metadata/schema/vehicle SHA/policy SHA를 검증한다.

Isaac reference의 literal Z=+5를 Gazebo에 그대로 넣으면 안 된다. Edo world는 수면이 Z=0, seabed가 Z=-10이므로, frame adapter가 Isaac의 env-local 시작점과 같은 **상대 경로**를 안전한 중층 위치(예: Gazebo Z=-5)로 옮겨야 한다. 시작점과 모든 수평 waypoint의 depth error는 0으로 유지한다.

## 8. 단계 A-4: plant와 actuator를 먼저 맞춘다

처음부터 Edo legacy plant를 robustness 증거로 사용하지 않는다. 다음 두 Gazebo model을 분리한다.

### `gazebo_calibrated`

Isaac nominal과 다음을 맞춘 parity model이다.

- mass 14.635 kg, inertia `[0.289,0.329,0.337]`
- volume/CoB와 복원모멘트
- 6축 added mass
- linear/quadratic damping
- T1~T8 위치·방향·열 순서
- T200 +64.1/-51.5 N 비대칭, deadband 0.075, time constant 0.05 s

### `gazebo_legacy_ood`

기존 13 kg, zero added mass/linear damping, 선형 ±50 N actuator를 유지한다. calibrated model 통과 후 model mismatch stress로 사용한다.

차이는 0.5 m/s에서 이미 크다. surge drag의 절댓값만 비교하면 Edo stock은 `58.42*0.5^2 = 14.6 N`, Isaac nominal은 `13.7*0.5 + 141*0.5^2 = 42.1 N`으로 약 2.9배다. stock Edo에서 잘 움직이는 것만으로 Isaac과 같은 0.5 m/s authority를 검증했다고 볼 수 없다.

`brov_ros2`가 계산한 T200 inverse PWM을 Gazebo의 선형 PWM→±50 N plugin에 그대로 보내면 실제 힘이 다르다. 다음 중 하나가 필요하다.

1. Gazebo plugin을 T200 forward curve와 lag를 포함하도록 수정하거나
2. `runtime PWM -> T200 forward/lag -> desired N -> Gazebo linear command` adapter를 둔다.

T1~T8 각각 정/역 impulse와 6축 basis action을 실행해 기대 주축 부호가 모두 일치한 뒤 closed-loop로 넘어간다.

## 9. 단계 A-5: 0.5 m/s 안정화는 parity 뒤 별도 profile로 수행한다

Isaac legacy LOS의 2점 왕복은 waypoint에서 다음을 동시에 즉시 반전한다.

- desired velocity 약 +0.5 → -0.5 m/s
- align target yaw 약 180°
- 기존 방향에서 누적된 integral은 유지

따라서 직선 순항이 안정해도 전환에서 큰 sway/heave/pitch transient가 생길 수 있다. `legacy_exact_0p5` 결과를 보존한 뒤 `stabilized_0p5`에서 다음을 A/B 시험한다.

1. 시작 시 0→0.5 m/s acceleration limit
2. waypoint 접근 시 측정된 감속도로 speed ramp-down
3. 2점 왕복에서는 정지→shortest-path attitude slew→반대 방향 ramp-up
4. square에서는 lookahead를 다음 segment까지 이어 계산하거나 corner fillet 사용
5. 일반 align 자세에도 quaternion shortest-path slew 적용
6. horizontal/depth 성분을 합친 뒤 total desired-speed norm을 profile 속력 이하로 다시 제한
7. reach에 낮은 actual speed와 짧은 dwell 조건 추가
8. Case A 수평 leg에서 level/yaw-only와 기존 3D align을 A/B 시험
9. independent depth hold 유지
10. waypoint 전환의 integral `유지/reset/leaky decay`를 별도 variant로 비교

감속 시작 거리는 고정 magic number 대신 다음 식과 식별된 감속도로 정한다.

```text
d_stop = v^2 / (2 * a_brake)
```

`reach_threshold`, delay 동안의 이동거리와 margin을 합쳐 waypoint 전환 전에 감속을 끝낸다. command shaper의 acceleration, yaw-rate와 integral 규칙은 단계 B 학습 curriculum에도 동일하게 넣는다.

다음 변경은 안정화로 허용하지 않는다.

- policy 출력에 경험적 per-axis gain/sign 추가
- zero-state bias를 임의 PWM offset으로 상쇄
- action/PWM clamp로 발산을 숨기고 Isaac parity라고 보고
- 서로 다른 policy/checkpoint 결과를 한 그래프에서 비교

### 첫 안정화용 최소 profile

Isaac exact 왕복과 별도로 다음 profile을 먼저 통과시키면 속도·제동·종점 hold를 가장 작게 진단할 수 있다.

```yaml
waypoints: "0,0,0;4,0,0"
waypoint_frame: start_heading
heading_mode: straight
cruise_speed: 0.50
lookahead_dist: 1.00
reach_threshold: 0.15
loop: false
depth_hold_kp: 0.8
depth_speed_limit: 0.05
terminal_hold_kp: 0.5
terminal_speed_limit: 0.05
horizontal_accel_limit_mps2: 0.10
horizontal_decel_limit_mps2: 0.10
```

상대 waypoint Z=0은 mission 시작 depth를 유지한다. `straight`는 yaw/속도 문제를 분리하고, 4 m는 현재 generic safety의 segment 상한 안이다. 0.10 m/s²라면 0→0.5 가속과 감속에 각각 5 s, 1.25 m가 필요해 약 1.5 m의 정상속도 구간이 남는다. 이 profile을 0.10→0.25→0.50 m/s로 commissioning한 뒤 `align`, 5 m 왕복, square 순으로 확장한다.

## 10. 단계 A 실행 순서와 원인 판정

| 순서 | 시험 | 실패 시 우선 원인 |
|---:|---|---|
| 0 | config/doc/test 일치, artifact SHA 고정 | profile/provenance |
| P0-1 | Case A fault bag offline metric 재생 | analyzer/schema/판정 계약 |
| P0-2 | G2 truth↔DVL/EKF position/velocity consistency | estimator/frame/scale/timestamp |
| P0-3 | exact Case A를 model-based G0→G1→G2로 수행 | guidance/estimator/actuator/full-stack |
| 1 | Isaac 동일 TorchScript 60 s reference | policy 자체 또는 export |
| 2 | offline LOS→obs→action→PWM golden replay | 수식/frame/artifact adapter |
| 3 | Gazebo T1~T8 impulse와 6축 basis | channel/frame/mixer/actuator |
| 4 | calibrated Gazebo + model-based 0.5 | plant/guidance/integration |
| 5 | G0 fixed body velocity 0.5 | 저수준 policy/actuator |
| 6 | G0 단방향 LOS 0.5 | LOS/frame/initialization |
| 7 | G0 5 m 왕복 60 s | turn discontinuity/integral |
| 8 | G1 ArduSub actuation | PWM/backend/channel |
| 9 | G2 full SITL estimator | estimator/timing/DVL frame |
| 10 | stabilized 0.5 straight/out-back/square | command shaping |
| 11 | legacy OOD plant | DR/model robustness |

단계 5가 실패하면 LOS를 튜닝하지 않는다. 단계 6은 통과하고 7만 실패하면 network weight보다 turn shaping을 먼저 고친다. calibrated는 통과하고 legacy OOD만 실패하면 그 차이를 단계 B DR 범위로 넘긴다.

## 11. 단계 A trace와 통과 기준

매 40 ms 다음을 한 행으로 저장한다.

```text
source_timestamp, receive_timestamp, dt, activation_generation, command_sequence,
ground_truth_position/velocity, estimated_position/velocity,
position_velocity_residual_1s/5s/10s,
position, quaternion, v_body, omega_body,
v_desired_body, q_desired, observation,
raw_unclamped_nn_action, policy_clamped_action, operationally_limited_action,
desired_wrench, thruster_force_requested/clamped,
allocation_residual_pre_clamp, allocation_residual_post_clamp,
pwm_inverse_raw, pwm_shaped_preview, pwm_sent, servo_output,
model_predicted_wrench, measured_wrench_proxy, waypoint_index,
control_active, output_enabled, output_gate_reason,
resolved_action/pwm_limits, reversal_mask, policy/vehicle/config_hash,
servo_mavlink_source_timestamp,
saturation/slew/clamp flags, estimator health,
vision_visible/quality/residual, target_progress,
reset/fault, run_outcome
```

### 계약 hard gate

- 동일 input의 단계별 deterministic golden replay에서 observation/action/PWM `max_abs_error <= 1e-5`
- T1~T8 정/역과 6축 basis의 기대 주축 부호 100% 일치
- NaN, reset, collision, bottom/surface contact, growing oscillation 0회
- G0/G1 25 Hz parity는 interval median 40 ms, p99 44 ms 이하/drop 0; G2는 평균 24–26 Hz, p95 60 ms 이하, max 120 ms 이하; active command gap 0.25 s는 별도 watchdog ceiling
- `control_active && output_enabled` 정상상태(명시적 accel/brake/q-slew/waypoint 전환 window 제외)에서 raw NN clip, operational action cap, T200 requested-force clamp, PWM absolute/slew limit을 각각 계산하고 개입률 1% 미만
- pre-clamp reproduction residual과 post-clamp authority residual을 분리한다. `S_tau=diag(85 N,85 N,120 N,26 Nm,14 Nm,22 Nm)`로 규격화하고 force clamp가 없는 구간에서 `||S_tau^-1*(B*f_clamped-tau_cmd)||/max(||S_tau^-1*tau_cmd||,0.01) < 5%`
- reset·known correction이 없는 source-time window에서 G2 10 s `||Delta p-integral(v dt)|| <= 0.10 m`; G2 truth/real 교정 전 provisional
- `2*acos(clamp(abs(dot(q_d[k],q_d[k-1])),0,1)) <= omega_slew*Delta t+tolerance` 위반 0회
- activation generation마다 첫 preview discard 정확히 1회, 이후 active preview→sent 1:1 일치, inactive sent 0회, initial neutral 제외 configured servo vector 100% 일치, deactivation 후 0.25 s 이내 neutral echo

### Case A bag regression gate

- P0→P1→P2→P1 full cycle 5회 연속 성공
- horizontal lateral RMS/max 0.15/0.30 m 이하
- 수평 P1↔P2 leg의 waypoint 전환·command-slew 제외 구간에서 depth RMS/max 0.10/0.20 m 이하
- 0.10 profile에서 `p95(max(0,||v_reported||-||v_d||)) <= 0.05 m/s`; 별도로 `max(||v_reported||) <= 0.20 m/s`
- `control_active && output_enabled` 정상상태(명시적 accel/brake/q-slew/waypoint 전환 window 제외)의 operational action cap/T200 force clamp는 각각 1% 미만, 전체 profile은 5% 미만, 0.2 s 초과 연속 cap 0회
- target distance/error 증가 또는 along-track progress 정체·감소가 profile별 configured dwell을 넘으면 `TRACKING_FAILURE`로 종료하고, `control_active=false→sent 중단→neutral echo→disarm request→outcome` 순서를 자동 검증

### Isaac reference 상대 gate

- velocity vector RMSE: `Gazebo <= max(1.2*Isaac, Isaac+0.02 m/s)`
- cross-track RMS: `Gazebo <= max(1.2*Isaac, Isaac+0.05 m)`
- attitude geodesic RMSE: `Gazebo <= max(1.2*Isaac, Isaac+2 deg)`
- waypoint 전환 시각: Isaac 대비 ±10%

Isaac raw reference가 없을 때만 다음 잠정 절대값을 사용한다.

- settling 후 speed 0.50±0.05 m/s
- velocity vector RMSE 0.10 m/s 이하
- cross-track RMS 0.25 m 이하, max 0.50 m 이하
- 수평 직선 depth error 0.20 m 이하
- straight/level target 대비 roll/pitch 또는 geodesic tracking error 10° 이하(command-slew window 제외)
- 마지막 세 개 5 s window의 error RMS가 증가하지 않음

이 값은 기존 프로젝트의 확정 요구사항이 아니라 최초 자동 gate 제안이다. Isaac reference와 bag 분석 후 고정한다.

### `stabilized_0p5` 추가 gate

| 지표 | 잠정 기준 |
|---|---:|
| `||v_d||>=0.45` 구간 목표방향 투영속도 `v_parallel=v·v_d/||v_d||` 평균 | 0.45–0.55 m/s |
| 속도 vector RMSE | 0.08 m/s 이하 |
| cross-track RMS / max | 0.15 / 0.30 m 이하 |
| 수평·level 구간 depth error RMS / max | 0.10 / 0.20 m 이하 |
| straight/level target 대비 roll/pitch/yaw tracking error max(command-slew 제외) | 각 10° 이하 |
| 종점 overshoot | 0.30 m 이하 |
| 종점 정착 | 5 s 안에 0.15 m 및 speed 0.05 m/s 이내 |
| 전체 profile operational action/PWM cap 체류율 | 5% 미만(정상상태는 1% 미만) |
| 반복성 | 같은 조건 5회 성공 |

G0/G1 nominal에서는 policy interval median 40 ms, p99 44 ms 이하와 drop 0을 목표로 한다. G2 full SITL에서는 observation/PWM 평균 24–26 Hz, p95 interval 60 ms 이하, 최대 120 ms 이하, Gazebo real-time factor 0.9 이상을 잠정 gate로 두며, 0.25 s watchdog은 별도의 hard safety ceiling으로 유지한다.

## 12. 재학습 구현 범위와 시작 조건

### 현 deadline의 최소 조건

기존 문서와 실험 결과만으로 재학습 사유는 이미 충분하다. frozen policy의 0.10/0.60 결과는 시작 허가가 아니라 새 정책과 비교할 baseline이다. 다음 항목을 구현한 뒤 최소 재학습을 시작한다.

1. 학습과 runtime이 공유할 16-D observation의 quaternion hemisphere, 적분 clamp/reset/stale-freeze 규칙을 고정한다.
2. action frame/sign을 legacy 보존 또는 explicit `T6` 중 하나로 결정하고 양쪽에 같은 golden vector를 적용한다.
3. `0/0.05/0.10/0.50/0.58–0.63`, hold·stop·restart·reversal command sampler를 구현한다.
4. velocity-priority reward, overspeed/cross-axis/saturation metric을 구현한다.
5. 5 s 안에 한 command transition을 넣고 PPO horizon을 128 step으로 확대한 뒤, 장기 validation 실패 시에만 15 s fine-tune한다.
6. Isaac straight/hold/reversal과 exact Gazebo Case A의 JSON/CSV evaluator 및 artifact lineage를 구현한다.

### 후속 full-roadmap 조건

아래는 최종 일반화·시스템 승인 조건이며, 이번 최소 재학습을 막지 않는다.

최종 일반화 policy artifact를 승인하기 전에는 다음을 모두 만족한다.

1. 0.5/0.1/0.05 profile이 서로 분리되고 test/doc/config가 일치한다.
2. frozen artifact가 Isaac과 calibrated Gazebo에서 동일 trace로 평가된다.
3. 6축 action과 8 thruster frame/sign이 golden test로 고정된다.
4. Gazebo 25 Hz async runtime과 T200 adapter가 완성된다.
5. 실기 bag에서 latency, bias, saturation, heave/ballast 영향을 계산한다.
6. G2 position/velocity consistency가 truth 및 실기 provisional gate를 통과한다.
7. exact Case A를 model-based controller가 return까지 통과한다.
8. speed/q command shaper와 heading contract가 확정된다.
9. 0.5 straight와 turn 실패가 구분된다.
10. metric JSON과 자동 pass/fail harness가 있다.

이 full-roadmap 조건은 최종 sim2real 일반화 전에 닫는다. 현 최소 재학습에서는 확인된 frame/sign/estimator 오류만 수정하고, 미확인 시스템 요소를 policy DR로 덮지 않는다.

## 13. 단계 B-1: 새 observation/action 계약

### Observation

비교를 위해 두 candidate를 학습한다.

#### V2-A — fixed 16-D

```text
[quat_unique(q_d^-1*q), v_error, omega,
 clamp(integral(v_error), ±5),
 clamp(integral(q_error_vector), ±5)]
```

현재 runtime과 shape를 유지한 비교 기준이다.

#### V2-B — 구조적 개선 후보인 19-D

V2-A에 `v_desired_body_zup(3)`를 추가한다. 현재 16-D만으로는 다음 두 상태가 policy에 같게 보인다.

```text
hover:             v=0,   v_d=0,   v_error=0
0.5 m/s steady:    v=0.5, v_d=0.5, v_error=0
```

두 번째는 항력 상쇄 action이 필요하지만 첫 번째는 그렇지 않다. 실기에서 확인된 zero-observation sway bias 약 `+0.232`를 고려하면 command-conditioned observation을 우선 검토해야 한다. 다만 현 일정에서는 V2-A parity smoke retrain과 병렬 비교할 선택사항으로 두고, V2-A의 zero-bias/저속 gate가 실패하면 V2-B를 최종 후보로 승격한다.

actuator memory가 추가로 필요하면 previous action 6개를 포함한 V2-C 또는 recurrent policy를 ablation한다. schema가 달라지면 `policy_node`의 입력 차원, metadata와 golden test를 함께 올린다.

state age, localization validity와 position/velocity consistency는 policy가 임의로 보상할 관측으로 넣기보다 control safety gate로 사용한다. 현재 `localization/valid=true`도 vision/position-velocity consistency failure와 공존할 수 있으므로 별도 `consistency_health`와 fault reason을 추가한다. 기존 status invalid 또는 새 consistency gate fail sample에서는 observation 적분과 waypoint 진행을 freeze하고 neutral/fault 규칙을 적용한다.

### Action

새 policy는 다음 explicit 계약으로만 학습한다.

```text
action FLU/Z-up
-> per-profile action limit
-> wrench scale [85,85,120,26,14,22]
-> T6=diag(1,-1,-1,1,-1,-1)
-> SNAME B+
-> per-thruster force saturation
-> T200 inverse
-> PWM abs/slew
-> actuator lag/deadband
```

policy가 요청한 wrench와 saturation 후 achieved wrench를 모두 reward/log에 사용한다.

## 14. 단계 B-2: command/guidance curriculum

재학습이 0.5만 잘하고 저속에서 bias를 내거나, 저속만 잘하고 0.5에서 느려지지 않도록 명령 bin을 균형 있게 샘플한다.

1. `C0 hold`: zero velocity, identity/작은 attitude, terminal/depth hold
2. `C1 low`: 0.01–0.05 m/s 단일축과 Case C 0.05
3. `C2 pool`: exact Case A 0.10, P0→P1→P2→P1 full return, depth ±0.05
4. `C3 0.5 LOS`: 5 m straight, out-and-back, square, command shaper
5. `C4 mixed`: episode 중 0→0.05→0.10→0.5→0, q/v 독립·동시 변경
6. `C5 stress`: wider attitude, DR corners, sensor dropout, current/tether

핵심 변경:

- q command를 episode 시작에 한 번만 고정하지 않고 중간에 rate-limited 변경
- Eq. 9는 velocity template이 아니라 Frenet–Serret `q_d(t)`로 교정하고, exact 0.5 m/s sphere command와 runtime형 속도 명령을 함께 학습
- start ramp, waypoint turn, terminal hold와 외란 후 복귀 포함
- 5 s episode 안 2–3 s에 한 transition을 넣고, 60/120 s Case A/LOS는 우선 validation으로 사용. 장기 적분 실패가 확인될 때만 15 s fine-tune
- 60/120 s rollout과 방향별 outbound/return metric은 validation에 반드시 포함

## 15. 단계 B-3: DR와 센서/시간 모델

### Dynamics/actuator

Nominal T200 deadband, forward/reverse 비대칭, saturation과 약 50 ms lag는 이미 구현돼 있다. 아래 목록에서 미구현인 것은 주로 파라미터 randomization과 plant 범위 확장이다.

- mass, COM, inertia, 실제 ballast
- volume, CoB, fluid density
- 6축 added mass, linear/quadratic damping
- current와 식별된 tether disturbance
- thruster 위치/축, 개별 gain, forward/reverse 비대칭
- deadband, time constant, saturation, battery, 추진기 열화

### Observation/runtime

- Water Linked의 실제 raw packet cadence/range mode, source-to-FCU latency, noise/FOM/dropout을 실측한 뒤 반영. 기존 15 Hz/100 ms는 provisional stress ablation으로만 유지
- DVL/IMU bias, noise, outlier, dropout와 bottom-lock loss
- bag empirical 25 Hz jitter: median 약 40.0 ms, p95 44.5 ms, p99 72.6 ms, max 약 79.8 ms
- timestamp skew, computation/communication latency
- 0.2 s transient gap과 장기 stale neutral/fault
- held sample에서는 integral과 waypoint 진행 freeze

DR 범위는 임의로 넓히지 않고 bag, restrained step, calibrated-vs-legacy Gazebo 식별 결과로 설정한다. bag에서 관측된 position/velocity 불일치는 현재 status가 invalid였다는 뜻이 아니라, 새 consistency regression이 반드시 fail해야 할 사례로 분류하며 policy DR로 만들지 않는다. AprilTag/single-marker loss·outlier·capture delay는 16/19-D low-level policy DR가 아니라 full-stack localization 시험으로 분리한다. `SERVO_OUTPUT_RAW`의 약 76 ms recorder 지연도 실제 T200 motor lag로 사용하지 않는다.

## 16. 단계 B-4: reward, episode와 PPO

현재 velocity reward는 `0.2*exp(-||v_e||²)`이고 action reward는 `0.3*exp(-||a||)`이다 (`envs/vel_env.py:313-328`).

- 0.05 m/s를 전혀 추종하지 않아도 `exp(-0.05²)=0.9975`라 거의 최대다.
- 0.5 m/s를 전혀 추종하지 않아도 velocity 항은 완전 추종보다 약 0.044만 감소한다.
- 작은 action 사용으로 얻는 reward가 tracking 개선보다 커질 수 있다.

따라서 다음처럼 바꾼다.

- velocity, attitude, angular-rate error를 물리 acceptance tolerance로 정규화
- cross-axis velocity와 `||v|| > ||v_d||+margin` overspeed를 별도 penalty/metric으로 사용
- heave/depth error를 별도 기록하고 필요한 가중치 부여
- quaternion geodesic shortest-path error 사용
- effort는 tracking 성공 뒤 작은 cost가 되도록 축소/조건화
- delta action, delta PWM, saturation, allocation residual, integral clamp penalty
- NaN, out-of-bound, collision, 비정상 depth/attitude에 명시적 termination cost

PPO/네트워크 조정은 계약 수정 뒤 수행한다.

- observation normalization을 쓰면 normalizer를 artifact/runtime에 포함
- 2.56 s인 현재 rollout horizon이 최소 하나의 command transient/settling을 포함하도록 확대
- entropy 0과 std collapse를 확인하고 entropy/min-std를 sweep
- 64×64와 더 큰 network는 동일 seed/validation으로 ablation
- 여러 training seed와 fixed holdout validation으로 checkpoint 선택

## 17. 단계 B 승인 matrix

| 분류 | 필수 시험 |
|---|---|
| 정적 | q/-q, yaw ±pi, zero bias, velocity/attitude error sweep |
| Isaac nominal | fixed velocity, legacy exact 0.5, stabilized 0.5 |
| Isaac DR | nominal seeds, held-out seeds, corner cases |
| Gazebo calibrated | G0/G1/G2 0.5 straight/out-back/square |
| Gazebo legacy OOD | 동일 scenario에서 model mismatch robustness |
| 저속 | pool A 0.10, pool C 0.05, depth/takeoff/terminal hold |
| Case A bag regression | exact P0→P1→P2→P1, estimator consistency, command step/saturation |
| localization | one-shot alignment, marker loss/reacquisition, truth-vs-aligned residual |
| lifecycle/operator abort | explicit stop→STOPPING→IDLE, neutral/disarm request; hardware ACK는 별도 검증 |
| safety fault | jitter, DVL dropout/stale, consistency fail, thruster degradation→SAFETY_FAULT |

최소 gate:

- observation/action/PWM golden error `<=1e-5`
- hover에서 불필요한 normalized action bias 각 축 `|a|<=0.05` 잠정 기준
- 0.5 m/s는 단계 A의 absolute/Isaac-relative gate 이상
- 저속 A/C는 각 reach/dwell/action/PWM envelope 충족
- reset·known correction 없는 source-time 10 s position/velocity consistency residual 0.10 m 이하의 truth-calibrated provisional gate 충족
- Case A P1→P2→P1 5 cycle 연속 성공과 tracking-failure 판정 정상
- held-out DR 100 seeds에서 잠정 success 95% 이상
- NaN/reset/collision 및 frame/sign mismatch 0
- 60/120 s에서 growing oscillation과 integral windup 없음

## 18. Artifact와 최종 산출물

기존 `policy.pt`를 덮어쓰지 않는다.

```text
artifacts/policies/<policy_version>/
  policy.pt
  metadata.yaml
  golden_vectors.npz
  validation_summary.json
  training_config.yaml
  dr_config.yaml
```

metadata에는 다음을 포함한다.

- source/checkpoint commit과 seed
- observation schema/shape/수식, clamp와 normalization
- action frame/order/range, wrench scale
- guidance/command-shaper version
- vehicle/T200/Gazebo adapter SHA
- simulator/RSL-RL/PyTorch version
- validation profile과 결과
- TorchScript SHA-256

runtime은 schema, vehicle hash와 policy hash가 맞지 않으면 fail-closed해야 한다.

## 19. 최종 작업 순서

1. **완료:** `model_299.pt` Isaac straight 0.10/60 s가 평균 velocity error 0.0074 m/s로 안정적임을 확인했다.
2. **완료(실패 기준선):** `brov_ros2` direct full-SITL Case A의 straight leg에서 목표/실제 속력 평균 0.101/0.331 m/s와 횡속도 RMS 0.320 m/s를 확인했다.
3. **부분 완료:** active synchronized GT logger/source와 runtime 축별 action/allocation logger는 구현했다. Isaac `test_policy.py`의 동일 schema 축별 logger는 G0에 남아 있다.
4. **완료:** G0B 0.50 m/s A/B에서 GT translational tracking PASS, EKF tracking FAIL을 확인했다.
5. **완료:** stock GPS를 제거하고 Water Linked-aligned no-GPS VPD/EKF3 경로와 fresh full-cycle GT/DVL A/B를 구현했다. DVL은 feedback gap을 줄였지만 common-mode action jitter를 제거하지 못했다.
6. **완료:** G1–G7의 1차 구현, headless 50-iteration `deploy_v2` 학습과 MK2 metadata/T6 배포 경로를 만들었다.
7. **완료(FAIL):** MK2는 Isaac steady 0.5 m/s를 통과했으나 fresh Gazebo GT/DVL Case-A-shaped 실행에서 각각 whole action cap 98.9/98.4%로 실패했다. artifact는 격리하고 실기 배포를 금지했다.
8. **현재 작업:** bounded-action/pre-clamp penalty, 15--20 s Case-A recovery curriculum과 측정 기반 최소 plant DR로 새 후보를 짧게 재학습한다. visualization은 계속 OFF한다.
9. 새 후보가 Gazebo GT Case A 3회 연속 통과한 뒤 DVL-EKF 3회를 통과해야 한다. 그 뒤에만 동일 artifact/config으로 실기 Case A를 단계적 speed ramp와 abort gate 아래 재시도한다.

DVL sample-and-hold, 소규모 mass/damping/thruster-gain DR과 19-D observation은 최종 sim2real 후보 전 권장 항목이다. full current/tether/failure DR, localization/fault matrix, Case C와 100-seed 승인은 별도 backlog으로 유지한다.

## 주요 근거

- [`CASE_A_FAULT_20260814_170757_ANALYSIS.md`](CASE_A_FAULT_20260814_170757_ANALYSIS.md): 실기 Case A bag 정량 분석과 regression 요구사항
- [`../case_a_fault_20260814_170757/metadata.yaml`](../case_a_fault_20260814_170757/metadata.yaml): 실험 bag metadata
- [`test_policy.py`](test_policy.py): Isaac 0.5 m/s LOS 검증
- [`guidance/los_guidance.py`](guidance/los_guidance.py): legacy LOS
- [`envs/vel_env.py`](envs/vel_env.py): 학습 observation/action/reward/DR
- [`envs/vel_env_cfg.py`](envs/vel_env_cfg.py): rate, episode와 command 설정
- [`../robots/data/BROV2/brov2_heavy.yaml`](../robots/data/BROV2/brov2_heavy.yaml): Isaac nominal plant/T200/sensor
- [`../../brov_ros2-main/docs/SIM2SWIM_DEMO.md`](../../brov_ros2-main/docs/SIM2SWIM_DEMO.md): 실기 A/C profile
- [`../../brov_ros2-main/brov_base/brov_base/observation.py`](../../brov_ros2-main/brov_base/brov_base/observation.py): current runtime observation
- [`../../brov_ros2-main/brov_base/brov_base/guidance.py`](../../brov_ros2-main/brov_base/brov_base/guidance.py): current depth/terminal guidance
- [`../../brov_ros2-main/brov_localization/brov_localization/localization_node.py`](../../brov_ros2-main/brov_localization/brov_localization/localization_node.py): one-shot pool alignment와 validity
- [`../../brov_ros2-main/brov_control/brov_control/policy_node.py`](../../brov_ros2-main/brov_control/brov_control/policy_node.py): runtime inference/allocation/PWM
- [`../../brov_ros2-main/runtime/experiments/sim2sim_brov_ros2_0p1_smoke_20260817_021003`](../../brov_ros2-main/runtime/experiments/sim2sim_brov_ros2_0p1_smoke_20260817_021003): direct full-SITL rosbag, Gazebo capture와 정량 summary
- [`../../brov_ros2-main/brov_bringup/config/mission_manager_sim2swim_a.yaml`](../../brov_ros2-main/brov_bringup/config/mission_manager_sim2swim_a.yaml): 현재 0.5 config
- [`../../brov_ros2-main/brov_bringup/test/test_sim2swim_contract.py`](../../brov_ros2-main/brov_bringup/test/test_sim2swim_contract.py): 0.1 contract test
- [`../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/TRACK_B.md`](../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/TRACK_B.md): 기존 Track B 결과와 한계
- [`../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/SITL_Models/Gazebo/models/bluerov2_heavy/model.sdf`](../../Edo_Project/gazebosim_bluerov2_ardupilot_sitl/SITL_Models/Gazebo/models/bluerov2_heavy/model.sdf): Gazebo plant/ArduPilot mapping
