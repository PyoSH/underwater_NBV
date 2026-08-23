# Case A sim2real rosbag 정량 분석

작성 기준: 2026-08-17

분석 대상: [`case_a_fault_20260814_170757`](../case_a_fault_20260814_170757)

연결 문서: [`SIM2SIM_05MS_RETRAIN_PLAN.md`](SIM2SIM_05MS_RETRAIN_PLAN.md)

## 1. 결론

이 rosbag은 **0.5 m/s 실험이 아니라 0.10 m/s sim2swim Demo (a), Case A 실험**이다. 파일명에 `fault`가 있지만 runtime이 검출한 software safety fault로 종료된 것이 아니다. bag 단독으로는 Trigger 요청자가 사람인지 script인지 구분할 수 없지만, 사용자가 **선속도·위치 추종 불량을 보고 수동 stop했음**을 확인했다. 따라서 실험 outcome은 `OPERATOR_ABORT(TRACKING_CONCERN)`이다. stop 요청 후 IDLE에 진입했고 neutral/disarm이 요청됐으며 servo 8채널 neutral은 확인됐다. hardware disarm ACK는 bag에 없다.

실험은 takeoff와 첫 outbound leg에 도달했고, 사용자 관측상 P2 종점에서 반대 방향으로 돌아서는 동작도 수행했다. 다만 이후 선속도·위치 추종이 불량해 사용자가 stop했고, 그 시점까지 31.28 s 동안 return waypoint에는 도달하지 못했다. 따라서 `return 미도달`은 방향전환 실패의 증거가 아니라, 수동 종료 시점까지의 translational tracking 결과다.

1. 0.10 m/s 명령인데도 EKF-reported 속력은 평균 0.151 m/s, 최대 0.463 m/s였고 velocity vector error RMS는 0.173 m/s였다.
2. `LOCAL_POSITION_NED`의 position 변위와 velocity 적분이 outbound에서 0.72 m, return에서 1.19 m 불일치해, 실기 position 결과에 estimator 문제가 혼재할 수 있다.
3. active sample의 26.2%에서 최소 한 action 축이 정확히 ±1에 포화되고, 11.2%에서 최소 한 thruster의 offline 재구성 요구추력이 현재 code/config의 T200 model clamp 범위를 넘는다.
4. 기록된 configured logical command 경로에서 preview→sent 값은 일치하고, sent→servo는 configured channel/reversal mask와 일치한다. 이는 실제 ESC 회전, thrust 방향·크기 또는 기계적 응답을 검증하지 않는다.
5. return 전환에서 `q_d` 109.7° one-sample step과 최대 자세오차 114.4°가 기록됐다. 다만 사용자 관측상 자세와 방향전환은 대체로 수행됐으므로, 이는 주요 실패 원인이 아니라 후순위 transition metric으로 두어야 한다.

기존 문서에는 observation 불일치, 저속/stop/reversal command 부재, velocity reward의 낮은 민감도, 짧은 episode/horizon, action 계약과 validation/manifest 부재가 이미 재학습 gap으로 기록돼 있다. 따라서 **재학습은 정확도·robustness 보완을 위해 진행한다.** 다만 2026-08-17 같은 actor를 Isaac nominal 0.10 m/s에서 직접 실행했을 때 평균 velocity error 0.0074 m/s이고 180° 전환 뒤에도 지속 요동이 없었다. 따라서 Gazebo·실기 요동을 저속 OOD 단독 원인으로 보지 않고 deploy feedback/frame/timestamp와 plant/action 경로를 먼저 분리한다. 광범위 localization·fault matrix·plant 전면 식별은 현 deadline에서 후순위다.

## 2. 분석 범위와 한계

[`metadata.yaml`](../case_a_fault_20260814_170757/metadata.yaml)에 기록된 bag 범위는 다음과 같다.

| 항목 | 값 |
|---|---:|
| duration | 84.247302 s |
| messages | 45,529 |
| active control | 59.039926 s |
| observation/action samples | active 각 1,457 |
| sent PWM samples | 1,456 |

`rosbag2` SQLite CDR을 직접 역직렬화하고 모든 시각을 bag 시작에 대한 상대시각으로 정렬했다. custom message 정의는 `brov_interfaces/msg`와 동일한 schema를 사용했다.

해석에는 다음 한계가 있다.

- `/brov/localization/odometry_pool`은 ground truth가 아니라 controller가 사용한 EKF odometry의 one-shot 변환 결과다.
- ArUco pose에도 covariance, reprojection error와 detection quality가 기록되지 않았다.
- `Float32MultiArray` debug topic에는 source timestamp가 없어 수 ms 단위 지연은 recorder 수신시각 기준이다.
- `/brov/action`은 이미 `[-1,1]`로 clamp된 값이므로 raw unclamped NN 출력의 크기는 알 수 없다.
- `SERVO_OUTPUT_RAW`는 autopilot output echo이지 ESC RPM, 전류 또는 실제 thrust가 아니다.
- 실제 물리 thrust와 독립 ground truth가 없으므로 최종 motor force 방향·크기와 유체계수는 이 bag 하나로 식별할 수 없다.

## 3. 실제 실행 contract

`/brov/mission/resolved`의 immutable contract는 다음과 같다.

```text
contract       brov_pool_position_mission_v1
cruise_speed   0.10 m/s
lookahead      0.40 m
reach          0.15 m
heading        takeoff_then_align
loop           true
```

pool-frame path는 다음과 같다.

```text
P0 = (1.3115, 0.7045, 0.3567)
P1 = (1.3115, 0.7045, 0.7000)  # 0.3433 m 수직 이동
P2 = (2.8115, 0.7045, 0.7000)  # 1.5000 m 수평 이동
```

현재 source checkout의 `mission_manager_sim2swim_a.yaml`에는 `0.50 # 0.10`이 있지만 이 실행의 resolved contract는 0.10이다. 다음 실험에서는 source YAML의 문자열이 아니라 다음을 함께 저장해야 한다.

- 실제 `/brov/mission/resolved`
- 설치된 ROS share에 있는 parameter file의 SHA-256
- source commit과 dirty 상태
- launch argument와 ROS parameter dump
- policy, vehicle, mission 및 safety config SHA-256

Case A의 `loop=true`는 P0→P1 뒤 P1↔P2를 반복한다. 따라서 `/brov/mission_complete=false`가 계속된 것은 그 자체로 fault가 아니다. 판정은 명시적 stop 전 return leg 미도달과 추종오차를 사전 정의한 regression gate에 비교해야 한다.

## 4. 실험 시간축

| 상대시각 | 사건 | 해석 |
|---:|---|---|
| 4.653 s | camera tilt neutral 확인 후 localization 수집 | 이전 sample clear |
| 7.360 s | localization `INITIALIZED`, epoch 1 | 20 inlier one-shot alignment |
| 7.430 s | resolved mission 기록 | 0.10/0.40/0.15 contract |
| 14.643 s | `control_active=true` | 실제 active 시작 |
| 14.687 s | 첫 post-start PWM 확인 | orchestrator ACTIVE |
| 24.407 s | waypoint idx 0→1 | takeoff 도달 |
| 42.407 s | waypoint idx 1→2 | outbound P2 도달, return 시작 |
| 73.651 s | `/_ros2cli_requester_std_srvs_Trigger` | 명시적 stop 요청 |
| 73.657 s | STOPPING | 자동 fault 아님 |
| 73.683 s | `control_active=false` | active 종료 |
| 73.765 s | servo 8채널 neutral 확인 | stop 뒤 약 82 ms |

`/rosout` 322개는 INFO 281/WARN 41/ERROR 0/FATAL 0이고, WARN은 모두 marker lost다. 기록된 `/rosout`과 status에서는 control fault, watchdog timeout 또는 EKF stale이 관측되지 않았다. 배열 검사에서도 NaN/Inf와 integrator clamp 도달이 없었다. 이는 기록된 topic/log-level 범위의 결론이며 내부에서 log되지 않은 상태까지 부재했다고 증명하지는 않는다. 종료 상태는 `STOPPING→IDLE`이다. 후속 logging에서는 폴더명 대신 다음 enum을 summary에 기록해야 한다.

```text
SUCCESS
OPERATOR_ABORT
SAFETY_FAULT
TRACKING_FAILURE
INFRASTRUCTURE_FAILURE
```

## 5. waypoint와 경로 추종

controller가 사용한 aligned odometry 기준 결과다.

| leg | active 시간 | 결과 | 목표 최소거리 | 종료거리 |
|---|---:|---|---:|---:|
| P0→P1 | 9.76 s | 도달 | 전환 observation 약 0.044 m¹ | 전환 |
| P1→P2 | 18.00 s | 도달 | 약 0.147 m | 전환 |
| P2→P1 | 31.28 s | 사용자 stop 전 미도달; 방향전환은 사용자 확인 | 0.495 m | 1.063 m |

¹ 비동기 nearest aligned-odometry sample에서는 약 0.052 m다. takeoff의 실제 reach rule은 `min(0.15,0.05) m`이고 observation update에서 전환됐다.

return leg는 시작 후 약 6.7 s 동안 목표에서 반대로 멀어져 거리가 최대 2.47 m까지 증가했다. 이후 최소 약 0.50 m까지 접근했지만 다시 멀어졌다. 1 s smoothing을 적용해도 최소거리가 약 0.53 m이므로 단일 position spike 때문에 reach 판정만 놓친 사례가 아니다.

return 구간에서 aligned odometry가 누적한 path length는 약 10.36 m인데 순변위는 약 0.50 m뿐이다. 단순히 추진력이 부족해 천천히 전진한 것이 아니라 큰 우회·진동과 estimator correction이 섞인 궤적이다.

| 지표 | outbound P1→P2 | return P2→P1 |
|---|---:|---:|
| lateral Y RMS | 0.162 m | 0.201 m |
| lateral Y max | 0.332 m | 0.526 m |
| depth error RMS | 0.065 m | 0.135 m |
| depth error max | 0.202 m | 0.344 m |
| 평균 순경로 진행 | 약 0.080 m/s | 약 0.012 m/s |

첫 outbound 도달만으로 안정적인 LOS라고 판정해서는 안 된다. 최소 한 번의 완전한 P1→P2→P1 cycle과 방향별 지표를 별도로 통과해야 한다.

## 6. 속도 추종과 state consistency

`depth_speed_limit=0.05 m/s`가 수평 cruise와 독립 적용되므로 실제 목표 벡터 norm은 최대 다음 값까지 증가했다.

```text
sqrt(0.10^2 + 0.05^2) = 0.1118 m/s
```

active 전체에서 목표속력 평균은 약 0.100 m/s지만 EKF-reported body 속력 평균은 0.151 m/s, 최대는 0.463 m/s였다. velocity vector error RMS는 0.173 m/s, p95는 0.314 m/s다. 이 값은 독립 ground truth 속력이 아니라 policy가 받은 state다.

| leg | 목표 norm 평균/최대 | EKF-reported norm 평균/최대 | vector error RMS | 목표 반대방향 비율 |
|---|---:|---:|---:|---:|
| P0→P1 | 0.073 / 0.107 | 0.106 / 0.240 | 0.130 m/s | 40.3% |
| P1→P2 | 0.106 / 0.112 | 0.146 / 0.321 | 0.154 m/s | 23.6% |
| P2→P1 | 0.105 / 0.112 | 0.167 / 0.463 | 0.194 m/s | 32.7% |

더 중요한 문제는 동일 `LOCAL_POSITION_NED` message에서 나온 position과 velocity가 장기적으로 맞지 않는다는 점이다.

| leg | `||Delta position - integral velocity dt||` |
|---|---:|
| P0→P1 | 0.052 m |
| P1→P2 | 0.721 m |
| P2→P1 | 1.194 m |
| 전체 active | 약 1.58 m |

position sample의 최대 한-step 이동은 outbound 약 0.051 m, return 약 0.104 m였다. 25 Hz 환산 순간속도는 각각 약 1.26 m/s와 2.58 m/s지만 debug velocity 최대 norm은 0.463 m/s 수준이다. 현재 discontinuity gate는 한 sample translation 0.50 m만 검사하므로 이러한 작은 correction의 누적을 검출하지 못한다.

이는 frame 변환 이후에 새로 생긴 오차가 아니다. [`mavlink_interface.py`](../../brov_ros2-main/brov_base/brov_base/mavlink_interface.py)의 같은 `LOCAL_POSITION_NED` 수신 message에서 저장한 `x/y/z`와 `vx/vy/vz`끼리 이미 불일치한다. 다음 중 무엇이 원인인지는 추가 계측이 필요하다.

- DVL velocity injection의 body/world frame, scale 또는 timestamp
- EKF position correction과 velocity source의 불일치
- `LOCAL_POSITION_NED` source epoch 또는 reset
- MAVLink source timestamp와 수신시각 정렬
- tether/current에 의한 실제 motion과 estimator correction의 결합

### 필수 보완

1. ground truth가 있는 SITL에서 `LOCAL_POSITION_NED` position·velocity를 각각 truth와 비교한다.
2. 1 s, 5 s, 10 s sliding window로 `Delta p - integral(v dt)`를 발행한다.
3. reset·known correction이 없는 window에서 source timestamp와 EKF correction/reset term을 함께 기록한다. 10 s residual 0.10 m는 잠정 gate로만 사용하고, G2 truth 및 tethered/static·restrained-motion 실험으로 threshold를 교정한다.
4. DVL raw velocity, injected MAVLink message, EKF innovation/reset counter와 source timestamp를 bag에 기록한다.
5. position/velocity consistency gate failure를 policy domain randomization으로 보상시키지 말고 제어 gate에서 제외한다.

## 7. one-shot localization과 ArUco 재관측 불일치

현재 localization은 초기 20개 stationary pair로 `pool_T_odom`을 만든 뒤 고정한다. marker loss나 재관측은 alignment를 갱신하거나 invalid로 만들지 않는다. 이 의도는 [`brov_localization/README.md`](../../brov_ros2-main/brov_localization/README.md)에 명시돼 있다.

상태는 bag 시작 뒤 0.005 s `UNINITIALIZED`, 4.696 s `COLLECTING`, 7.360 s `INITIALIZED`로 변했고 이후 `INVALID`는 0회였다. aligned odometry 1,902개는 모두 같은 epoch/session/alignment를 유지했으며 약 24.73 Hz, 최대 gap 약 79.7 ms였다.

bag에는 ArUco pose 179개가 기록됐다. `/rosout`에는 marker loss 41회와 reacquisition 40회가 있고, 마지막 loss 뒤에는 active 종료까지 약 35.82 s 동안 복구되지 않았다. ArUco pose의 recorder interval은 median 약 124 ms지만 p95 약 682 ms, max 약 2.09 s였다. pose header age도 p95 약 705 ms이고 1 s를 넘은 sample이 7개라서 receive timestamp만으로 moving fusion을 해서는 안 된다. camera status는 16.0–16.2 fps와 RTP lost/late/duplicate 0을 기록했으므로 단순 RTP packet loss보다 가림·시야 이탈·영상 품질·검출 실패 정황이 강하지만 영상 없이 직접 원인을 확정할 수는 없다.

마지막 marker loss는 return idx 2 시작 약 4.54 s 전이다. 그 뒤에도 localization은 invalid로 전환되지 않았다. 현재 `valid=true`는 고정 alignment identity와 fresh local odometry를 뜻할 뿐, pool-frame absolute error가 작다는 뜻은 아니다.

엄격히 초기화 이후인 ArUco source stamp에 aligned odometry를 위치 선형보간·자세 slerp한 120 pair에서 다음 차이가 관측됐다.

| aligned odometry - ArUco | 값 |
|---|---:|
| position residual norm mean | 0.340 m |
| position residual norm median | 0.216 m |
| position residual norm p95 | 0.904 m |
| position residual norm max | 1.486 m |
| orientation residual median | 4.24° |
| orientation residual p95 | 26.78° |
| orientation residual max | 48.99° |

마지막 vision sample의 position residual은 약 `(-0.280,+0.249,-0.010) m`, norm 0.375 m였고 자세 차이는 9.13°였다. 최대 position residual은 marker flicker 구간에 발생했으므로 전체 차이를 순수 DVL drift로 간주할 수 없다.

이 비교만으로 aligned odometry와 ArUco 중 어느 쪽이 실제 truth인지 확정할 수 없다. camera pose에 capture timestamp, covariance와 품질 지표가 없고 marker detection 자체가 간헐적이기 때문이다. 하지만 **초기 20개 정렬 pair를 공유한 뒤 갱신이 분리된 camera pose와 local-odometry 경로가 크게 달라도 localization status가 계속 valid였다는 사실**은 확정할 수 있다.

### 필수 보완

- capture/source timestamp를 end-to-end로 검증하기 전에는 ArUco를 무조건 continuous fusion하지 않는다.
- moving 상태에서도 quality가 검증된 pose만 사용해 `vision - aligned odometry` residual monitor를 구현한다.
- translation 0.15 m, rotation 10°라는 기존 initialization gate를 online diagnostic의 출발점으로 사용하고, 지속 위반 시 warn/freeze/stop 정책을 명시한다.
- `/brov/aruco/visible`, marker id, reprojection error, pose covariance, detection latency와 reject reason을 bag에 기록한다.
- marker dropout 시간과 false pose/outlier를 Gazebo full-stack localization 시험에 재현한다. 이는 low-level policy observation DR와 분리한다.
- re-alignment는 정지·중립·명시적 operator 승인 뒤 새 epoch/mission으로만 수행한다.

## 8. 후순위 관측: waypoint 명령 불연속과 자세 결합

waypoint 전환에서 command가 한 sample에 바뀌었다. 이는 bag의 정량 사실이지만, 사용자가 방향전환 성공을 확인했으므로 현재 선속도 문제의 1차 원인으로 간주하지 않는다.

| 전환 | `v_d` vector jump | `q_d` geodesic jump | 전환 직후 기록 변화 |
|---|---:|---:|---|
| idx 0→1 | 0.126 m/s | 28.2° | pitch action `+0.068→-1.0`, T8 큰 reverse jump |
| idx 1→2 | 0.173 m/s | 109.7° | `Delta a_pitch≈+1.50`, yaw action -1, T7 PWM `Delta≈+1.375` |

두 번째 전환 직후 자세오차는 114.4°까지 증가했다. return leg의 자세오차 p95는 약 33.4°이고 최대 114.4°다.

`takeoff_then_align`은 3D desired velocity 방향으로 yaw와 pitch를 함께 맞춘다. 수평 leg에서도 독립 depth hold가 `v_d.z`를 만들기 때문에 목표 pitch가 약 -29°에서 +27°까지 변했다. 완전구동 ROV는 heave로 수심을 제어할 수 있으므로, level/yaw-only attitude와 3D align을 별도 시험해야 한다.

### 후속 command shaper 후보

1. total desired-speed norm을 최종 단계에서 profile 속력 이하로 제한한다.
2. 시작과 waypoint 접근에 acceleration/deceleration limit을 둔다.
3. braking distance와 estimator/actuator delay로 감속 시작점을 정한다.
4. reach는 거리만 보지 않고 낮은 actual speed와 짧은 dwell을 요구한다.
5. return은 `감속→정지→shortest-path q slew→역방향 ramp` 순서로 수행한다.
6. `q_d`에 명시적인 yaw/pitch rate limit을 두고 한 sample quaternion step을 없앤다.
7. Case A 수평 leg에서는 `level+yaw-only`, `yaw-only align`, 기존 `3D align`을 A/B 시험한다.
8. waypoint 전환 때 integral 유지/reset/leaky decay를 별도 variant로 평가한다.

## 9. policy, allocation과 actuator 경로

### Policy/action 포화

| action 축 | exact ±1 비율 | sample-hold 기준 가장 긴 exact ±1 연속시간 |
|---|---:|---:|
| surge | 1.37% | 약 0.76 s |
| sway | 14.96% | 약 4.32 s |
| heave | 0% | 없음 |
| roll | 0.62% | 약 0.32 s |
| pitch | 12.90% | 약 0.64 s |
| yaw | 2.06% | 약 1.20 s |

active sample의 `382/1457=26.22%`에서 최소 한 축이 exact ±1이었다. `|action|>=0.99` 기준은 27.45%다. 적분항은 ±5 clamp에 도달하지 않았고 observation에 NaN/Inf도 없었다. 즉 단순 observation shape/NaN fault가 아니라 command/state 변화에 대한 큰 policy response다.

### Action→T200

기록 action을 현재 allocation과 T200 inverse에 다시 넣으면 preview와 최대 약 `7.1e-6`, RMS `3.4e-7` 이내로 일치한다.

```text
limited action
× [85,85,120,26,14,22]
→ allocation B+
→ requested thruster force
→ T200 inverse
→ PWM preview
```

하지만 requested force 기준 active sample의 `163/1457=11.19%`에서 최소 한 thruster가 현재 code/config의 T200 model clamp `-51.5/+64.1 N` 범위를 넘었다. 이는 실측 thrust 한계가 아니라 offline 명령 재구성 결과다. PWM plateau가 reverse 약 -0.9915, forward 약 +0.9710이므로 `|PWM|>=0.99`만으로 포화를 세면 forward clamp를 놓친다.

### Preview→sent→servo

- 첫 active preview 한 개는 activation edge 안전규칙에 따라 discard됐다.
- 나머지 1,456 preview와 sent PWM 값의 최대 차이는 0이다.
- logical PWM과 reversal mask `[+,-,-,+,+,+,+,-]`로 계산한 servo vector가 active servo sample의 99.79%에서 과거 sent command와 exact microsecond 단위로 일치한다.
- command→`SERVO_OUTPUT_RAW` recorder-time apparent echo lag은 median 약 76 ms, p95 약 116 ms다. source timestamp가 없고 반복 command의 causal pairing이 일부 모호하므로 motor time constant로 해석하지 않는다.

따라서 기록된 configured logical command 경로에서는 수치·채널 불일치가 발견되지 않았다. 다만 실제 ESC 회전, thrust 방향·크기와 기계적 응답은 단일-thruster impulse나 RPM/current/force 계측으로 별도 검증해야 한다.

`control_active=false` 이후에도 policy node는 action과 preview를 계산했다. post-stop action/preview는 각 264개였고 그중 `258/264=97.73%`가 `|action|>=0.99`였지만, sent PWM은 0개였다. 따라서 actuator 포화 지표의 분모는 반드시 `control_active && output_enabled`로 한정하고 shadow preview와 실제 출력을 분리한다.

### 누락된 계측

다음 bag부터 아래 값을 별도 topic과 source timestamp로 기록한다.

```text
raw_unclamped_nn_action
policy_clamped_action
operationally_limited_action
desired_wrench_policy_frame
desired_wrench_SNAME
thruster_force_requested
thruster_force_clamped
allocation_residual_pre_clamp, allocation_residual_post_clamp
pwm_inverse_raw, pwm_shaped_preview, pwm_sent, servo_output
control_active, output_enabled, output_gate_reason
activation_generation, command_sequence
resolved_action_limits, resolved_pwm_limits, reversal_mask
policy_hash, vehicle_hash, config_hash
servo_mavlink_source_timestamp
model_predicted_wrench
ESC RPM/current/voltage 또는 Gazebo measured-force proxy
saturation/slew/deadband flags
```

## 10. timing

active observation rate는 약 24.68 Hz다.

| interval | 값 |
|---|---:|
| median | 40.01 ms |
| p95 | 44.52 ms |
| p99 | 72.58 ms |
| max | 79.76 ms |
| 60 ms 초과 | 18회 |
| 100 ms 초과 | 0회 |

평균은 25 Hz 계약과 맞지만 약 1%의 sample에서 70–80 ms 지터가 있다. integrator는 실측 `dt`를 사용하지만 frozen policy 자체에는 temporal state가 없다.

sim2sim에서는 다음을 분리해야 한다.

- policy/observation scheduling jitter: 위 empirical distribution
- MAVLink/recorder servo echo 지연: 실제 actuator lag로 사용하지 않음
- T200 motor lag: 별도 nominal 0.05 s와 DR
- sensor source timestamp, sample-and-hold와 receive delay

## 11. sim2sim과 sim2swim 미구현 항목

### 시간 제약을 반영한 MUST

1. training/runtime 16-D의 quaternion hemisphere, 적분 clamp/reset/stale-freeze를 동일화한다.
2. `0/0.05/0.10/0.50/0.58–0.63 m/s`, hold·stop·restart·180° reversal과 episode 중 command 변경을 구현한다.
3. velocity-priority reward, overspeed/cross-axis/saturation metric과 최소 15–30 s episode/horizon을 구현한다.
4. action frame/sign을 legacy 보존 또는 explicit `T6` 중 하나로 고정하고 Isaac/runtime golden vector를 만든다. runtime만 단독 수정하지 않는다.
5. Isaac 저속·hold·reversal 및 exact Gazebo Case A의 JSON/CSV pass/fail과 checkpoint/policy/config lineage를 구현한다.
6. 현 frozen policy 0.10/0.60은 baseline으로 한 번씩 평가한 뒤, 위 보완을 적용한 정책을 재학습한다.
7. policy는 position을 직접 보지 않으므로 새 position learner/reward는 만들지 않는다. Gazebo에서는 GT와 EKF를 함께 기록해 estimator 문제만 별도로 분리한다.

### 후속 분석 로드맵

아래 P0/P1/P2는 완전한 시스템 승인을 위한 backlog이며 이번 데모 복구의 모든 선행조건이 아니다.

### P0 — 후속 시스템 강화

1. **estimator consistency harness**
   - Gazebo truth, injected DVL, EKF position/velocity와 ROS observation을 동시에 비교
   - `Delta p - integral(v dt)` sliding metric과 fail gate
   - reset/source epoch/innovation/source timestamp 기록
2. **Case A exact regression scenario**
   - 0.10/0.40/0.15, P0→P1→P2→P1, `takeoff_then_align`
   - outbound와 return을 별도 판정
   - Isaac 0.5 parity profile과 파일·launch·결과 완전 분리
3. **command shaper**
   - speed norm cap, accel/decel, braking, stop/dwell, quaternion slew
   - level/yaw-only/3D-align A/B profile
4. **runtime tracking-failure monitor**
   - target-distance progress, overspeed, cross-track/depth, attitude error
   - action/force saturation fraction과 연속 체류
   - estimator residual과 online localization residual
5. **truth/estimate dual logging**
   - G0/G1/G2에서 같은 schema로 truth와 estimated state를 저장
6. **artifact/config provenance**
   - resolved mission, installed config SHA, policy/vehicle/safety SHA와 commit

### P1 — 후속 학습 확장

- raw/limited action 및 requested/achieved wrench를 reward와 trace에 포함
- runtime과 동일한 command shaper를 training environment에 이식
- 25 Hz empirical jitter, T200 lag/deadband/비대칭과 sensor sample-and-hold
- 60–120 s multi-waypoint evaluator
- Case A/C 저속과 Isaac/Gazebo 0.5를 한 approval matrix에서 각각 판정
- 여러 seed, calibrated plant와 held-out OOD plant 결과 저장

### P2 — 최종 sim2swim 승인 전 강화

- ArUco quality/capture timestamp와 online residual monitor
- DVL raw/injection/EKF innovation logging
- RPM/current 또는 독립 thrust proxy
- restrained single-thruster 및 6축 acceleration sign 시험
- tracking failure의 자동 neutral/disarm 정책과 operator runbook

## 12. 후속 full-roadmap 및 최종 후보 요구사항

### 최종 sim2real 후보 조건

아래는 첫 smoke retrain의 blocker가 아니다. 다만 이 bag의 position/velocity 불일치를 그대로 DR로 복사하거나 최종 sim2real policy를 승인하기 전에는 해결한다.

1. Gazebo ground truth에서 estimator frame·scale·timestamp 검증
2. 실기 10 s position/velocity consistency threshold 확정
3. command shaper와 heading contract 확정
4. exact Case A를 model-based controller가 통과
5. T1~T8 physical sign과 calibrated actuator/plant 검증

### Observation/action

- 현 최소 반복은 fixed 16-D parity를 먼저 확인한다. `v_desired_body_zup(3)`를 포함한 19-D는 현재 미구현인 구조적 개선 후보이며, 16-D의 zero-state bias가 남으면 최종 후보로 승격한다.
- previous action 6-D 또는 recurrent policy는 actuator memory ablation으로 분리한다.
- state age/validity와 estimator inconsistency는 policy가 보상할 입력이 아니라 safety gate로 사용한다.
- action 계약에서 explicit FLU/Z-up→SNAME `T6`를 선택하면 학습과 runtime 양쪽을 함께 변경해 새 policy를 학습한다. legacy 의미를 보존하는 경우에도 6축 golden vector로 고정한다.

### Curriculum

```text
hold
→ 0.05 m/s
→ Case A 0.10 m/s takeoff/outbound/return
→ 0.25 m/s long straight
→ Isaac parity 0.50 m/s straight/out-and-back
→ mixed 0↔0.05↔0.10↔0.50 m/s
```

- 5 s fixed command만 사용하지 말고 최소 15–30 s multi-command 학습 episode를 사용한다. 60–120 s와 full return은 validation에 포함한다.
- speed ramp, braking, stop/dwell, reversal과 rate-limited attitude target을 training/runtime에서 동일하게 사용한다.
- 109° one-step attitude 명령을 학습으로 견디게 하지 않고 command contract에서 제거한다.
- level/yaw-only Case A를 기본 후보로 두고 3D-align은 별도 stress/curriculum으로 평가한다.

### Reward와 validation

- tolerance-normalized velocity vector error와 geodesic attitude error
- cross-axis motion 및 `||v|| > ||v_d||+margin` overspeed penalty
- delta action/PWM, action·force saturation과 allocation residual penalty
- long-horizon integral growth와 waypoint transition excursion
- cross-track/depth는 LOS full-stack validation metric으로 사용하고, low-level policy reward에 넣으려면 해당 state를 observation에도 제공해 POMDP를 만들지 않는다.

## 13. bag 기반 provisional 회귀 gate

아래 값은 새 확정 요구사항이 아니라 현재 실패를 다시 통과시키지 않기 위한 초기 gate다.

| 분류 | provisional gate |
|---|---:|
| Case A return | P2→P1 0.15 m reach와 low-speed dwell 도달 |
| 반복성 | P1→P2→P1 5회 연속 성공 |
| 10 s estimator consistency | reset·known-correction 없는 source-time window에서 `||Delta p-integral(v dt)|| <= 0.10 m`; G2 truth/real 교정 전 provisional |
| horizontal lateral RMS/max | 0.15 / 0.30 m 이하 |
| depth RMS/max | 수평 P1↔P2 leg의 waypoint 전환·command-slew 제외 구간에서 0.10 / 0.20 m 이하 |
| EKF-reported overspeed at 0.10 profile | `p95(max(0,||v||-||v_d||)) <= 0.05 m/s`; 별도로 `max(||v||) <= 0.20 m/s` |
| q target step | `2*acos(clamp(abs(dot(q_d[k],q_d[k-1])),0,1)) <= omega_slew*Delta t+tolerance` |
| action cap | `control_active && output_enabled` 정상상태에서 1% 미만, 전체 profile 5% 미만, 0.2 s 초과 연속 cap 0회 |
| T200 force clamp | 같은 window에서 정상상태 1% 미만, 전체 profile 5% 미만, 0.2 s 초과 연속 clamp 0회 |
| active rate | 24–26 Hz, gap 0.25 s 초과 0회 |
| software outcome | `TRACKING_FAILURE`가 success로 분류되는 경우 0회 |
| tracking-fault response | `control_active=false→sent 중단→0.25 s 이내 neutral echo→disarm request→TRACKING_FAILURE`; hardware ACK는 별도 계측 |

online ArUco residual gate는 capture timestamp와 quality가 검증된 뒤 기존 initialization threshold인 translation 0.15 m/rotation 10°부터 조정한다.
여기서 정상상태는 명시적 accel/brake, quaternion slew와 waypoint 전환 window를 제외한 구간이다.

## 14. 원인 판정의 경계

### 이 bag으로 확인됨

- 실행 contract는 0.10 m/s였다.
- 사용자가 선속도·위치 추종 불량을 보고 수동 stop했으며, 그 전까지 return waypoint은 미도달이었다.
- 사용자 관측상 P2의 방향전환은 수행됐다. 따라서 return 미도달을 회전 실패로 분류하지 않는다.
- position과 velocity는 장기 적분 일관성을 잃었다.
- waypoint command step과 policy/thruster saturation이 동시에 발생했다.
- preview→sent 값과 sent→configured servo vector에서 불일치가 발견되지 않았다. 실제 thrust는 검증되지 않았다.
- software fault가 아니라 명시적 ROS CLI Trigger stop으로 끝났다.

### 가능성이 높지만 추가 시험 필요

- Isaac 0.58–0.63 m/s 학습과 deploy/Gazebo·실기 0.10 m/s의 distribution mismatch, 정책 bias와 velocity reward 비중이 선속도 포화·표류에 기여했다.
- 3D align과 depth hold 결합이 불필요한 pitch motion을 만들었다.
- one-shot 정렬 이후 local/DVL/EKF odometry drift 또는 vision 품질 문제가 pool localization 불일치에 기여했다.

### 이 bag으로 단정할 수 없음

- 실제 T1~T8 물리 thrust 방향과 크기
- DVL frame/scale와 EKF 중 어느 부분이 position/velocity 불일치의 직접 원인인지
- ArUco와 aligned odometry 중 어느 쪽이 실제 위치에 더 가까운지
- hydrodynamic coefficient, tether/current와 motor degradation의 기여도
- 재학습만으로 문제가 해결되는지

## 15. 권장 다음 실행 순서

1. frozen policy Isaac `straight 0.10/0.60`을 실행해 변경 전 baseline을 남긴다.
2. observation/action 계약, 저속·hold·stop·reversal command, velocity-priority reward, 15–30 s episode/horizon과 evaluator/manifest를 구현한다.
3. 수정된 계약으로 smoke retrain 후 Isaac 0.10/0.60, hold·stop·reversal에서 checkpoint를 선택한다.
4. Gazebo surge pulse로 부호를 확인하고 GT/EKF를 함께 기록한 exact Case A를 실행한다.
5. Gazebo Case A에서 `P2 도달/방향전환/P1 return`, 0.10 m/s 속도, cross-track과 action cap을 3회 연속 통과한다.
6. 같은 artifact/config으로 실기 Case A 0.10 m/s를 재시도한다. 실기 0.5 m/s는 별도 safety review 없이는 수행하지 않는다.

## 주요 구현 근거

- [`observation.py`](../../brov_ros2-main/brov_base/brov_base/observation.py): 16-D observation과 frame/dt/integral
- [`guidance.py`](../../brov_ros2-main/brov_base/brov_base/guidance.py): LOS, depth hold와 `takeoff_then_align`
- [`mavlink_interface.py`](../../brov_ros2-main/brov_base/brov_base/mavlink_interface.py): `LOCAL_POSITION_NED`, reversal과 PWM 송신
- [`policy_node.py`](../../brov_ros2-main/brov_control/brov_control/policy_node.py): policy, allocation, T200 inverse와 preview/live gate
- [`localization_node.py`](../../brov_ros2-main/brov_localization/brov_localization/localization_node.py): one-shot pool alignment
- [`localization.yaml`](../../brov_ros2-main/brov_localization/config/localization.yaml): timestamp/stationary/residual gate
- [`SIM2SWIM_DEMO.md`](../../brov_ros2-main/docs/SIM2SWIM_DEMO.md): Case A 운영 contract
