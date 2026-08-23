# Stage 2 — no-GPS DVL/INS/AHRS gap 수정 및 frozen-policy 회귀

작성일: 2026-08-17

## 결론

Stage 2의 센서 경로 수정과 원인 분리는 완료했다.

1. 기존 Edo full SITL의 `LOCAL_POSITION_NED`는 실기와 달리 **5 Hz synthetic GPS**를 EKF3가 융합한 결과였다. 실기 로봇에는 GPS가 없으므로 이 경로의 rate/delay를 튜닝하는 작업은 중단했다.
2. Gazebo body velocity를 `15 Hz / 100 ms / 0.003 m/s` DVL로 모델링하고, Water Linked 계열과 같은 `VISION_POSITION_DELTA`로 ArduSub EKF3 ExternalNav에 주입하는 no-GPS 경로를 구현했다. 정책은 계속 `ATTITUDE_QUATERNION + LOCAL_POSITION_NED`만 사용한다.
3. 정책을 뺀 축 펄스에서 DVL-only EKF의 instantaneous velocity는 정상이다. A1의 EKF↔GT vector RMSE는 `0.0111 m/s`, XY frame-fit determinant는 `0.9930`이고 모든 축 부호가 맞았다.
4. 그러나 XKF position과 XKF velocity의 장기 closure에는 작은 지속 bias가 남았다. injected VPD 자체의 36 s 수평 누적 오차는 약 `0.009–0.010 m`인데 XKF 수평 closure는 `0.073–0.077 m`다. 주된 잔차는 DVL frame/delta 생성 뒤의 EKF fusion에서 생긴다.
5. 이 DVL-only 체인에서 frozen policy를 0.5 m/s로 재실행하자 5 m 미션은 완료했고 실제 목표방향 속도는 `0.469 m/s`로 회복됐다. 반면 실제 vector error는 `0.113 m/s`, cross-speed RMS는 `0.102 m/s`, 한 축 이상 action bound 점유는 `67.9%`였다. 따라서 **선속도 과소응답의 일부는 잘못된 simulator estimator였지만, 사용자가 관측한 jitter의 잔여 주원인은 policy/action/plant 쪽에 있다.**

이 결과로 Stage 3 재학습의 인과가 정리됐다. no-GPS DVL runtime 경로를 기준으로 삼되, 재학습에서는 observation/action 계약, 논문의 command 생성, velocity-priority reward, 긴 horizon과 saturation 억제를 반드시 구현한다. 임의로 GPS나 EKF gain을 더 튜닝해 policy 문제를 가리면 안 된다.

## 구현한 경로

```text
Gazebo Odometry (world ENU / body FLU)
  -> 15 Hz DVL emulator (body FRD, bottom-lock, delay/noise/FOM)
  -> MAVLink VISION_POSITION_DELTA
  -> ArduSub EKF3 ExternalNav + IMU/AHRS + pressure/barometer
  -> ATTITUDE_QUATERNION + LOCAL_POSITION_NED
  -> brov_ros2 16-D observation / LOS / frozen policy
```

GPS는 센서 생성과 EKF source에서 모두 제외했다.

```text
GPS1_TYPE=0
GPS2_TYPE=0
SIM_GPS_DISABLE=1
EK3_SRC1_POSXY=6
EK3_SRC1_VELXY=6
EK3_SRC1_POSZ=1
EK3_SRC1_VELZ=0
VISO_TYPE=1
```

`SET_GPS_GLOBAL_ORIGIN`은 GPS 측정/융합이 아니라 no-GPS local NED 좌표계의 원점을 한 번 정의하기 위해서만 사용했다. DataFlash의 `GPS/GPA` message count는 A0/A1 모두 0이다.

구현·재현 파일:

- [`stage2_sitl_dvl_injector.py`](stage2_sitl_dvl_injector.py): DVL sampling, frame 변환, delay/noise/FOM, VPD 주입 및 raw diagnostics
- [`stage2_dvl_only.parm`](stage2_dvl_only.parm): realistic provisional no-GPS profile
- [`stage2_dvl_oracle.parm`](stage2_dvl_oracle.parm): delay/noise를 제거한 diagnostic upper bound
- [`stage2_set_ekf_origin.py`](stage2_set_ekf_origin.py): deterministic no-GPS NED origin
- [`stage2_sitl_axis_pulse.py`](stage2_sitl_axis_pulse.py): policy-free symmetric axis pulse
- [`run_stage2_dvl_only_pulse.sh`](run_stage2_dvl_only_pulse.sh): fresh A0/A1 runner
- [`run_stage2_dvl_only_policy_0p5.sh`](run_stage2_dvl_only_policy_0p5.sh): fresh frozen-policy 0.5 m/s runner
- [`analyze_stage2_pulse.py`](analyze_stage2_pulse.py): rosbag/DataFlash/delta/closure analysis

DVL frame·confidence 단위시험은 10/10 통과했다. 모든 live run은 SITL identity thruster reversal, policy SHA-256 `0d89f3270f46214f1569b7d48dcb5e25363b1d9b7353b82ced0fc67c0093a472`, 25 Hz controller와 fresh ArduSub state를 사용했다.

## A1 realistic 대 A0 oracle

A1은 현재 Isaac YAML의 명목값인 `15 Hz / 100 ms / noise 0.003 m/s`를 사용한다. 이 값은 아직 실기에서 측정한 값이 아니므로 **provisional profile**이다. A0는 같은 VPD/EKF topology에서 delay/noise/FOM만 제거한 upper-bound 진단이며 실기 재현 결과가 아니다.

| 정책 없는 36 s 펄스 | A1 realistic | A0 oracle |
|---|---:|---:|
| GPS/GPA message | 0 / 0 | 0 / 0 |
| VPD source rate | 14.997 Hz | 15.006 Hz |
| realized delay / noise std | 0.100 s / 약 0.003 m/s | 0 / 0 |
| ROS EKF↔GT velocity RMSE | 0.01106 m/s | 0.00961 m/s |
| DataFlash XKF1↔SIM2 velocity RMSE | 0.00982 m/s | 0.00852 m/s |
| XY fit determinant / singular values | 0.9930 / 0.9992, 0.9938 | 0.9989 / 1.0022, 0.9968 |
| injected VPD delta 수평 누적오차 | 0.00928 m | 0.01020 m |
| DataFlash XKF full 수평 closure | 0.07737 m | 0.07336 m |
| DataFlash XKF 10 s 수평 closure p95 | 0.05004 m | 0.04441 m |
| DataFlash XKF full 수직 closure | 0.24782 m | 0.27663 m |

지연·잡음을 모두 제거해도 full horizontal closure는 5.2%만 줄었다. 반면 DVL 입력 delta의 누적오차는 약 1 cm에 불과하다. 따라서 100 ms delay, 0.003 m/s noise, FLU↔FRD sign 또는 VPD delta 생성은 주원인이 아니다. 작은 XKF velocity bias와 position-side correction이 누적된다.

수직 결과는 `POSZ=Baro`와 body-odom XYZ velocity가 함께 융합되는 mixed-source 문제를 포함한다. A0/A1 fresh boot의 barometer ground pressure와 초기 수심도 달랐으므로 두 수직 수치의 차이를 noise/delay의 인과 효과로 해석하지 않는다. 다만 두 run 모두 큰 수직 closure가 남았다는 사실은 실제 DVL/Bar30 parameter와 raw log가 필요하다는 근거다.

## Frozen policy 0.5 m/s 회귀

동일 5 m 단방향 mission과 동일 frozen actor를 사용했다. 아래 세 run은 모두 첫 수평 1 s와 terminal hold를 제외한 physical Gazebo GT 지표다. `GT feedback`과 `stock GPS-EKF`는 Stage 1, `DVL-EKF`는 Stage 2 fresh run이다.

| 지표 | GT feedback | stock GPS-EKF | no-GPS DVL-EKF |
|---|---:|---:|---:|
| takeoff+5 m 완료 | 13.480 s | 13.641 s | **11.603 s** |
| 실제 목표방향 속도 | 0.468 m/s | 0.436 m/s | **0.469 m/s** |
| 실제 vector velocity RMSE | **0.0588 m/s** | 0.1089 m/s | **0.1129 m/s** |
| 실제 cross-speed RMS | **0.0404 m/s** | 0.0811 m/s | **0.1024 m/s** |
| 한 축 이상 `abs(action)>=0.99` | 58.2% | 64.1% | **67.9%** |
| requested thruster force clamp | 5.94% | 12.12% | **16.51%** |

DVL-EKF run의 추가 결과:

- controller-visible `v_parallel=0.472 m/s`, physical `v_parallel=0.469 m/s`: stock GPS run의 과대관측(`0.478` 대 physical `0.436`)은 제거됐다.
- horizontal steady EKF↔GT velocity RMSE는 `0.0588 m/s`; 15 ms constant-lag 보정 뒤에도 `0.0422 m/s`다. 빠른 closed-loop 운동에서 pulse보다 estimator error가 커진다.
- cross-track RMS/max는 `0.048/0.082 m`, attitude error p95는 `8.86°`로 경로와 자세는 대체로 유지됐다.
- depth error RMS/max는 `0.129/0.151 m`로 RMS 0.10 m gate를 넘었다.
- action bound는 pitch `52.8%`, yaw `16.5%`, roll `9.2%`이며, surge는 bound에 닿지 않았다. jitter를 단순 surge authority 부족으로 설명할 수 없다.
- topic contract 누락 0, desired/action rate 약 `24.81 Hz`, max gap 약 `80 ms`, mission 완료 후 stop/disarm/neutral을 확인했다.

따라서 Stage 2의 판정은 다음과 같다.

| 항목 | 판정 |
|---|---|
| stock Gazebo GPS 경로의 실기 동형성 | FAIL — 실기에 없는 sensor/fusion |
| no-GPS DVL frame/sign/scale와 instantaneous velocity | PASS |
| DVL VPD input delta 자체 | PASS 수준 — 36 s 수평 누적 약 1 cm |
| EKF position–velocity long-horizon consistency | BORDERLINE/FAIL — 10 s 약 5 cm, 36 s 약 7–8 cm; 수직은 더 큼 |
| frozen actor의 0.5 m/s along-speed | PASS |
| frozen actor의 vector/cross-axis/saturation | FAIL |
| 5 m mission mechanics 및 lifecycle | PASS |

## Stage 3에 넘기는 수정 목표

Stage 2에서 simulator estimator의 구조적 오류를 제거했지만 jitter가 남았다. 따라서 Stage 3에서는 다음을 필수로 한다.

1. 논문에서 time-varying attitude trajectory에 쓰인 곡선 항과 정확한 `||v_d^b||=0.5 m/s` velocity sampling을 분리해 구현한다. 현재 구현처럼 곡선 항을 velocity template으로 쓰지 않는다.
2. Isaac/runtime의 quaternion hemisphere, integral dt/clamp/reset, 16-D ordering을 하나의 observation contract와 golden vector로 고정한다.
3. action frame/sign과 allocation을 6축 basis test로 고정한다. runtime 한쪽만 바꾸지 않는다.
4. `0/0.10/0.50/0.58–0.63 m/s`, hold/stop/restart/reversal을 episode 안에 포함한다.
5. velocity/cross-axis error를 attitude/effort보다 우선하고 continuous saturation을 penalty와 checkpoint gate에 포함한다.
6. 15–30 s multi-command episode와 충분한 PPO horizon을 사용한다.
7. no-GPS DVL sample-and-hold/delay를 학습 observation에 넣되, 현재 A1 값을 실기 실측값이라고 부르지 않는다. 실제 BlueOS/DVL/ArduSub artifact를 얻으면 그 값으로 교체한다.

재학습 전 broad Gazebo mass tuning을 먼저 하지 않는다. real의 질량이 Isaac에 더 가까운데도 Gazebo와 같은 jitter가 관측됐고, no-GPS DVL 교체 뒤에도 cross-axis/saturation 문제가 남았기 때문이다. 소규모 mass/damping/thruster gain DR는 Stage 3 robustness 항목으로 넣되 jitter 원인의 대체 설명으로 사용하지 않는다.

## 산출물

- A1: [`stage2_results/stage2_dvl_only_axis_pulse_20260817_164511`](stage2_results/stage2_dvl_only_axis_pulse_20260817_164511)
- A0: [`stage2_results/stage2_dvl_only_oracle_20260817_165430`](stage2_results/stage2_dvl_only_oracle_20260817_165430)
- DVL frozen-policy: [`stage2_results/stage2_dvl_policy_0p5_20260817_170427`](stage2_results/stage2_dvl_policy_0p5_20260817_170427)

각 폴더에는 rosbag, ArduSub DataFlash `.BIN`, `mav.tlog`, final `.parm`, launch/service logs와 machine-readable JSON이 있다. 첫 arm 시도가 stale observer와 service name을 공유해 거부된 폴더 `stage2_dvl_policy_0p5_20260817_170228`은 유효 성능 run이 아니므로 최종 산출물에서 제외했다.

## 남은 실기 증거 한계

기존 sim2real rosbag에는 raw DVL packet, injected `VISION_POSITION_DELTA`, BlueOS extension 설정, ArduSub full params와 XKF/VISO innovation이 없다. 따라서 다음 실기 전 최소한 아래를 보존해야 A1을 실제 값으로 교정할 수 있다.

- DVL model/firmware와 BlueOS Water Linked extension version
- `orientation`, `should_send`, `rangefinder` 설정
- DVL mount xyz/rpy
- raw `velocity`, timestamp/dt, valid, FOM, altitude
- ArduSub `AHRS/EK3/VISO/RNGFND` params와 `.BIN`

이 증거가 없을 때 임의 EKF noise/delay gain을 더 튜닝하거나 이를 논문 재현 성능으로 주장하지 않는다.
