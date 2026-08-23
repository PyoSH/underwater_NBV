# Fresh Case-A GT vs Water Linked-aligned DVL-EKF A/B

실행일: 2026-08-17

## 결론

동일한 fresh Gazebo/ArduSub 초기조건에서 frozen policy로 `takeoff → 2 m outbound → 180° 반전 → 2 m return`을 각각 1회 실행했다. 두 run 모두 waypoint RLE `[0,1,2,1]`을 완주했고 control fault, DVL invalid packet, output watchdog fault는 없었다.

결론은 **GT feedback이 물리 추종을 크게 개선하지만 control-output jitter를 제거하지는 않으며, DVL-EKF/`LOCAL_POSITION_NED`가 특히 반전 이후 횡·심도 오차와 추진기 clamp를 추가로 증폭한다**는 것이다. 따라서 원인은 `estimator 하나`가 아니라 다음 두 층이다.

1. GT에도 남는 frozen actor의 pitch 중심 `-1↔+1` action 왕복과 높은 actor-bound 점유
2. Water Linked 계열 VPD→EKF3→`LOCAL_POSITION_NED` feedback이 더하는 속도·심도·cross-axis 오차

이 결과는 3단계 paper-aligned 재학습의 근거가 된다. DVL/EKF 개선만으로 현 jitter가 사라질 것이라고 기대해서는 안 되며, 반대로 재학습만으로 feedback 오차를 덮어서도 안 된다.

## 실험 계약

이 실험은 estimator 인과 분리를 위해 production pool-localization/resolved-mission layer를 제외하고, 동일한 direct-relative Case-A motion contract를 사용했다. `feedback_source=gazebo_truth`는 production resolved-mission과 fail-closed로 함께 쓸 수 없기 때문이다. 따라서 **전체 Case-A 운동 시퀀스 검증**이지만 camera/AprilTag/PREPARE orchestration까지 포함한 full production Sim2Swim 배포 검증은 아니다.

| 항목 | 고정값 |
|---|---|
| policy SHA-256 | `0d89f3270f46214f1569b7d48dcb5e25363b1d9b7353b82ced0fc67c0093a472` |
| mission | `P0=(0,0,0) → P1=(0,0,+0.20 NED) → P2=(2,0,+0.20 NED)` |
| guidance | `takeoff_then_align`, loop, 0.50 m/s, lookahead 0.40 m, reach 0.15 m |
| actuation | 동일 allocator/T200 inverse/SITL identity channel map |
| A | Gazebo truth가 guidance와 16-D policy observation을 구동 |
| B | DVL-EKF의 MAVLink `LOCAL_POSITION_NED`/attitude가 같은 경로를 구동 |
| 공통 shadow path | 두 run 모두 동일 DVL injector, no-GPS EKF3, rangefinder와 MAVLink telemetry 실행·기록 |

Water Linked-aligned profile은 `DVL_DOWN`, `VISION_POSITION_DELTA`, downward `DISTANCE_SENSOR`, `VISO_DELAY_MS=10`, no-GPS EKF3 ExternalNav이다. 이 실행은 원인 분리용 ideal-measurement lower bound이므로 artificial queue delay, velocity noise, FOM, dropout을 0으로 두고 confidence 100, `VISO_POS=0`을 사용했다. 실제 altitude가 3.6–3.8 m여서 modeled auto range rate는 5 Hz였고 모든 valid packet에 rangefinder가 발행됐다.

실기 Water Linked의 exact model, saved orientation/range mode, DVL-to-base xyz/rpy, per-packet FOM/noise/dropout/latency는 확보되지 않았다. 따라서 이 profile을 실제 센서의 exact digital twin으로 해석하지 않는다.

Wire-level로도 실제 extension은 VPD `time_usec=0`과 DVL AHRS `angle_delta`를 쓰지만, 현재 injector는 진단용 Gazebo source time과 `angle_delta=0`을 보낸다. 대상 ArduSub backend가 현재 두 값을 fusion에 사용하지 않아 이 A/B의 velocity 결론에는 영향이 없지만, exact packet parity라고 부르지는 않는다.

## A/B 유효성

두 run은 Gazebo, ArduSub EEPROM, EKF, policy integrator를 모두 fresh start했다. ARM 뒤 같은 rising-depth barrier에서 START했으며 source/config hash는 feedback source 한 줄을 제외하고 일치했다.

| 유효성 지표 | 결과 |
|---|---:|
| START GT position 차이 | 0.00196 m |
| START GT velocity 차이 | `7.8e-8 m/s` |
| START GT attitude 차이 | `0.000184 deg` |
| START DVL sequence 차이 | 0 |
| waypoint RLE | 양쪽 모두 `[0,1,2,1]` |
| full-cycle time | GT 11.602 s / DVL-EKF 11.882 s |
| action rate | GT 24.993 Hz / DVL-EKF 24.997 Hz |
| action max gap | GT 43.15 ms / DVL-EKF 41.79 ms |
| DVL invalid / rangefinder coverage | 0 / 100% |
| return edge→control inactive | GT 0.758 s / DVL-EKF 0.437 s |

마지막 `2→1` edge는 외부 supervisor가 감지한 뒤 ROS service로 STOP하므로 두 run 모두 0.25 s strict shutdown gate를 넘겼다. 이 사이 GT 19개, DVL-EKF 11개의 다음-outbound PWM sample이 발행됐다. 성능 분석은 `2→1` edge에서 끝내 이 sample들을 분모에서 제외했으며, 정확히 한 lap에서 같은 tick에 output을 닫는 lifecycle 구현은 별도 보완 사항이다.

## 직선 정상상태 비교

180° 전환을 straight metric에 섞지 않도록 outbound는 진입 첫 1 s와 전환 전 2 s를 제외했고, return은 전환 뒤 첫 2 s와 다음 outbound command가 publish되는 마지막 tick을 제외했다. 따라서 단일 run의 유효 정상상태 창은 outbound 약 1.2 s, return 약 1.8–2.1 s로 짧으며 반복 통계가 아니라 진단 결과다.

| 물리 GT 지표 | GT out | DVL-EKF out | GT return | DVL-EKF return |
|---|---:|---:|---:|---:|
| leg duration | 4.158 s | 4.200 s | 3.841 s | 4.121 s |
| `v_parallel` mean | 0.4498 | 0.4534 | **0.4718** | **0.4330** m/s |
| vector velocity RMSE | **0.0756** | **0.0993** | **0.0495** | **0.1312** m/s |
| cross-speed RMS | 0.0501 | 0.0811 | 0.0319 | 0.1060 m/s |
| cross-track RMS | 0.0012 | 0.0060 | 0.0064 | 0.0042 m |
| depth error RMS | **0.0086** | **0.1177** | **0.0068** | **0.1048** m |
| attitude error RMS | 4.19° | 6.04° | 4.31° | 6.91° |
| raw EKF↔GT velocity RMSE | 0.0375 | 0.0418 | 0.0285 | 0.0398 m/s |
| any-axis `abs(action)>=0.99` | **44.8%** | **73.3%** | **62.2%** | **71.2%** |
| requested-force clamp | 10.3% | 20.0% | 4.4% | 26.9% |

GT feedback의 straight physical tracking은 outbound의 `v_parallel`이 기준보다 0.0002 m/s 낮은 경계값인 것을 제외하면 velocity RMSE 기준을 통과한다. 그러나 GT에서도 pitch action은 steady out/return에서 `-1↔+1`, action-delta RMS `1.042/1.191`, cap `44.8/62.2%`다. 이는 runtime limiter가 만든 지속 plateau가 아니라 actor output이 tick 단위로 포화 왕복하는 control jitter다.

DVL-EKF는 outbound velocity RMSE를 31%, return RMSE를 165% 증가시켰고 return cross-speed를 3.3배로 만들었다. DVL arm의 EKF vertical position bias는 약 0.112 m이며 실제 depth error와 거의 같다. 이 profile은 `POSZ=Baro`, `VELZ=none`이므로 이 수직 오차를 DVL 센서 단독 문제가 아니라 Baro를 포함한 `LOCAL_POSITION_NED` feedback topology 문제로 분류한다.

## Takeoff와 180° 반전

| 지표 | GT feedback | DVL-EKF feedback |
|---|---:|---:|
| takeoff duration | 3.603 s | 3.561 s |
| takeoff physical path length | 0.224 m | 0.334 m |
| takeoff attitude RMS | 3.84° | 6.70° |
| takeoff force clamp | 5.56% | 20.22% |
| turn-window velocity RMSE | 0.1546 | 0.2059 m/s |
| turn-window cross-speed RMS | 0.0576 | 0.1238 m/s |
| turn-window depth RMS | 0.0146 | 0.1268 m |
| turn-window action cap | 66% | 65% |
| turn-window force clamp | 4% | 11% |

두 run 모두 180° 반전과 return arrival를 완료했다. 반전 중 큰 attitude transient와 약 65% action cap은 양쪽 공통이므로 waypoint reversal/policy 쪽 현상이다. DVL-EKF는 같은 반전에서 cross-speed를 2.15배, depth error를 8.7배로 늘리고 force clamp를 추가했다. 즉 반전은 공통 policy/action 문제를 크게 자극하고, estimator feedback은 그 결과를 다시 증폭한다.

## 판정과 다음 단계

- **Mission mechanics:** PASS — takeoff, outbound, 180° 반전, return 모두 완료
- **Exact one-lap shutdown:** FAIL — return edge 뒤 0.758/0.437 s 동안 control이 더 활성
- **Water Linked-like transport contract:** PASS — no-GPS VPD, rangefinder, valid/rate/schema 계약 통과
- **동일 초기조건 A/B:** PASS
- **GT physical straight tracking:** 대체로 PASS, 단 actor-bound/action jitter는 FAIL
- **DVL-EKF physical tracking/equivalence:** FAIL
- **frozen policy deployment acceptance:** 양쪽 모두 FAIL

따라서 3단계는 진행한다. 우선 G1–G7에서 paper command 의미, observation/integral/action 계약, velocity-priority reward, long-horizon transition curriculum과 validation을 수정한다. 새 policy의 acceptance는 GT와 DVL-EKF 양쪽에서 따로 요구해야 한다. 동시에 실제 Water Linked 설정과 raw packet/FCU artifact를 확보한 뒤 G8 sensor model의 rate·mount·noise·delay·dropout 범위를 실측값으로 교체한다.

## 재현 artifact

- 분석기: [`analyze_stage2_case_a_ab.py`](analyze_stage2_case_a_ab.py)
- machine-readable 결과: [`stage2_results/case_a_feedback_ab/STAGE2_CASE_A_GT_DVL_AB_STRICT.json`](stage2_results/case_a_feedback_ab/STAGE2_CASE_A_GT_DVL_AB_STRICT.json)
- GT run: [`stage2_case_a_gt_pair_20260817_180500`](stage2_results/case_a_feedback_ab/stage2_case_a_gt_pair_20260817_180500)
- DVL-EKF run: [`stage2_case_a_dvl_pair_20260817_180800`](stage2_results/case_a_feedback_ab/stage2_case_a_dvl_pair_20260817_180800)
- 공통 mission: [`stage2_case_a_0p5_2m.yaml`](stage2_case_a_0p5_2m.yaml)
- no-GPS Water Linked profile: [`stage2_waterlinked_default.parm`](stage2_waterlinked_default.parm)
