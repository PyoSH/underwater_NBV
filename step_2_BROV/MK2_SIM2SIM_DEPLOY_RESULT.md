# MK2 재학습 정책의 Gazebo SITL 배포 결과

작성일: 2026-08-17  
대상 정책: `sim2swim_deploy_v2_mk2_s42_i49`  
판정: **artifact/inference/T6/mission sequence PASS, control/lifecycle FAIL,
실기 승격 금지**

## 1. 결론

재학습된 `deploy_v2` 정책을 기존 `model_299`와 분리된 MK2 artifact,
실행 노드, 설정 및 launch로 `brov_ros2`에 연결했다. Fresh Gazebo/ArduSub에서
동일한 2 m Case-A-shaped 미션을 다음 두 feedback으로 각각 한 번 수행했다.

1. Gazebo ground truth를 정책 observation에 사용하는 원인 격리 실행
2. GPS를 완전히 끄고 Water Linked 기본 코드 경로에 맞춘
   `VISION_POSITION_DELTA -> ArduSub EKF3 -> LOCAL_POSITION_NED` 실행

두 실행 모두 `takeoff -> outbound -> 180도 반전 -> return`의 waypoint RLE
`[0,1,2,1]`을 완주했고, fault 없이 STOP, neutral, disarm까지 끝났다. 그러나
MK2 정책은 GT feedback에서도 action이 거의 항상 한계에 걸리고 큰 횡방향
속도를 만들었다. 따라서 이번 결과는 **기록된 artifact 선택, TorchScript
inference, T6-to-requested-wrench 및 mission sequence가 의도대로 작동했다는
증거인 동시에, 이 재학습 모델은 Gazebo/실기 배포 후보로 부적합하다는
부정적 검증 결과**다. 전체 deployment acceptance는 통과하지 않았다.

평균 진행방향 속도만 보면 약 0.44--0.47 m/s로 목표 0.5 m/s에 가깝지만,
그 값은 jitter를 숨긴다. GT 실행의 outbound velocity-vector RMSE는
0.355 m/s, cross-speed RMS는 0.328 m/s였고, 전체 cycle의 action-bound
점유율은 98.9%였다.

## 2. MK2 배포 구현

기존 `demo_policy` 및 legacy node를 덮어쓰지 않았다.

- 별도 policy bundle:
  `brov_ros2-main/artifacts/policies/sim2swim_deploy_v2_mk2_s42_i49`
- 별도 executable: `policy_node_mk2`
- 별도 launch: `sim2sim_mk2_case_a.launch.py`
- 별도 controller/mission:
  `rl_controller_mk2_deploy_v2.yaml`,
  `mission_sim2sim_mk2_case_a_0p5.yaml`
- metadata-bound contract validator: `brov_control/policy_contract.py`
- full fresh-run harness: `run_mk2_case_a_deploy.sh`
- host build/sync/run wrapper: `run_mk2_case_a_deploy_host.sh`
- MK2 analyzer: `analyze_mk2_case_a_ab.py`
- rosbag TorchScript replay: `validate_mk2_bag_replay.py`

MK2 node는 sibling metadata와 policy/vehicle SHA를 검증한 뒤에만 시작한다.
배우의 FLU/Z-up action을 clip/scale한 다음, allocation matrix 앞에서 정확히

`T6 = diag(1,-1,-1,1,-1,-1)`

을 적용해 SNAME/FRD wrench로 바꾼다. 알 수 없거나 metadata가 없는 계약은
시작 시 거부한다. bundled model_299에는 checksum-bound legacy sidecar를
추가했으며, sidecar가 없는 artifact는 거부한다.

## 3. 실행 계약과 범위

두 arm은 각각 Gazebo와 ArduSub를 fresh boot하고 동일한 다음 계약을 썼다.

- policy SHA:
  `c185869418f13d868b8d71c4ca8f6f245a9d7103bca36704870df4a738ac2c4f`
- checkpoint SHA:
  `9e19fc8b5e59d5d632891f0b671ca3d4a0f5e4e1d48643b1c083f5385f45f26c`
- vehicle SHA:
  `8bb397f4a8a0d50c11bfaf1f88143b4375b84dbd77f41fbe7c512ced1b15be12`
- observation: `brov_velocity_observation_v2`, 16-D
- action: `explicit_flu_zup_to_sname_frd_v1`, 6-D
- mission: relative P0 `(0,0,0)`, P1 `(0,0,0.20)`, P2 `(2,0,0.20)`
- `takeoff_then_align`, loop, 0.50 m/s, lookahead 0.40 m, reach 0.15 m
- policy/control rate: 25 Hz
- GPS off, EKF3 ExternalNav, `VISO_TYPE=1`, POSXY/VELXY source 6
- Water Linked code-default topology: DVL_DOWN, POSITION_DELTA, rangefinder,
  10 Hz below 3 m / 5 Hz above 3 m

두 run의 START 시 physical GT position, velocity, attitude 및 DVL source time은
동일했다. DVL diagnostic sequence의 절대 번호는 프로세스별 startup count라
10 차이가 있었지만, 대응 source time은 둘 다 27.76 s였다. 분석기는 이
process-local 번호가 아니라 source-time 차이 0.00 s를 공정성 gate로 사용한다.

중요한 범위 제한이 있다. GT feedback과 production resolved mission은 현재
fail-closed로 함께 쓸 수 없으므로, 이번 인과 A/B는 실제 카메라/pool
localization/mission-manager를 우회한 **direct-relative Case-A-shaped** 미션이다.
따라서 전체 production Sim2Swim orchestrator 승인 시험으로 표현하지 않는다.

## 4. 실제 Gazebo 결과

아래 steady 구간은 각 수평 leg 시작 1 s와 waypoint edge 마지막 0.05 s를
제외했다.

| feedback/구간 | v_parallel | vector RMSE | cross-speed RMS | depth RMS | attitude RMS/max | action cap | force clamp |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT outbound | 0.471 | 0.355 | 0.328 | 0.024 m | 3.27/5.64 deg | 100.0% | 34.7% |
| GT return | 0.461 | 0.254 | 0.230 | 0.047 m | 15.68/52.88 deg | 98.6% | 17.8% |
| DVL-EKF outbound | 0.438 | 0.471 | 0.417 | 0.093 m | 4.18/7.90 deg | 100.0% | 52.4% |
| DVL-EKF return | 0.458 | 0.422 | 0.364 | 0.110 m | 17.19/61.39 deg | 100.0% | 52.7% |

전체 cycle action cap/force clamp는 GT에서 98.9%/30.2%, DVL-EKF에서
98.4%/47.5%였다. 25 Hz timing, artifact contract, waypoint RLE 및 stop 후
neutral은 통과했다. 반면 exact one-lap lifecycle gate는 실패했다. `2 -> 1`
edge 뒤 supervisor polling 때문에 STOP이
약 0.439 s 늦어 다음 outbound PWM 11개가 발생했지만, inactive 이후 PWM은
0개였고 neutral echo는 GT 40 ms, DVL-EKF 0.3 ms 이내였다. 이 운영 지연은
별도 개선 항목이며 위 tracking 실패를 설명하지 않는다.

## 5. legacy model_299와 비교

기존 Stage-2 저장 JSON은 turn 주변 2 s를 제외했으므로, 공정 비교를 위해 두
legacy bag을 MK2와 같은 `start settle=1 s`, `edge guard=0.05 s`로 다시
분석했다. 결과는
`STAGE2_CASE_A_GT_DVL_AB_WINDOW_1S_0P05.json`에 별도로 보존했다.

Legacy 값은 현재 `brov_ros2` observation builder와 명시적 legacy no-T6
action path에서 실행된 deployed baseline이다. 과거 training-time observation
구현 전체를 재현한 값으로 해석하지 않는다.

| feedback/구간 | legacy vector RMSE | MK2 vector RMSE | legacy cross RMS | MK2 cross RMS | legacy cap | MK2 cap |
|---|---:|---:|---:|---:|---:|---:|
| GT outbound | 0.068 | 0.355 | 0.046 | 0.328 | 55.1% | 100.0% |
| GT return | 0.067 | 0.254 | 0.049 | 0.230 | 60.0% | 98.6% |
| DVL-EKF outbound | 0.086 | 0.471 | 0.067 | 0.417 | 65.8% | 100.0% |
| DVL-EKF return | 0.162 | 0.422 | 0.144 | 0.364 | 61.0% | 100.0% |

즉 MK2는 평균 `v_parallel`은 만들지만 다른 translational axis를 강하게
흔든다. GT feedback에서도 발생하므로 DVL/EKF만이 원인은 아니다. DVL-EKF는
여기에 depth, vector error 및 force-clamp gap을 더한다.

## 6. 기록된 artifact/inference/T6 경로 검증

### T6와 allocator 입력

- GT active sample 379개, DVL-EKF 398개가 모두 action-wrench pairing됨
- `wrench_sname - action_flu * scale * T6` 최대 절대오차:
  `3.8147e-6`
- metadata/profile/action/observation/policy/vehicle SHA 모두 exact match

따라서 이번 bag의 실패는 T6 누락, T6 적용 위치 또는 잘못된 artifact 로딩으로
설명되지 않는다. 아직 비교하지 않은 training/runtime observation 분포나 의미
차이까지 배제하는 검사는 아니다.

### TorchScript rosbag replay

- GT 962개, DVL-EKF 998개 observation을 bag에서 다시 inference
- 기록 action 대비 최대오차 `4.7684e-7`, RMS 약 `7e-8`
- replay PASS
- unbounded actor output이 unit 범위를 벗어난 sample:
  GT 99.58%, DVL-EKF 89.78%

즉 export/runtime inference는 학습 checkpoint와 일치하며, Gazebo observation에
대한 actor overflow가 매우 크다는 것이 직접 확인됐다. 입력 분포 mismatch와
bounded-action 학습 계약 mismatch 중 어느 쪽이 지배적인지는 다음 replay
비교에서 분리해야 한다.
현재 `/brov/policy/action_raw`는 `PolicyRunner` clip 이후 값이므로, 위 unbounded
비율은 별도 offline replay로 복원했다.

## 7. Water Linked 정합성의 한계

이번 DVL arm은 GPS 없이 실제 Water Linked BlueOS 기본 **메시지 토폴로지**인
`VISION_POSITION_DELTA`와 rangefinder를 거쳐 EKF3/LOCAL_POSITION_NED를
사용했다. 그러나 실제 장착 DVL의 model, mount xyz/rpy, 저장된
`orientation/should_send/rangefinder`, raw cadence, FOM, latency, dropout 및
bottom-lock 분포가 확보되지 않았다.

이번 실행은 altitude 약 3.6--3.8 m라 5 Hz branch였고, injector queue delay,
noise, FOM과 dropout을 0, confidence를 100, mount를 base origin
(`VISO_POS=0`)으로 둔 ideal lower-bound다. ArduSub의 `VISO_DELAY_MS=10`은
유지했다. 따라서 “실제 Water Linked 센서를 정확히
복제했다”거나 “실기 DVL에서도 같은 수치가 나온다”고 주장할 수 없다.
그럼에도 ideal DVL-EKF보다 GT에서도 이미 크게 실패했으므로, 현재 MK2
candidate를 실기로 보내지 않는 판정에는 충분하다.

## 8. 다음 재학습의 최소 보완 순서

시간 제약상 iteration만 늘리지 않는다.

1. 이번 GT rosbag의 16-D observation을 Isaac policy에 replay하여 training
   observation 분포와 축별 범위를 비교하고, unit 밖 actor 출력의 첫 원인을
   확정한다.
2. PPO와 배포가 같은 bounded-action 계약을 쓰게 하고, pre-clamp actor와
   overflow를 reward와 로그에 포함한다. 현재처럼 unbounded Gaussian 출력을
   먼저 clip하고 reward/actuator penalty가 clipped action만 보면 90--100%의
   raw overflow **크기**를 구분하지 못한다. PPO log-prob의 raw sample 처리와는
   별개인 문제다.
3. training curriculum에 실제 ROS Case-A 명령을 그대로 넣는다: 양성부력으로
   시작하는 약 0.10 m/s 수직속도, takeoff, 2 m 직선, 즉시 180도 q/v 반전,
   stop/restart를 한 episode/horizon 안에서 학습한다.
4. steady actor pre-clip 초과율, action delta, requested-force clamp를 배포 gate와
   같은 정의로 학습/evaluation에 넣는다. 새 candidate의 steady cap 목표는
   1% 미만, whole cycle 5% 미만이다.
5. Gazebo/실기에서 측정된 actuator gain/lag, 부력·drag·added mass 및 관측
   hold/delay 범위만 domain randomization에 추가한다. 임의 sensor/plant tuning은
   하지 않는다.
6. 시각화는 계속 완전히 끄고, 짧은 headless retrain 뒤 Isaac뿐 아니라 fresh
   GT Case-A gate를 먼저 통과한 모델만 DVL-EKF로 진행한다.

## 9. 산출물과 재실행

- strict 분석:
  `stage3_results/mk2_case_a_feedback_ab/MK2_CASE_A_GT_DVL_AB_STRICT.json`
- GT run:
  `brov_ros2-main/runtime/experiments/sim2sim_mk2_case_a_20260817_gt_r2`
- DVL-EKF run:
  `brov_ros2-main/runtime/experiments/sim2sim_mk2_case_a_20260817_dvl_r1`
- 각 run의 `bag/`, `mav.tlog`, DataFlash `.BIN`, params, manifest,
  `policy_replay.json`을 보존했다.

Fresh run은 host에서 다음처럼 실행한다. wrapper가 support files를 container에
동기화하고 `install_mk2` overlay를 재빌드한 뒤 실행한다.

```bash
./step_2_BROV/run_mk2_case_a_deploy_host.sh gazebo_truth mk2_gt_repeat
./step_2_BROV/run_mk2_case_a_deploy_host.sh mavlink_ekf mk2_dvl_repeat
```

이번 artifact는 결과 보존과 다음 재학습 비교용으로만 유지한다. 기존
`demo_policy`를 대체하지 않았고, real vehicle에는 배포하지 않는다.
