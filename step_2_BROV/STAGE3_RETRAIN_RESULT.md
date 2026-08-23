# Stage 3 deploy-v2 구현 및 재학습 결과

작성일: 2026-08-17  
목적: Stage 1/2에서 관측된 GT-feedback 공통 jitter와 DVL-EKF 증폭을 고려하여,
논문 공개 계약의 오구현을 바로잡고 Case-A 배포 명령을 포함한 새 정책을
headless Isaac Lab에서 재학습한다.

## 1. 결론

`deploy_v2` 정책의 1차 재학습과 Isaac Lab 검증을 완료했다.

- 학습: 2,048 environments, 128-step rollout, 50 iterations, seed 42
- 표본 수: 13,107,200 transitions
- 학습 wall time: 약 85.4 s
- 0.5 m/s 정상 구간: `v_parallel=0.48197 m/s`, velocity-vector
  `RMSE=0.02063 m/s`, cross-speed `RMS=0.00335 m/s`
- 0.1 m/s 정상 구간: `v_parallel=0.10020 m/s`, velocity-vector
  `RMSE=0.00360 m/s`
- zero-velocity hold 정상 구간: velocity-vector `RMSE=0.00308 m/s`
- 세 시험의 정상 구간 action-bound 및 thruster-force clamp 개입률: `0%`
- 모든 시험에서 한 tick 사이의 `-1 <-> +1` opposite-bound flip: `0회`

따라서 새 정책은 **Isaac native-state 평가의 steady 구간에서는 기존에 관측된
강한 action jitter를 재현하지 않았다.** 다만 이것만으로 Gazebo/Water Linked
DVL-EKF/실기 jitter가 해결됐다고 판정하지 않는다. 0.5 m/s의 180도 반전
과도구간을 포함하면 action-bound `5.03%`, force-clamp `5.14%`로 provisional
5% 기준을 각각 0.03/0.14%p 초과한다.

## 2. 이번에 구현한 계약

### 논문 공개 내용의 교정

- Eq. 9를 잘못된 velocity template로 사용하던 경로를 분리했다.
- `paper_ref_v1`은 Eq. 9의 analytic Frenet-Serret frame으로 시간변화하는
  `q_d(t)`를 만들고, `v_d^b`는 별도로 unit sphere에서 샘플하여 모든
  episode에서 정확히 `0.5 m/s`가 되도록 했다.
- 논문 공개 16-D observation
  `[q_e(4), v_e^b(3), omega^b(3), z_v(3), z_q(3)]`를 공용 pure-Torch
  contract로 분리했다.
- quaternion hemisphere를 고정하고, 적분 state를 `±5`로 clamp하며,
  reset/중복 observation에서 잘못 선적분하지 않도록 per-environment sample
  token을 적용했다.
- 논문에 명시됐지만 빠져 있던 mass domain randomization을 nominal mass의
  `+-5%`로 구현하고 inertia도 같은 비율로 조정했다. 범위는 논문 미공개이므로
  600 g ballast 시험에 맞춘 project assumption이다.

### Case-A용 deploy 확장

실제 학습 artifact는 exact-paper reference가 아니라 `deploy_v2`다.

- 5 s episode마다 2--3 s 사이에 정확히 한 번의 command transition
- balanced speed bins: `0`, `0.1`, `0.5 m/s`
- stop, restart, exact velocity reversal 및 180도 attitude command
- velocity tolerance `0.10 m/s` 중심의 precision reward
- action L2, delta-action, per-thruster physical-force clamp penalty
- policy FLU/Z-up action을 SNAME/FRD allocation으로 변환하는 명시적
  `T6=diag(1,-1,-1,1,-1,-1)` 계약
- T200 정/역 비대칭 force limit을 allocation 뒤에 실제 적용하고 requested와
  achieved force/wrench를 분리 기록

이 항목들은 논문 미구현분이 아니라 현재 Case-A gap을 줄이기 위한 프로젝트
확장이다.

### 학습 성능 개선

- headless에서도 실행되던 per-environment thrust-arrow drawing을 기본 OFF
- debug draw resource lazy initialization
- render interval을 100 Hz physics가 아니라 25 Hz policy step에 맞춤
- profile/seed/rollout/save interval CLI와 artifact manifest 추가
- resume, evaluation, export에서 profile 및 checkpoint SHA를 fail-closed 검증

2048-env capacity smoke는 약 133k--156k FPS였고, 최종 iteration은
153,035 FPS였다. 기존 model_299 학습의 약 2.4k FPS 병목은 debug visualization
경로가 주원인이었다.

## 3. 학습 실행

```bash
python train.py --headless \
  --num_envs 2048 \
  --max_iterations 50 \
  --num_steps_per_env 128 \
  --save_interval 5 \
  --experiment_name stage3_deploy_v2_2048x128_seed42_20260817 \
  --profile deploy_v2 \
  --seed 42
```

| 지표 | 시작 | 종료/최대 |
|---|---:|---:|
| mean reward | 18.9505 | 113.7405 |
| policy action std | 0.9802 | 0.4355 |
| total FPS | 128,950 | 153,035, max 155,221 |
| collection time/iteration | 1.895 s | 1.658 s |
| value loss | 4.280 | peak 56.543 후 38.400 |

이번 시간 제한형 run은 첫 배포 candidate를 얻기 위한 50-iteration run이다.
reward가 마지막 iteration까지 증가했더라도, 0.5 m/s steady gate가 이미 충분히
통과하므로 원인 분리 없이 iteration 수만 늘리지 않는다.

## 4. Isaac 평가

평가는 seed 42, nominal mass/inertia/hydrodynamics, native simulator state,
25 Hz policy rate로 수행했다. `steady`는 시작 1 s와 각 waypoint transition
전후 1 s를 제외한다.

| 시험 | 구간 | `v_parallel` | vector RMSE | cross RMS | action bound | force clamp |
|---|---|---:|---:|---:|---:|---:|
| hold 0 | steady | - | 0.00308 | - | 0.00% | 0.00% |
| straight 0.1 | steady | 0.10020 | 0.00360 | 0.00289 | 0.00% | 0.00% |
| straight 0.5 | steady | 0.48197 | 0.02063 | 0.00335 | 0.00% | 0.00% |
| straight 0.1 | whole, 1 reversal | 0.09993 | 0.00887 | 0.00377 | 0.93% | 0.20% |
| straight 0.5 | whole, 3 reversals | 0.46445 | 0.11758 | 0.01885 | 5.03% | 5.14% |

해석:

1. 0/0.1/0.5 m/s steady tracking은 모두 통과했다.
2. 0.5 m/s 반전 과도응답이 아직 전체-run error와 약 5%의 limit 개입을 만든다.
3. 이는 기존 GT Case-A에서 steady pitch가 매 tick `-1 <-> +1`로 왕복하던
   것과 다르다. 새 policy는 steady 구간에서 bound 개입과 opposite-bound flip이
   모두 0이다.
4. 평가는 Isaac native state이므로 Water Linked DVL, Baro, ArduSub EKF 및
   `LOCAL_POSITION_NED`의 gap을 포함하지 않는다. 다음 판정은 같은 artifact를
   Gazebo GT와 DVL-EKF feedback에 각각 넣어야 한다.

## 5. 최종 artifact

- checkpoint: `logs/stage3_deploy_v2_2048x128_seed42_20260817/model_49.pt`
  - SHA256: `9e19fc8b5e59d5d632891f0b671ca3d4a0f5e4e1d48643b1c083f5385f45f26c`
- training manifest:
  `logs/stage3_deploy_v2_2048x128_seed42_20260817/artifact_manifest.json`
- TorchScript:
  `logs/stage3_deploy_v2_2048x128_seed42_20260817/exported_jit/policy.pt`
  - SHA256: `c185869418f13d868b8d71c4ca8f6f245a9d7103bca36704870df4a738ac2c4f`
- TorchScript metadata:
  `logs/stage3_deploy_v2_2048x128_seed42_20260817/exported_jit/policy.pt.metadata.json`
  - SHA256: `93f66730e7021ecbf5eaa043e4f57b318d475d2a12ecf638ebbadd8c9272d0de`
- evaluation JSON:
  - `eval_hold_0p0.json`
  - `eval_straight_0p1.json`
  - `eval_straight_0p5.json`
- evaluation plots:
  - `plots/policy_eval_velocity_hold_0mps_deploy_v2.png`
  - `plots/policy_eval_straight_line_0.1mps_deploy_v2.png`
  - `plots/policy_eval_straight_line_0.5mps_deploy_v2.png`

TorchScript는 `(N,16) -> (N,6)` finite inference를 통과했고, checkpoint의
deterministic actor MLP 여섯 tensor와 bitwise exact하게 일치한다.

## 6. 후속 MK2 배포 결과

위에서 식별한 blocker는 별도 `policy_node_mk2`, metadata/SHA fail-closed gate,
명시적 T6 및 legacy 계약 분리로 해결했다. 이후 fresh Gazebo에서 GT feedback과
Water Linked-aligned DVL-EKF feedback을 각각 실행했다.

배포 artifact, TorchScript replay, T6-to-wrench 및 lifecycle은 통과했지만,
GT에서도 whole-cycle action-bound 점유율 98.9%, force clamp 30.2%로 제어 성능은
실패했다. 따라서 이 model_49 artifact는 실기 승격 없이 진단 후보로 격리했다.
정식 결과와 재실행 방법은 `MK2_SIM2SIM_DEPLOY_RESULT.md`를 따른다.
