# step_2_BROV — BlueROV2 수중 궤적 추종 RL 환경

## 프로젝트 목표

BlueROV2 Heavy(8-thruster)를 IsaacLab으로 시뮬레이션하는 RL 환경. 두 갈래로 구성된다:

1. **물리 검증 단계 (완료)** — `BROVTrajEnv`: 수중 동역학/유체역학이 실제 BROV2 거동과 일치하는지 확인, 추진기 모델(PWM→추력) 정확도 확인, 중성부력/자세 안정성 확보. `bottom_up.py`의 축별 테스트로 검증됨.
2. **Sim2Swim 재현 단계 (구현·1차 검증 완료, 본 학습 전)** — `BROVVelEnv` + `LOSGuidance`: "Sim2Swim: Zero-Shot Velocity Control for Agile AUV Maneuvering in 3 Minutes"(arXiv:2512.08656, SINTEF Ocean) 논문을 계층형(고전 유도 + RL 저수준 속도 컨트롤러)으로 재현. 실물 실험체가 BlueROV2 Heavy로 동일.

**향후 계획**:
- step_1_NBV(수중 인지)과 step_2_BROV(수중 물리)를 통합하는 **step_3** — `LOSGuidance`를 커버리지/NBV 기반 유도 모듈로 교체하고, 학습된 Sim2Swim 저수준 정책은 그대로(frozen) 재사용하는 구조를 염두에 둠. 이 때문에 물리 엔진(Fossen 코어+BROV2 액추에이터)을 `robots/dynamics/`로 승격해 step_2/step_3이 공유하도록 리팩토링하고(2026-07), step_2_BROV 내부도 `envs/`/`guidance/`/`physics_tests/`/`legacy/` 역할별 서브디렉토리로 재배치함(2026-07).
- **부족구동(underactuated) 어뢰형 AUV** 지원 — MarineGym의 iAUV(주추진기1+핀4, cruciform rudder) 구현을 참고 아키텍처로 확정. 자세한 로드맵은 메모리 참조.

---

## 파일 구조

역할별 서브디렉토리로 재배치됨(2026-07) — 파일명은 `envs/traj_env.py`처럼 basename을 겹치지 않게 지어서, 이 코드베이스의 `sys.path.insert()`+bare-import 관례에서 동일 파일명이 다른 디렉토리에 있을 때 생기는 `sys.modules` 캐시 충돌 위험을 피했다. 모든 상호 import는 `agents.rsl_rl_ppo_cfg`와 동일하게 dotted-package 형태(`from envs.vel_env import BROVVelEnv`)를 쓴다.

```
step_2_BROV/
├── CLAUDE.md              ← 이 파일
│
├── envs/                              # DirectRLEnvCfg/DirectRLEnv 정의
│   ├── scene_cfg.py                   #   BROVSceneCfg — traj_env_cfg.py/vel_env_cfg.py 공유
│   ├── traj_env_cfg.py, traj_env.py   #   BROVTrajEnv — end-to-end 경로추종(물리검증용), PWM 8-dim 액션
│   └── vel_env_cfg.py, vel_env.py     #   BROVVelEnv  — Sim2Swim 저수준 6DOF 속도/자세 컨트롤러, wrench 6-dim 액션
├── guidance/los_guidance.py           # LOSGuidance — 3D LOS 유도(고전 제어). BROVVelEnv.attach_guidance()로 연결
├── physics_tests/bottom_up.py         # 물리 검증 테스트 함수 5종 (neutral_buoyancy/straight_line/rotation/six_dof/thruster_model)
├── legacy/brov_env.py                 # [레거시] MIT CSAIL origin, 실행 불가 — 참조 전용
├── agents/rsl_rl_ppo_cfg.py           # BROVVelPPORunnerCfg — RSL-RL PPO 설정(rsl-rl-lib>=5.0.0 신 스키마)
│
├── train.py                # BROVVelEnv RSL-RL 학습 런처
├── test_policy.py          # 학습된 정책 검증 — LOSGuidance+정책 통합, 논문 Fig.4 (a)/(b)/(c) 3-trial 재현
├── validate_physics.py     # BROVTrajEnv 물리 검증 테스트 런처 (--test 플래그)
└── run_experiment.sh       # train.py → test_policy.py(Fig.4 3-trial) 순차 자동화 (컨테이너 내 python.sh 직접 호출)

공유 물리 엔진 (상위 디렉토리, step_2/step_3 공용 — 2026-07 리팩토링으로 승격):
../robots/dynamics/fossen.py            # Hydrodynamics — Fossen 6-DOF 코어(기종 무관, 계수는 생성자 인자)
../robots/dynamics/brov2/thruster.py    # BROV2ThrusterModel, build_allocation_matrix — BROV2 8-스러스터 전용
../robots/dynamics/brov2/params.py      # load_brov2_yaml 등 — brov2_heavy.yaml 로더/좌표변환

공유 에셋:
../robots/assets/brov_rigid.py          # ArticulationCfg, BROV_RIGID_CFG (BlueROV2_buoyancy.usd 사용)
../robots/data/BROV2/
  ├── BlueROV2_buoyancy.usd             # 실제 사용 USD 에셋
  └── brov2_heavy.yaml                  # 부력/CoB/유체계수/스러스터 위치·방향 정본 (mass/inertia/collision은 USD가 정본)
```

---

## 아키텍처 개요

### 두 갈래 의존 관계

```
[BROVTrajEnv — 물리검증]                    [BROVVelEnv — Sim2Swim]
validate_physics.py                          train.py / test_policy.py
  ├── envs/traj_env_cfg.py → BROVTrajEnvCfg    ├── envs/vel_env_cfg.py → BROVVelEnvCfg
  │     └── envs/scene_cfg.py → BROVSceneCfg   │     └── envs/scene_cfg.py (공유)
  ├── envs/traj_env.py → BROVTrajEnv           ├── envs/vel_env.py → BROVVelEnv
  └── physics_tests/bottom_up.py → test_* 함수  │     └── guidance/los_guidance.py → LOSGuidance (평가 시만 연결)
                                                 └── agents/rsl_rl_ppo_cfg.py → BROVVelPPORunnerCfg

  둘 다 → robots/dynamics/fossen.py (Hydrodynamics)
        → robots/dynamics/brov2/thruster.py (BROV2ThrusterModel, build_allocation_matrix)
        → robots/dynamics/brov2/params.py (load_brov2_yaml, coBM_vector_ned, thruster_pos_dir_ned)
```

### BROVVelEnv 물리 루프 (Sim2Swim, 매 policy step)

```
_pre_physics_step   →  action(6-dim wrench) clamp [-1, 1]
  (decimation=4 반복)
    _apply_action
      τ_cmd = F_max * action                          # 논문 Table 4 실측 최대추력(von Benzon et al. 2022)
      f_desired(8) = B_pinv @ τ_cmd                    # build_allocation_matrix()의 pseudo-inverse
      pwm = BROV2ThrusterModel.inverse_thrust(f_desired)
      f_thrust, t_thrust = BROV2ThrusterModel.compute(pwm)
      f_hydro,  t_hydro  = Hydrodynamics.compute(...)
      set_forces_and_torques (body_ids=[0])
_get_observations   →  16-dim obs (q_e, v_e_b, ω_b, z_v, z_q)
_get_rewards        →  Sim2Swim Eq.5-8
_get_dones          →  terminated(out_of_bounds) / truncated(episode_length_s)
```

`attach_guidance(los)` 미호출 시(학습 중) `_current_v_d_b()`가 논문 Eq.9 랜덤 궤적을 자동생성, 호출 시(평가/배포) `LOSGuidance.compute()`로 대체 — 호출 지점 하나만 갈아끼우는 구조라 `_get_observations()` 중복 호출/적분항 이중누적 문제가 없음.

### BROVTrajEnv 물리 루프 (물리검증, 매 policy step)

```
_pre_physics_step   →  action(8-dim PWM) clamp [-1, 1]
  (decimation=4 반복)
    _apply_action
      f_thrust, t_thrust = BROV2ThrusterModel.compute(pwm)   # PWM 직접 입력, 할당행렬 없음
      f_hydro,  t_hydro  = Hydrodynamics.compute(...)
      set_forces_and_torques (body_ids=[0])
_get_observations   →  17-dim obs (pos_env, quat, lin_vel_b, ang_vel_b, wp_dir_b, wp_dist)
_get_rewards        →  progress + waypoint + action + upright
_get_dones          →  terminated(out_of_bounds) / truncated(episode_length_s)
```

---

## Sim2Swim(BROVVelEnv) MDP 정의

### 관측 (16차원)
| 인덱스 | 량 | 정의 |
|--------|-----|--------|
| 0-3 | q_e | 쿼터니언 오차 `q̄_d ⊗ q` [w,x,y,z] |
| 4-6 | v_e_b | body-frame 속도 오차 `v^b - v_d^b` |
| 7-9 | ω_b | body-frame 각속도 |
| 10-12 | z_v | v_e_b 적분 상태 |
| 13-15 | z_q | q_e vector part 적분 상태 |

### 행동 (6차원, [-1,1])
surge, sway, heave, roll, pitch, yaw 스케일. `τ_cmd = F_max * a` → `B_pinv`로 8-thruster PWM 할당.

**F_max** (논문 Table 4, von Benzon et al. 2022 실측값): `[85.0, 85.0, 120.0, 26.0, 14.0, 22.0]` (surge,sway,heave,roll,pitch,yaw → N,N,N,N·m,N·m,N·m)

**할당행렬 B (6×8)** — `build_allocation_matrix()`가 YAML 위치/방향에서 매번 계산(하드코딩 안 함). 참고용 근사값(YAML 갱신 시 실제 값 재계산됨):
```
Fx: [-0.7071,-0.7071, 0.7071, 0.7071,   0,      0,      0,      0    ]
Fy: [ 0.7071,-0.7071, 0.7071,-0.7071,   0,      0,      0,      0    ]
Fz: [ 0,       0,      0,      0,      1,      1,      1,      1    ]
Tx: [ 0.0011,-0.0011, 0.0011,-0.0011, 0.2177,-0.2177, 0.2177,-0.2177]
Ty: [ 0.0011, 0.0011,-0.0011,-0.0011,-0.1290,-0.1290, 0.1110, 0.1110]
Tz: [ 0.1692,-0.1692,-0.1683, 0.1683,   0,      0,      0,      0    ]
```

### 보상 (Sim2Swim Eq.5-8)
```
r = w_quat·exp(-‖q_e_vec‖²) + w_vel·exp(-‖v_e_b‖²) + w_omega·exp(-‖ω_b‖²)
  + w_quat·exp(-∠(q_d,q))                                    ← Eq.7, 별도 4번째 항(제곱 없음). w_quat을 Eq.6 항과 재사용(논문 Table 1 표기 그대로)
  + w_action·exp(-‖a‖)
```
가중치(Table 1): `w_quat=0.4, w_vel=0.2, w_omega=0.05, w_action=0.3`

### 속도 명령 (학습 중, 논문 Eq.9)
`v_d^b(t) = q_cmd ⊗ [a, b·sin(ωt), c·cos(ωt)]`, `[a,b,c]=[0.5,0.5,0.3]`, `ω=0.2 rad/s`, `q_cmd`는 에피소드마다 랜덤 샘플.

### 도메인 랜덤화 — 1단계 (mass는 2단계로 연기)
| 파라미터 | 중심값(YAML 실측) | 범위 | 근거 |
|---|---|---|---|
| `volume` | 0.014665 m³ | ±10% → [0.01320, 0.01613] | 부력 부호 전환까지 포함(Sim2Swim 검증 시나리오 재현) |
| `coBM` | (0,0,0.010) m | 반경 15mm 구 균등 샘플 | YAML 자체가 "실측 안 됨"으로 표기 |
| `added_mass[3:]`(Kṗ,Mq̇,Nṙ) | [0.189,0.135,0.222] | ±40% | von Benzon 논문이 명시한 회전축 added-mass 오차 30-100%의 중간값 |
| **mass** | 14.635 kg | **미구현(2단계)** | PhysX 질량 직접 조작(`root_physx_view.set_masses()`) 필요, 두 참고 코드베이스(legacy `Project_BROV`, "Learning to Swim" 논문) 모두 선례 없음 |

### 종료
- `terminated`: env origin 기준 위치가 `max_bound`(기본 20m) 초과
- `truncated`: `episode_length_s`(학습 5.0s) 경과

---

## LOSGuidance (`guidance/los_guidance.py`)

경로 추종은 RL이 아니라 고전 3D LOS 유도가 담당 — `attach_guidance()`로 `BROVVelEnv`에 연결하면 매 스텝 `compute(pos_env, root_quat_w) -> (v_d_b, q_d)`를 자동 호출. 완전구동(BROV2)이라 표준 Fossen LOS(선수각 조향)가 아니라 "월드 프레임에서 lookahead 지점을 향하는 속도벡터"를 만드는 attitude-independent 방식.

`heading_mode`: `"align"`(진행방향 정렬) / `"upright"`(항상 수평) / `"random_at_waypoint"`(웨이포인트 도달마다 `roll,pitch~U(-π/2,π/2), yaw~U(-π,π)` 재샘플, Sim2Swim Trial(c) 재현).

**주의(실제 겪은 버그)**: 도달 판정은 `next_wp`(지금 향하는 목표) 기준이어야 한다 — `cur_wp`(이미 지나온 출발점) 기준으로 하면 전진할수록 오히려 멀어져서 영원히 도달 판정이 안 나고, lookahead 지점이 고착되며 근방에서 `to_los` 벡터가 0에 가까워져 목표속도가 발산한다.

---

## 정책 검증 워크플로우 (`test_policy.py`)

Sim2Swim 논문은 시뮬레이션 단독 검증 없이 학습 reward 곡선 → 바로 실물 3-trial(Fig.4)로 넘어감. `test_policy.py`가 이 3-trial을 시뮬레이션으로 재현:

```bash
python test_policy.py --checkpoint logs/<exp>/model_XXX.pt --test straight_line               # Fig.4(a)
python test_policy.py --checkpoint logs/<exp>/model_XXX.pt --test square_ballast --duration 60  # Fig.4(b)
python test_policy.py --checkpoint logs/<exp>/model_XXX.pt --test square_random_attitude        # Fig.4(c)
```

- `square_ballast`(600g 밸러스트, 논문 Trial(b))는 **근사**: 실제 PhysX mass는 안 바꾸고 동일한 순부력 결손을 내는 volume 감소 + port 방향 CoB 오프셋으로 대체(mass DR 2단계 미구현이라 완전 재현 아님).
- 결과 플롯은 논문 Fig.4 레이아웃 재현(위 4단: u/v/w 각 패널 + 자세 통합 패널, 아래: 3D 궤적 — Position/Waypoints/Radius of acceptance/Forward direction, 시작=초록X/끝=빨강X). `plots/policy_eval_<test>.png`.
- `--record_video`: 경로 전체를 내려다보는 고정 조망 카메라(웨이포인트 바운딩박스에서 자동 계산) — `validate_physics.py`의 로봇-추적 체이스캠과는 다른 방식(경로 맥락을 보여주기 위함).
- 재현성을 위해 `env.reset()`이 매번 넣는 랜덤 도메인 랜덤화 값을 시나리오별 고정값으로 덮어씀(`_apply_physics_scenario()`).

---

## 물리 검증 워크플로우 (`validate_physics.py`, BROVTrajEnv)

```bash
python validate_physics.py --test neutral_buoyancy --duration 10.0 [--headless]
python validate_physics.py --test straight_line --thrust 0.5 --duration 3.0
python validate_physics.py --test rotation --thrust 0.3 --duration 3.0
python validate_physics.py --test six_dof --thrust 0.5 --rotation_thrust 0.3 --duration 3.0 --headless
python validate_physics.py --test thruster_model
# 위 테스트에 --record_video 추가 시 로봇을 따라가는 체이스캠으로 mp4 기록
```

### 검증 기준
- **중성부력**: |ΔZ| < 0.1 m / 10 s — 하강 시 `brov2_heavy.yaml`의 `volume` 증가, 상승 시 감소
- **직선이동**: 주축 변위 > 0.05 m, 횡방향 표류 < 주축 변위 × 0.5
- **추진기**: PWM=1.0에서 정방향 약 64.1N, 역방향 약 -51.5N (YAML 실측)

---

## 디버그 시각화 (`envs/vel_env.py`)

`BROVVelEnv`에 `_set_debug_vis_impl`/`_debug_vis_callback`(IsaacLab 내장 훅) + `isaacsim.util.debug_draw` 병용:
- 🟢 초록 화살표: 현재 자세 (로봇 위치에 표시)
- 🔴 빨강 화살표: 목표 자세 q_d (같은 위치에 겹쳐 그림 — 위치 목표가 없는 속도컨트롤러라 자세 오차를 직접 비교하기 위함)
- 🔵 파랑 화살표: 목표 속도 v_d_b (방향+길이=속력 비례)
- 🟠 주황 화살표 8개: 스러스터별 실제 추력 (길이=추력 크기, `envs/traj_env.py`의 `BROVTrajEnv`도 동일 방식 — 현재는 각 파일에 중복 구현돼 있음, 통합 여지 있음)

---

## 알려진 이슈

1. **`traj_env_cfg.rew_scale_terminated` 미적용**: `envs/traj_env.py._get_rewards()`에서 미사용.
2. **mass 도메인 랜덤화 미구현**: 위 도메인 랜덤화 표 참조. 2단계 작업.
3. **디버그 시각화 코드 중복**: `envs/traj_env.py`/`envs/vel_env.py`에 스러스터 추력 화살표 로직이 각각 구현됨 — 통합 여지.
4. **`legacy/brov_env.py` 실행 불가**: relative import 파손, 레거시 참조용.
5. **본 규모 학습 미실행**: `train.py`의 `num_envs=512`/`max_iterations=300`은 초기 추정값. 지금까지 검증에 쓴 체크포인트(`model_99.pt`)는 소규모 테스트 런이라 추종 품질이 논문과 비교할 수준이 아님.

---

## 다음 개발 순서

1. **학습 중 검증(validation) + wandb 연동** — `train.py`에 주기적 mid-training 평가와 로깅 추가. `step_1_NBV`의 방식(`train/train_GenNBV.py` 등)은 직접 만든 PPO 루프에 raw `wandb.init/log/watch/finish`를 손으로 꽂은 것이라 RSL-RL 기반인 `train.py`엔 그대로 못 옮김 — 대신 RSL-RL의 `RslRlOnPolicyRunnerCfg`가 자체 지원하는 `logger`/`wandb_project`/`wandb_entity` 필드(`agents/rsl_rl_ppo_cfg.py`의 `BROVVelPPORunnerCfg`에 추가)를 쓰는 게 올바른 경로. mid-training 평가 자체(주기적으로 정책을 멈추고 짧은 롤아웃 돌려 성능 확인)는 step_1의 `--eval_interval`/`--eval_episodes` CLI 플래그 네이밍과 "같은 env 인스턴스로 greedy rollout" 방식을 참고할 만함(`algorithm/algo_GenNBV.py`의 `run_eval()` 패턴). **아직 미구현.**
2. **학습 중 검증에 비디오 녹화 옵션 추가** — 위 mid-training 평가 시, `train.py`에 사용자가 지정하는 CLI 플래그(예: `--eval_record_video`)에 따라 `test_policy.py`의 영상 기록 방식(`_save_video`, 카메라 세팅)을 재사용해 mp4로 저장. step_1_NBV엔 선례가 없음(영상 저장은 `evaluate/evaluate_utils.py`의 `save_episode_video`뿐이고 학습 루프/wandb와는 완전히 분리돼 있음) — step_2_BROV에서 새로 설계해야 함. **아직 미구현.**
3. **mass 도메인 랜덤화(2단계)** — `root_physx_view.set_masses()` 기반 구현
4. **부족구동 어뢰형 AUV 지원** — MarineGym iAUV(`m600.py`+`fin200.py`+`underwaterVehicleFin.py`) 참고, `robots/dynamics/`에 새 액추에이터(주추진기+핀) 추가. Fossen 코어는 그대로 재사용
5. **step_3 통합** — step_1_NBV(수중 인지) + step_2_BROV(수중 물리), `LOSGuidance`를 NBV/커버리지 유도 모듈로 교체, 학습된 Sim2Swim 정책은 frozen 재사용

**본 규모 학습(`num_envs=2048` 등)은 사용자가 직접 진행 중** — 이 목록에서 별도 항목으로 추적하지 않음.

---

## 참조

- Fosso, Amundsen, Xanthidis, Ohrem. *"Sim2Swim: Zero-Shot Velocity Control for Agile AUV Maneuvering in 3 Minutes."* SINTEF Ocean. arXiv:2512.08656.
- von Benzon, Sørensen, Uth, Jouffroy, Liniger, Pedersen. *"An Open-Source Benchmark Simulator: Control of a BlueROV2 Underwater Robot."* JMSE 2022, 10, 1898. — hydro_coef 계수 및 F_max(Table 4) 출처.
- Fosso et al. 전신 논문 "Learning to Swim" — arXiv:2410.00120.
- Chu, Huang, Li, Lin, Li, Carlucho, Petillot, Yang. *"MarineGym: A High-Performance Reinforcement Learning Platform for Underwater Robotics."* IROS 2025. arXiv:2503.09203. — 부족구동 AUV(LAUV/iAUV) 참고 아키텍처.
- Fossen, T.I. (2011). *Handbook of Marine Craft Hydrodynamics and Motion Control.* Wiley.
- BlueRobotics T200 Thruster — [thrust data](https://bluerobotics.com/store/thrusters/t100-t200-thrusters/t200-thruster-r3-rp/)
- IsaacLab `DirectRLEnv` — `set_external_force_and_torque` body-frame 기준 적용
