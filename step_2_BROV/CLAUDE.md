# step_2_BROV — BlueROV2 수중 궤적 추종 RL 환경

## 프로젝트 목표

**현재 단계: 수중 동역학 · 유체역학 환경 구현 선행**

BlueROV2 Heavy (8-thruster)를 IsaacLab으로 시뮬레이션하는 RL 환경.
RL 학습 연결 이전에 물리 시뮬레이션의 정확도를 먼저 확보하는 것이 목표다.

1. 수중 동역학 / 유체역학이 실제 BROV2 거동과 일치하는지 검증
2. 추진기 모델(PWM→추력) 정확도 확인
3. 중성 부력 / 자세 안정성 확보
4. 이후 RL 학습 런처(RSL-RL 등) 연동

---

## 파일 구조

```
step_2_BROV/
├── CLAUDE.md           ← 이 파일
├── train.py            # 물리 검증 테스트 런처 (RL 학습 아님)
├── envCfg.py           # BROVTrajEnvCfg — 전체 파라미터 관리
├── sceneCfg.py         # BROVSceneCfg — IsaacLab 씬 정의
├── env.py              # BROVTrajEnv — 메인 RL 환경 (DirectRLEnv 서브클래스)
├── hydrodynamics.py    # BROV2ThrusterModel + BROV2Hydrodynamics
├── bottom_up.py        # 물리 검증 테스트 함수 3종
└── brov_env.py         # [레거시] MIT CSAIL origin, 실행 불가 — 참조 전용

공유 에셋 (상위 디렉토리):
../../robots/assets/brov_joint.py   # ArticulationCfg, BROV_CFG
../../robots/data/BROV2/
  ├── BlueROV2_buoyancy.usd         # 실제 사용 USD 에셋
  └── config.yaml                   # URDF→USD 변환 설정
```

---

## 아키텍처 개요

### 의존 관계

```
train.py
  ├── envCfg.py  →  BROVTrajEnvCfg
  │     └── sceneCfg.py  →  BROVSceneCfg
  │           └── robots/assets/brov_joint.py  →  BROV_CFG (ArticulationCfg)
  ├── env.py  →  BROVTrajEnv
  │     └── hydrodynamics.py  →  BROV2ThrusterModel, BROV2Hydrodynamics
  └── bottom_up.py  →  test_neutral_buoyancy / test_straight_line / test_thruster_model
```

### 물리 루프 (매 policy step)

```
_pre_physics_step   →  action clamp [-1, 1]
  (decimation=4 반복)
    _apply_action
      BROV2ThrusterModel.compute(pwm)   →  f_thrust, t_thrust  (body frame)
      BROV2Hydrodynamics.compute(...)   →  f_hydro,  t_hydro   (body frame)
      set_external_force_and_torque(body_ids=[0])
_get_observations   →  17-dim obs
_get_rewards        →  scalar reward
_get_dones          →  terminated / truncated
```

---

## 수중 동역학 구현 (`hydrodynamics.py`)

### BROV2ThrusterModel

BlueRobotics T200 기반 8-thruster 모델.

**변환 파이프라인:** PWM [-1,1] → 1차 지연 필터 (τ=0.05 s) → RPM 다항식 → 추력 다항식 → body-frame 합력/토크

```
데드밴드 : |pwm| < 0.075 → 추력 = 0
최대 RPM : ±3900
```

**추진기 배치 (body frame: X=전방, Y=좌방, Z=상방)**

| ID | 위치 | 방향 | 역할 |
|----|------|------|------|
| T1 | 전우 | +45° XY | surge/sway/yaw |
| T2 | 전좌 | -45° XY | surge/sway/yaw |
| T3 | 후우 | 225° XY | surge/sway/yaw |
| T4 | 후좌 | 135° XY | surge/sway/yaw |
| T5~T8 | 전우/전좌/후우/후좌 | -Z (하방) | heave/roll/pitch |

> **주의:** 위치·방향 값은 근사치. `BlueROV2_buoyancy.usd` 확인 후 수정 필요.

**"up" 명령:** T5~T8에 -1 → (-1)×(-Z) = +Z body → 상승. 일관성 유지.

### BROV2Hydrodynamics

Fossen (2011) 기반 6-DOF 유체역학. **모든 힘/토크는 body frame.**

```
compute() 반환: forces_b - (f_damping + f_added_mass + f_coriolis)
               torques_b - (t_damping + t_added_mass + t_coriolis)
부력은 복원력(+), 나머지는 운동 저항(-)
```

**현재 유체역학 계수 (튜닝된 값):**
```python
_ADDED_MASS        = [6.36,  7.12,  18.68, 0.189, 0.135, 0.222]
_LINEAR_DAMPING    = [13.70, 0.00,  33.00, 0.00,  0.80,  0.00 ]
_QUADRATIC_DAMPING = [141.0, 217.0, 190.0, 1.19,  0.47,  1.50 ]
```

참고값 (BlueROV.yaml 원본):
```python
# [5.5, 12.7, 14.57, 0.12, 0.12, 0.12]   # added mass
# [4.03, 6.22, 5.18, 0.07, 0.07, 0.07]   # linear damping
# [18.18, 21.66, 36.99, 1.55, 1.55, 1.55] # quadratic damping
```

---

## MDP 정의

### 관측 (17차원)
| 인덱스 | 량 | 프레임 |
|--------|-----|--------|
| 0-2 | pos_env (env origin 기준 XYZ) | env-local |
| 3-6 | quat [w, x, y, z] | world |
| 7-9 | lin_vel_b | body |
| 10-12 | ang_vel_b | body |
| 13-15 | wp_dir_b (다음 waypoint 방향 단위벡터) | body |
| 16 | wp_dist (거리 m) | — |

### 행동 (8차원, [-1, 1])
T1~T8 PWM 명령. policy 25 Hz.

### 보상
```
R = rew_progress + rew_waypoint + rew_action + rew_upright
```
| 성분 | 수식 | 스케일 |
|------|-----|--------|
| progress | prev_dist - curr_dist | 1.0 |
| waypoint | 1.0 (dist < 0.5 m) | 10.0 |
| action | -‖actions‖ | 0.05 |
| upright | body_Z · world_Z | 0.3 |

### 종료
- `terminated`: env origin 기준 |x|>12 or |y|>12 or |z|>10 m
- `truncated`: 60 s 경과

---

## 주요 설정 파라미터 (`envCfg.py`)

| 파라미터 | 값 | 비고 |
|---------|-----|------|
| `sim.dt` | 1/100 s | 100 Hz 물리 |
| `decimation` | 4 | 정책 25 Hz |
| `episode_length_s` | 60.0 s | |
| `volume` | 0.022747843 m³ | 부력 계산 — 중성부력 조정 시 변경 |
| `cob_offset` | 0.01 m | COB가 COM보다 +Z 방향으로 위에 있는 거리 |
| `water_density` | 997.0 kg/m³ | |
| `trajectory_type` | "circle" | "helix" 옵션 |
| `num_waypoints` | 12 | |
| `trajectory_radius` | 3.0 m | |

---

## 물리 검증 워크플로우

### 실행 방법
```bash
# 중성 부력 확인 (추력=0, Z 드리프트 측정)
python train.py --test neutral_buoyancy --duration 10.0 [--headless]

# 방향 이동 확인 (전진/우측/상승)
python train.py --test straight_line --thrust 0.5 --duration 3.0

# 추진기 PWM→추력 변환표 출력
python train.py --test thruster_model
```

### 검증 기준
- **중성부력**: |ΔZ| < 0.1 m / 10 s
  - 하강 시: `envCfg.volume` 증가
  - 상승 시: `envCfg.volume` 감소
- **직선이동**: 주축 변위 > 0.05 m, 횡방향 표류 < 주축 변위 × 0.5
- **추진기**: PWM=1.0에서 약 30~40 N 예상

---

## 알려진 이슈

1. **`rew_scale_terminated` 미적용**: `envCfg.py`에 정의되어 있으나 `env.py._get_rewards()`에서 사용되지 않음.

2. **`hydrodynamics.py` 클래스 기본값 불일치**: `_VOLUME = 0.0134` (클래스 기본값) ≠ `envCfg.volume = 0.022747843`. 런타임에는 `cfg.volume`이 생성자에 전달되므로 정상 동작. 하지만 `BROV2Hydrodynamics`를 직접 인수 없이 생성하면 오류.

3. **추진기 위치/방향 미검증**: `hydrodynamics.py`의 `_POS`, `_DIR`은 근사치. `BlueROV2_buoyancy.usd` USD 에셋에서 실제 joint 위치 확인 필요.

4. **`brov_env.py` 실행 불가**: `.assets.brov`, `.rigid_body_hydrodynamics`, `.BROV_thruster_dynamics` relative import 파손. 레거시 참조용으로만 유지.

5. **RL 학습 런처 미존재**: RSL-RL 등 외부 RL 라이브러리 연동이 필요하며 현재 단계에서는 미구현.

---

## 다음 개발 순서

1. **USD 에셋 확인** — `BlueROV2_buoyancy.usd`에서 실제 thruster joint 위치·방향 추출 후 `_POS`, `_DIR` 수정
2. **중성부력 튜닝** — `test_neutral_buoyancy`로 `envCfg.volume` 조정
3. **직선이동 방향 검증** — `test_straight_line`으로 T1~T8 방향 부호 확인
4. **유체역학 계수 튜닝** — 실제 BROV2 데이터 또는 MarineGym 원본값과 비교
5. **RL 런처 연동** — RSL-RL `runner.py` 추가 및 학습 설정 YAML 작성

---

## 참조

- [MarineGym](https://github.com/srl-ethz/MarineGym) — 동역학 모델 원본 (`underwaterVehicle.py`, `t200.py`)
- Fossen, T.I. (2011). *Handbook of Marine Craft Hydrodynamics and Motion Control.* Wiley.
- BlueRobotics T200 Thruster — [thrust data](https://bluerobotics.com/store/thrusters/t100-t200-thrusters/t200-thruster-r3-rp/)
- IsaacLab `DirectRLEnv` — `set_external_force_and_torque` body-frame 기준 적용
