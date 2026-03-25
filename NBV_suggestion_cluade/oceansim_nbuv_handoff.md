# OceanSim-NBUV 프로젝트 인수인계 문서

> Claude Code에서 이어서 작업하기 위한 컨텍스트 문서입니다.

---

## 1. 프로젝트 개요

### 목적
NBUV (Next Best Underwater View) 알고리즘을 OceanSim 시뮬레이터 안에서 구현하여, 추후 RL 기반 능동 인식 연구의 **baseline**으로 사용.

### 논문 기반

| 논문 | 역할 |
|------|------|
| **NBUV** — Sheinin & Schechner, CVPR 2016 | 핵심 알고리즘. 카메라+조명 pose 최적화로 albedo 추정 정보량 최대화 |
| **OceanSim** — Song et al., IROS 2025 (arXiv:2503.01074v2) | GPU 기반 수중 센서 시뮬레이터. Isaac Sim Extension. |
| **Cardaillac & Dansereau**, 2025 (arXiv:2504.17817) | 비교 대상. mesh-free, 고정 리그, 2DOF 방식. |

---

## 2. 시스템 구성

```
BlueROV2 (base, 동역학 무시 — Phase 1)
  ├── LeftArm  (UR5e) → UnderwaterCamera  (OceanSim)
  └── RightArm (UR5e) → UsdLux.SphereLight
```

- 카메라와 조명을 **독립적으로** 6DOF 제어 → 총 12DOF
- **Mesh prior**: USD geometry를 직접 coarsening (소나 파이프라인 대체)
  - 근거: NBUV 논문 Section 7 — "optimized views were largely insensitive to the coarsened topography prior"
- **후보 탐색**: 반구 샘플링 → IK 필터 → baseline 필터 → exhaustive argmax

### OceanSim의 정확한 역할

OceanSim은 scene 제공자가 아니라 **Isaac Sim 위의 센서 모델 레이어**입니다.

| 구성요소 | 역할 |
|----------|------|
| `UnderwaterCamera` | Eq.1 수중 감쇠 효과 post-process → `I_s(t)` 제공 |
| `ImagingSonar` | GPU ray tracing → point cloud (Phase 1 미사용) |
| `mhl_water.usd` | caustic + 수면 변형 (Warp kernel) |
| 물 파라미터 | `β_attn`, `β_bs`, `B∞` → NBUV medium params와 직접 대응 |

---

## 3. USD Stage 구조

```
/World
├── /World/Environment
│   ├── shipwreck.usd
│   └── mhl_water.usd         (OceanSim 제공)
├── /World/ROV                (BlueROV2, 동역학 무시)
│   ├── /LeftArm  (UR5e) → /tool0/Camera
│   └── /RightArm (UR5e) → /tool0/UnderwaterLight
└── /World/NBUVController     (oceansim_nbuv Extension)
```

---

## 4. 핵심 수식 구현 위치

| 수식 | 내용 | 파일 | 함수 |
|------|------|------|------|
| OceanSim Eq.1 | 수중 감쇠: `I_c = J·exp(-β_attn·d) + B∞·(1-exp(-β_bs·d))` | `oceansim_bridge.py` | `apply_water_column_effect` |
| NBUV Eq.2 | 조명 조도: `D_s ∝ C₀·exp(-β·l_LS)/l_LS²` | `nbuv_lighting.py` | `compute_D_s` |
| NBUV Eq.3 | 유효 조도: `E_s = D_s·cos(θ_LS)` | `nbuv_lighting.py` | `compute_E_s` |
| NBUV Eq.4 | Backscatter: `B_s = B∞·(1-exp(-β_bs·l_SC))` | `nbuv_lighting.py` | `compute_B_s` |
| NBUV Eq.7 | SNR: `√(ρE_s) / √(ρE_s+B_s+σ²_RN)` | `nbuv_lighting.py` | `compute_SNR_s` |
| NBUV Eq.12 | 단일 관측: `ρ̂_s = (I_s - B_s) / E_s` | `nbuv_estimator.py` | `estimate_rho_single` |
| NBUV Eq.17 | ML 추정: `ρ̂^ML = Σ(ρ̂·q) / Σq` | `nbuv_estimator.py` | `update_rho_ml` |
| NBUV Eq.19 | 정보량: `q_s = E_s² / (ρ̂·E_s + B_s + σ²_RN)` | `nbuv_estimator.py` | `compute_q_s` |
| NBUV Eq.20 | 누적: `Q^ML_s(t) = Σ q_s(t')` | `nbuv_estimator.py` | `update_Q_ML` |
| NBUV Eq.22 | 총 이득: `I_{t+1}(O) = Σ_s I_{t+1}(ρ̂^ML_s)` | `nbuv_info_gain.py` | `compute_total_info_gain` |
| NBUV Eq.25 | 패치별 이득: `(1/2)ln(1 + q_s(t+1)/Q^ML_s(t))` | `nbuv_info_gain.py` | `compute_patch_info_gain` |

---

## 5. 프로젝트 구조

```
oceansim_nbuv/
├── standalone_run.py          # headless 실행 진입점
├── extension.py               # Isaac Sim GUI Extension 진입점
│
├── config/
│   ├── water_params.json      # β_attn=0.12, β_bs=0.05, B∞=0.03
│   ├── nbuv_params.json       # σ_RN=0.01, C₀=1.0, prior_var=1/12
│   └── workspace_constraints.json  # baseline 0.02~0.34m, 반구 반경 0.3~1.5m
│
├── models/
│   ├── mesh_builder.py        # USD → coarse mesh → 패치 배열 + Q^ML 초기화
│   ├── patch_visibility.py    # 가시성 + l_SC, l_LS, cos 계산 (단일/배치)
│   ├── oceansim_bridge.py     # OceanSim API 래퍼 + MockSensor
│   ├── nbuv_lighting.py       # Eq.2~7 조명 모델
│   ├── nbuv_estimator.py      # Eq.12~20 albedo 추정 + 상태 업데이트
│   └── nbuv_info_gain.py      # Eq.22~25 정보이득 계산
│
├── planner/
│   ├── candidate_generator.py # 반구 샘플링 → IK 필터 → baseline 필터
│   └── nbuv_optimizer.py      # exhaustive argmax I_{t+1}(O)
│
└── controller/
    └── arm_controller.py      # UR5e IK → joint 이동 + MockArmController
```

---

## 6. 알고리즘 루프 (매 스텝)

```
[초기화]
  mesh_builder: USD geometry → coarse mesh → 패치 배열 {T_k}
  initialize_state: Q^ML[N] = 1/σ²_max (= 12.0),  ρ̂[N] = 0.5

[스텝 t]
  ① obs = sensor.get_observation()
       → RGB(t), Depth(t), patch_I_s(t)

  ② vis = compute_visibility(patches, cam_pos, light_pos)
       → visible_mask[N], l_SC[N], l_LS[N], cos_cam[N], cos_light[N]

  ③ E_s, B_s = compute_patch_irradiance(vis, water_params)   (Eq.2~4)

  ④ patches = step_estimator(patches, obs, E_s, B_s)
       → ρ̂_s (Eq.12) → q_s (Eq.19) → ρ̂^ML (Eq.17) → Q^ML (Eq.20)

  ⑤ cam_cands, light_cands = generate_candidates(target_center)
       → 반구 샘플링 → IK 필터 → baseline 필터

  ⑥ result = find_best_pose(patches, cam_cands, light_cands)
       → vis_batch (M,N) → E_s_batch (M,N) → q_s_next (M,N)
       → I_{t+1}(O) (M,) → argmax (Eq.22~25)

  ⑦ arm_ctrl.move_to_pose(result['best_cam_pos'], result['best_light_pos'])
```

---

## 7. 실행 방법

### Mock 모드 (Isaac Sim 없이 단독 테스트)

```bash
cd oceansim_nbuv
python standalone_run.py --mock --n_steps 20 --mesh_shape plane --verbose
```

옵션:

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--mock` | — | Isaac Sim 없이 실행 |
| `--n_steps` | 10 | 루프 스텝 수 |
| `--mesh_shape` | `plane` | `plane` \| `sphere` |
| `--mesh_n` | 15 | 격자 분할 수 (패치 수 ≈ 2*(n-1)²) |
| `--seed` | 42 | 재현성용 random seed |
| `--output_dir` | `results` | 결과 저장 경로 |
| `--verbose` | — | 각 스텝 상세 출력 |

### Isaac Sim headless 모드

```bash
./python.sh standalone_run.py \
    --scene /path/to/shipwreck.usd \
    --mesh_prim /World/shipwreck \
    --n_steps 20 \
    --voxel_size 0.05
```

### Isaac Sim GUI Extension

```
Isaac Sim → Window → Extensions → oceansim_nbuv 활성화
```

`extension.py`의 `on_startup()` → `on_update()` 루프 자동 실행.

---

## 8. 출력 파일

`results/` 디렉토리에 저장됩니다.

| 파일 | shape | 내용 |
|------|-------|------|
| `albedo_map.npy` | `(N,)` | 최종 ρ̂^ML (albedo 추정값) |
| `Q_ML_final.npy` | `(N,)` | 최종 누적 정보량 |
| `n_obs.npy` | `(N,)` | 패치별 관측 횟수 |
| `patch_centers.npy` | `(N, 3)` | 패치 중심 좌표 |
| `traj_cam.npy` | `(T, 3)` | 카메라 이동 경로 |
| `traj_light.npy` | `(T, 3)` | 조명 이동 경로 |
| `info_gain_log.npy` | `(T,)` | 스텝별 최적 정보이득 |

---

## 9. 설계 결정 사항

| 항목 | 결정 | 이유 |
|------|------|------|
| 내부 연산 | **NumPy 전체** | 코드 가독성 우선, Phase 2에서 선택적 PyTorch 전환 |
| Mesh prior | USD geometry 직접 coarsening | NBUV 논문 Section 7 — coarsening에 둔감함 확인 |
| 실제 소나 파이프라인 | Phase 1 미구현 | NBUV 알고리즘 자체에 집중 |
| IK 방식 | Isaac Sim `LulaKinematicsSolver` | Mock 모드에서는 all-valid fallback |
| Baseline 범위 | 2cm ~ 34cm | NBUV 논문 실험 범위 |
| 후보 수 | 카메라 20 × 조명 10 = 200 | 계산 비용과 커버리지 균형 |
| BlueROV2 동역학 | **무시** | Phase 1 scope 외 |

---

## 10. 연구 로드맵

```
Phase 1 (현재 완료):
  NBUV 구현 → baseline 확보
  NumPy 전체, Mock + Isaac Sim 양방향 지원

Phase 2 (예정):
  RL policy가 exhaustive search를 대체
  → Isaac Lab / OpenAI Gym wrapper 연결
  → Observation: RGB, Depth, Q^ML
  → Action: UR5e joint delta (12-DOF)
  → Reward: 미결정 (mesh 기반 vs image quality ablation 예정)
  → Phase 1 trajectory → Behavioral Cloning 사전학습
```

### Ablation 변수

| 변수 | NBUV (본 구현) | Cardaillac 2025 |
|------|---------------|-----------------|
| Mesh 사용 | O | X |
| 독립 리그 | O (12DOF) | X (고정, 2DOF) |
| 탐색 방식 | Exhaustive | Gradient |
| RL 전환 | Phase 2 예정 | — |

---

## 11. 의존성

```
# 필수 (Mock 모드)
numpy

# Mesh coarsening
open3d

# Isaac Sim 모드
isaacsim          # NVIDIA Isaac Sim 설치 필요
omni.usd
omni.replicator.core
omni.isaac.core
pxr               # USD Python bindings
```

---

## 12. 알려진 제한 및 TODO

- [ ] **Occlusion 처리**: 현재 법선 기반 가시성만 사용. Isaac Sim GPU ray cast로 occluded patch 처리 필요
- [ ] **실제 소나 파이프라인**: `ImagingSonar` → point cloud → mesh reconstruction 연결
- [ ] **BlueROV2 동역학**: Phase 2에서 hydrodynamics 적용
- [ ] **Reward 함수 설계**: Phase 2 RL 전환 시 결정 필요
- [ ] **결과 시각화**: albedo map, trajectory, info gain 곡선 시각화 스크립트
- [ ] **IK 솔버 연동**: 실제 UR5e joint limit 반영한 reachability 검증

---

## 13. Mock 모드 검증 결과

```
패치 수: 162 (plane mesh, n=10)
스텝 수: 5

Step  1/5  gain=10.92285  observed=162/162  mean_rho=0.4999
Step  2/5  gain= 9.17250  observed=162/162  mean_rho=0.4822
Step  3/5  gain= 9.33871  observed=162/162  mean_rho=0.4715
Step  4/5  gain= 8.45563  observed=162/162  mean_rho=0.4595
Step  5/5  gain= 7.02419  observed=162/162  mean_rho=0.4482
```

정보이득 감소 추세 (관측 누적에 따른 불확실성 감소) — 정상 거동 확인.
