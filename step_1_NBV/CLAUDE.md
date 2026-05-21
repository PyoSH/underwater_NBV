# OceanRL NBV — 프로젝트 전체 현황 (2026-05-21)

---

## 1. 연구 개요

**수중 환경에서의 강화학습 기반 Next-Best-View (NBV) 계획**

- 수중 로봇이 카메라를 이동하며 미지의 3D 물체(암석 등)를 최적 시점에서 관측
- Isaac Sim + IsaacLab 기반 시뮬레이터에서 PPO 학습
- 관측 품질(Beer-Lambert 감쇠 기반 quality metric) 극대화가 목표

**학회 발표 목적**: Quality-Aware NBV를 수중 환경에 적용한 RL 프레임워크 제안

---

## 2. 환경 구조

| 항목 | 값 |
|------|-----|
| 시뮬레이터 | Isaac Sim 5.1 + IsaacLab |
| 컨테이너 | `docker exec env_pyoten` |
| 호스트 프로젝트 | `/workspace/pyoten/Programing/OceanRL_test/step_1_NBV/` |
| 컨테이너 프로젝트 | `/workspace/Programing/OceanRL_test/step_1_NBV/` |
| 체크포인트 | `/workspace/pyoten/checkpoints/` (host) = `/workspace/checkpoints/` (container) |
| 학습 실행 | `DISPLAY=:99 /workspace/isaac-sim/python.sh train_GenNBV_quality.py` |
| 두 Isaac Sim 동시 실행 | **불가** (GPU deadlock) |

---

## 3. 주요 파일

| 파일 | 역할 |
|------|------|
| `train_GenNBV_quality.py` | **현재 메인 학습 스크립트** (PPO + quality metric) |
| `env_GenNBV_quality.py` | Quality-aware 환경 (Beer-Lambert, TSDF quality 누적) |
| `env_GenNBV.py` | Binary coverage 환경 (부모 클래스) |
| `env_reward.py` | TSDF 적분, binary coverage 계산 |
| `envCfg.py` | 환경 설정 및 보상 가중치 |
| `algorithm2.py` | PPO Actor/Critic 네트워크 (CNN + LSTM) |
| `train_RL.py` | 구버전 PPO (binary coverage) |
| `train_scanRL.py` | DQN 학습 (사용 보류) |

---

## 4. 관측·행동 공간

| 항목 | 크기 | 내용 |
|------|------|------|
| Actor 관측 | (6, 84, 84) | 과거 6 프레임 gray image sequence |
| Critic 관측 | (6, 84, 84) | 과거 6 프레임 depth map sequence |
| Scalar 관측 | 3 | 구면좌표 (azimuth θ, elevation φ, distance ψ) |
| Voxel 관측 | (3, 40, 40, 40) | ch0=unknown / ch1=free / ch2=quality |
| 행동 | 6 (discrete) | (θ±, φ±, ψ±) 이산 이동 |

**Voxel 관측 채널**:
- ch0: `weight == 0` → 미관측
- ch1: `weight > 0 & tsdf > 0` → free space로 관측됨
- ch2: `quality_vol / Q_sat` clamp(0,1) → quality 누적값 (연속)

---

## 5. Quality Metric 설계

### Beer-Lambert 단방향 감쇠

```
q(v) = exp(-μ × d)
  μ : 수중 감쇠계수 (에피소드별 DR, 채널 평균)
  d : cam_pos ↔ voxel 중심 거리
```

OceanSim UWrenderer가 `exp(-μd)` post-process를 **1회만** 적용 → 단방향 감쇠.

### 누적 방식: Fisher 정보 합산 (NBUV CVPR 2016 Eq.20)

```
Q_vol(v) += q(v)   when surface_mask
```

### 포화 임계값

```
Q_sat = exp(-μ × psi_min) ≈ 0.80
  psi_min = 1.0m에서 단일 방문 시 포화
soft_coverage(v) = clip(Q_vol(v) / Q_sat, 0, 1)
coverage_q = mean over GT surface voxels of soft_coverage(v)
```

### 수중 DR 범위 (WaterParamRangeCfg)

| 파라미터 | min | max |
|----------|-----|-----|
| atten_coeff R | 0.03 | 0.15 |
| atten_coeff G | 0.03 | 0.10 |
| atten_coeff B | 0.10 | 0.40 |
| μ_mean (채널평균) | ~0.053 | ~0.217 |

`water_dr_enabled = False` (현재 DR 비활성 → μ = 고정값 0.217)

---

## 6. 환경 파라미터 (envCfg.py 현재값)

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| vol_dim | (40,40,40) | TSDF 볼륨 |
| voxel_size | 0.05m | 5cm |
| trunc_margin | 0.05m | voxel_size와 동일 유지 필수 |
| episode_length_s | 0.8333s | 50 steps |
| psi_min / psi_max | 1.0 / 4.5m | 카메라 거리 범위 |
| delta_theta/phi | 15° | 스텝당 각도 이동 |
| delta_psi | 0.20m | 스텝당 거리 이동 |
| k_c | 5.0 | binary coverage reward (현재 미사용) |
| **k_c_q** | **5.0** | quality coverage reward 가중치 |
| k_x | 0.02 | 이동 거리 패널티 |
| **c_step** | **0.02** | 스텝당 고정 패널티 |
| k_still | 0.05 | stall 패널티 |
| stall_thr | 1e-4 | coverage 증가 최소 임계값 |
| **coverage_terminal** | **0.52** | 성공 판정 임계값 (0.82→0.52로 하향) |
| coverage_bonus | 30.0 | 성공 보너스 |
| k_explore | 0.0 | exploration bonus (비활성) |

---

## 7. 보상 함수 (env_GenNBV_quality.py)

```
reward = k_c_q × Δcoverage_q
       - k_x × dist_moved
       - c_step
       - k_still × (Δcoverage_q < stall_thr)
       + coverage_bonus × (coverage_q ≥ coverage_terminal)
```

- **k_x > 0 금지**: 이동 패널티 → "접근 후 대기" 병폐 발생. 현재 0.02로 소폭 적용 중.
- **c_step ≪ k_c_q × Δcov_max**: c_step이 너무 크면 에이전트가 움직이지 않음.

---

## 8. 현재 학습 실행 (UW_NBV_1)

```bash
docker exec -d env_pyoten bash -c "
cd /workspace/Programing/OceanRL_test/step_1_NBV && \
DISPLAY=:99 /workspace/isaac-sim/python.sh train_GenNBV_quality.py \
    --num_envs 4 \
    --total_steps 500000 \
    --save_interval 10 \
    --wandb_name UW_NBV_1 \
> /workspace/logs/train_UW_NBV_1.log 2>&1"
```

- wandb 프로젝트: `RL_NBV` / 런: `UW_NBV_1`
- 최신 체크포인트: `genNBV_quality_step_0000450560.pt`
- 학습 중단 후 재시작 시 `--resume /workspace/checkpoints/UW_NBV_1/genNBV_quality_step_XXXXXXXX.pt`

---

## 9. 로그 포맷

```
[  460800]  rew=+0.320  pl=-0.004  vl=53.7  ent=1.475  ev=0.238  cov_q=0.483  cov_bin=0.857  fps=631
  [reward]  cov_q=+0.052  penalty=0.020  success=+0.293  stall=0.005  net=+0.321
  [coverage]  binary=0.703  quality=0.411  weight_filled=0.054  surf_voxels=412
  [quality]  vox_quality_mean=0.990  mu=0.217
  [diag/GT]  never=0.515  partial=0.006  full=0.479
```

| 지표 | 의미 |
|------|------|
| cov_q | 에피소드 종료 시 quality-weighted coverage 평균 |
| cov_bin | 에피소드 종료 시 binary coverage 평균 |
| binary | 현 배치의 binary coverage 평균 |
| quality | 현 배치의 quality coverage 평균 |
| weight_filled | 전체 voxel 중 weight>0 비율 |
| surf_voxels | GT surface voxel 수 |
| vox_quality_mean | TSDF surface 전체 voxel의 평균 quality (GT 외 포함, 참고용) |
| diag/GT never | GT surface 중 quality=0 비율 |
| diag/GT full | GT surface 중 quality≥1(포화) 비율 |

---

## 10. 핵심 발견: coverage_q 정체 원인

### 관찰된 수치 (step 460800)

| 지표 | 값 |
|------|----|
| binary coverage | 0.857 |
| coverage_q | 0.483 |
| diag/GT never | 0.515 |
| diag/GT full | 0.479 |

### 근본 원인: TSDF 분류 불일치

```python
# _compute_quality()의 현재 조건
surface_mask = (self._weight_vol > 0) & (self._tsdf_vol <= 0)
```

- **binary coverage**: `weight > 0` 만 요구 (카메라 레이가 근처를 지나간 voxel)
- **quality 누적**: `weight > 0` AND `tsdf ≤ 0` 모두 요구 (TSDF가 surface로 확정한 voxel만)

**메커니즘**:

TSDF 부호 규약: `sdf = depth_hit - depth_voxel`
- `sdf > 0` → voxel 중심보다 깊은 곳에 표면 → **free space로 분류**
- `sdf ≤ 0` → voxel 중심이 표면 뒤쪽 → **surface/occupied로 분류**

GT surface voxel은 5cm voxel 안에 mesh가 들어있다. 카메라 레이가 voxel 중심을 통과한 뒤 mesh를 만나면 `sdf > 0` → TSDF가 해당 voxel을 **free space**로 분류.

결과: GT surface voxel의 37%가 `weight > 0` (관측됨)이지만 `tsdf > 0` (free space 판정) → quality 누적 차단.

```
binary observed:      85.7%
quality > 0:          48.5%
-------
observed but quality=0: 37.2%  ← TSDF 오분류
```

### 제안 수정

```python
# 수정 전
surface_mask = (self._weight_vol > 0) & (self._tsdf_vol <= 0)

# 수정 후
surface_mask = self._weight_vol > 0
```

tsdf 조건 제거 시 coverage_q 상한 ≈ binary_coverage (~0.85). quality는 "얼마나 가까이서 관측했는가"를 측정하며 TSDF 확정 여부와 무관하다.

---

## 11. 진단 코드 (버그 수정 완료)

`env_GenNBV_quality.py::_reset_idx()` — super() 호출 **이전**에 진단 실행:

```python
def _reset_idx(self, env_ids) -> None:
    # super() 이전에 실행해야 _surf_vol이 현 episode 것
    for eid in env_ids:
        gt_mask = self._surf_vol[eid]
        if gt_mask.any():
            q_soft = (self._quality_vol[eid] / self._quality_Q_sat).clamp(0.0, 1.0)
            gt_q   = q_soft[gt_mask]
            self._diag_gt_never[eid]   = (gt_q == 0.0).float().mean()
            self._diag_gt_partial[eid] = ((gt_q > 0.0) & (gt_q < 1.0)).float().mean()
            self._diag_gt_full[eid]    = (gt_q >= 1.0).float().mean()

    self._terminal_coverage_q[env_ids] = self.curr_coverage_q[env_ids]
    super()._reset_idx(env_ids)  # ← 여기서 _surf_vol이 새 episode로 교체됨
    self._quality_vol[env_ids]     = 0.0
    self.curr_coverage_q[env_ids]  = 0.0
    self._prev_coverage_q[env_ids] = 0.0
```

**버그 이력**: 이전 코드는 super() 호출 후 진단 → `_surf_vol`이 새 episode GT로 교체된 뒤 old `quality_vol`과 비교 → never=0.791, full=0.207로 coverage_q=0.464와 모순 (기댓값 0.208 vs 실제 0.464). 수정 후 일치 확인.

---

## 12. 학습 히스토리

| 런명 | 내용 | 결과 |
|------|------|------|
| genNBV | Binary coverage PPO | coverage_bin ~0.85 도달 |
| genNBV_quality | Quality metric 도입 (coverage_terminal=0.82) | coverage_q ~0.47에서 정체, success 0% |
| UW_NBV_1 | coverage_terminal=0.52로 하향, c_step=0.02 | coverage_q ~0.48, success ~15%, length 소폭 감소 |
| UW_NBV_diag | 진단 run (버그 있음) | never=0.791 → coverage_q와 불일치, 무효 |
| UW_NBV_diag2 | 진단 run (버그 수정) | never=0.515, full=0.479 → coverage_q=0.483 일치 ✓ |

---

## 13. 알려진 주의사항

- **k_x > 0**: 이동 패널티는 "접근 후 대기" 행동 유도 가능. 소폭(0.02)만 허용.
- **c_step vs reward scale**: c_step ≪ k_c_q × max_delta_coverage 유지 (현재 0.02 vs 5×0.05≈0.25)
- **save_interval**: `num_envs × save_interval × episode_length` → 체크포인트 저장 간격. `--save_interval 10` 권장.
- **trunc_margin = voxel_size**: trunc_margin > voxel_size이면 coverage 과대 계상.
- **Isaac Sim 동시 실행 불가**: 두 학습 프로세스 동시 기동 시 GPU deadlock.
- **python 실행**: `python` 아닌 `/workspace/isaac-sim/python.sh` 사용.
- **coverage_terminal 설정 근거**: wandb 분포에서 max window mean ≈ 0.507 → 0.52로 설정.
- **Q_sat 설계 불일치**: Q_sat=0.80은 고정값이나 μ가 DR로 변함 (0.053~0.217). water_dr_enabled=False이면 무관.

---

## 14. 다음 수행 사항

1. **`_compute_quality()` tsdf 조건 제거**: `surface_mask = self._weight_vol > 0`
2. 수정 후 학습 재시작 → coverage_q가 binary coverage 수준(~0.85)까지 상승하는지 확인
3. coverage_q 상승 시 coverage_terminal 재조정 검토 (0.52 → 0.65~0.75)
4. length 감소 확인 (coverage_q 상승 → success 빈도 증가 → 에피소드 단축)

---

## 15. 학습 모니터링

```bash
# 실시간 로그
docker exec env_pyoten tail -f /workspace/logs/train_UW_NBV_1.log

# 최신 체크포인트
ls -lt /workspace/pyoten/checkpoints/UW_NBV_1/

# 프로세스 확인
docker exec env_pyoten pgrep -a python.sh
```
