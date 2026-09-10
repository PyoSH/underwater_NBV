# 서버 인계 — 카메라 (나) 재학습 (2026-09-10)

받는 쪽: 서버(kriso, 컨테이너 `env_pyoten`)의 Claude. 이 문서만 읽고 시작할 수 있게 쓴다.
정본은 `DEPLOY_3WEEK_PLAN.md` §15 — 여기는 **실행 순서와 명령**만 있다.

## 0. 한 줄 요약

학습 카메라를 실기와 완전히 일치시켰다(640x480, fx 465.518, HFOV 69.0). 깊이 버그와
물체 접지도 함께 고쳤다. 로컬 RTX 4080(16 GB)에서 렌더 비용·보상 눈금·베이스라인을
실측해 판정을 끝냈다. **서버에서 할 일은 (1) env 수 확정 (2) 베이스라인 재측정 (3) 재학습**
이다. 보상 파라미터는 **건드리지 말 것** (§4).

## 1. 무엇이 바뀌었나 (커밋에 포함됨)

| 파일 | 변경 | 근거 |
|---|---|---|
| `envs/scene_cfg.py:211-215` | 카메라 320x240 / aperture 20.955 → **640x480 / 32.9955** (fx 366.5 → 465.518) | 실기 수중 캘리브 `brov_ros2 runtime/calibration/camera_intrinsics.yaml`, 돔 포트. crop 은 fx 를 못 바꾸므로 카메라 자체를 맞춤 (§15.3~15.5) |
| `envs/scene_cfg.py:144` | 물체 z **-3.0 → -3.125** | seafloor 상면에 접지. 실기/Gazebo 와 일치 (§12.0) |
| `envs/env.py:173` | `rock_local` z **-3.0 → -3.125** | 구면 중심 = 물체 원점. 위와 **항상 같이** 움직여야 함 |
| `envs/tsdf_fusion.py` (신설) + `envs/env_reward.py` | 융합을 순수 함수로 분리, **유클리드→z-depth 변환** 추가 | 융합식 `sdf = d - vox_z` 는 z-depth 를 요구하는데 `distance_to_camera` (유클리드) 를 넣고 있었다. 모서리에서 2.37 voxel 오차 (§15.2). UW 렌더는 유클리드가 맞으므로 융합 직전에만 변환 |
| `envs/nbv_baselines.py` (신설) + `eval_core.py` | 베이스라인 정책 분리 | 배포 루프와 동일 코드 사용 |
| `envs/env_cfg.py:437` | `nbuv_px_per_voxel_edge` **21.0 유지** (주석만 갱신) | §4 참조 — 바꾸지 말 것 |
| `tools/measure_render_scaling.py` (신설) | 해상도/env 스케일링 계측 | §2 |
| `tools/measure_view_tradeoff.py` | `--resolution/--fx` override + 렌더 건전성 관문 | 신/구 카메라 대조용 |

수치 동일성: `deploy/test_tsdf_fusion.py` 20/20 — 분리 전 원본 구현을 참조로 5 시드에서
`torch.equal`. (로컬 ROS 컨테이너에서 실행; 서버에서는 불필요)

## 2. 로컬 실측 — 서버가 다시 잴 필요 없는 것

### 2.1 렌더 스케일링 (RTX 4080 SUPER 16 GB, TiledCamera)

| env | 해상도 | 결정당 | VRAM | 죽은 카메라 |
|---|---|---|---|---|
| 16 | 320x240 | 1.309 s | 5583 MiB | 0 |
| 16 | 640x480 | 1.368 s (1.05x) | 6607 MiB | 0 |
| 32 | 320x240 | 1.597 s | 6183 MiB | 0 |
| 32 | 640x480 | 1.806 s (1.13x) | 8500 MiB | 0 |

화소 4 배에 시간 5~13% — 병목은 물리(decimation 500)다. **비용은 VRAM: env 당 +70 MiB.**

```
640x480  VRAM ≈ 3.5 GB + 0.155 GB/env      (두 점 선형 적합)
```

→ **서버 VRAM 을 이 식에 넣어 env 수를 정할 것.** 96 env 유지에 약 18 GB.
env 수 = 등장 물체 수(§3)이므로 VRAM 이 허락하는 최대로. 96 이 되면 96.

### 2.2 보상 눈금 판정 (`measure_view_tradeoff`, GSO 24종, nbuv, px_per_voxel 21)

| ψ | 구 카메라 손익 | 신 카메라 손익 |
|---|---|---|
| 1.6 | **2.21** (최적) | 1.71 |
| 2.0 | 1.56 | **1.89** (최적) |
| 2.5 | 0.74 | 1.28 |

신 카메라에서 상자 [1.4, 2.0] 안의 품질비가 전부 ≥ 0.96 이라 거리는 품질에 무관해졌다.
이것은 센서가 좋아진 물리적 사실이지 고칠 문제가 아니다 (§15.7).

### 2.3 석고틀 베이스라인 (Gazebo GT depth, 실기 화각, cov_bin, 3 시드)

```
ceiling (sweep@100)   0.964
정책      @10    @15    @20    @25    @30    @39
orbit   0.534  0.590  0.605  0.619  0.637  0.652   ← 68%, @20 이후 포화
random  0.510  0.586  0.675  0.754  0.802  0.861   ← 89%, 상승 중
random − orbit @39 = +0.21   (GSO 0.25, 구 카메라 석고틀 0.35 와 같은 자릿수)
```

random 과 orbit 의 ψ 평균이 같다(1.73 vs 1.75) → random 의 이점은 **거리가 아니라
φ·θ 다양성**이다. 과업은 성립하고, RL 이 배울 것은 시점 다양성이다.

## 3. 서버에서 할 것 — 순서대로

### 3.0 동기화
코드는 커밋. 자산은 gitignored 라 rsync:
```
robots/data/real_object/            (석고틀 OBJ + USD)
robots/data/gso_usd/                (이미 있으면 생략)
step_3_NBV_BROV/eval_out/real_object_sweep_phi10_80/observable_mask.npy
```

### 3.1 스모크 — 4 env x 3 리셋, 다중 메쉬
```
/isaac-sim/python.sh -u smoke_test_stage1.py --headless --enable_cameras --num_envs 4 \
  --mesh_pool <gso_usd>/manifest.json --n_resets 3
```
통과 기준: env 마다 다른 자산, 죽은 카메라 0, 보상 유한, GT surf voxel 230~621.
**렌더 건전성 관문이 이 스크립트에 있다** — "죽은 카메라" 가 나오면 그 env 수는 못 쓴다.

### 3.2 env 수 확정
```
/isaac-sim/python.sh -u tools/measure_render_scaling.py --headless --enable_cameras \
  --num_envs {64,96,128} --resolution 640x480 --decisions 4 --out /tmp/scale.json
```
죽은 카메라 0 인 최대 env 수. VRAM 여유 10% 남길 것.

### 3.3 베이스라인 재측정 (카메라·깊이·접지가 바뀌었으므로 전부 새로)

GSO ceiling:
```
evaluate_nbv.py --headless --enable_cameras --policies random,orbit --num_envs <N> \
  --num_episodes 32 --max_decisions 40 --mesh_pool <gso>/manifest.json --camera_path tiled
```
실물체 ceiling (φ[10,80] 과 φ[30,60] 둘 다):
```
evaluate_nbv.py --headless --enable_cameras --policies random,orbit,approach --num_envs <N> \
  --num_episodes 32 --max_decisions 40 --mesh_pool robots/data/real_object/usd/manifest.json \
  --mesh_pool_split all --no_require_texture --fixed_object_pose \
  --psi_min 1.4 --psi_max 2.0 --phi_min_deg 10 --phi_max_deg 80 --camera_path tiled
```
observable_mask (sweep 100 결정) → `eval_out/.../observable_mask.npy` 갱신.

**대조 기준**: 로컬 Gazebo cov_bin 은 random@39 0.861 / orbit 0.652. Isaac cov_q 는 이보다
낮게 나오는 것이 정상(품질 가중). 자릿수가 다르면 원인을 찾을 것.

### 3.4 학습
```
/isaac-sim/python.sh -u train.py --headless --enable_cameras --num_envs <N> \
  --mesh_pool <gso>/manifest.json --quality_model nbuv --normal_source tsdf \
  --wandb_name <이름> ...   (PPO 인자는 CLI 로 — 코드 수정 금지)
```
수질 DR 은 **off** (§9-3 확정). 실물체는 학습에 넣지 않는다(홀드아웃).

### 3.5 판정
같은 결정 수에서 학습 정책 vs random. 계획서 §11.6: "동일 결정 수 random 대비 비교가
정직한 지표". `approach` 가 orbit 을 넘으면 정규화가 새는 것(상설 회귀 지표).

## 4. 하지 말 것

- **`nbuv_px_per_voxel_edge` 를 21 에서 바꾸지 말 것.** 26.6 으로 올리려던 분석이 local
  minimum 이었다(§15.7). 상자 안에서 거리 압력이 없는 것은 센서 개선의 결과이고, 정책이
  배울 것은 거리가 아니라 다양성이다.
- crop 을 넣지 말 것. (나) 는 실기 화각 그대로 학습한다.
- `distance_to_camera` 를 `distance_to_image_plane` 으로 바꾸지 말 것 — UW 렌더가 유클리드를
  정당하게 쓴다. 변환은 `env_reward._fuse_depth` 안에서만 한다.
- 물체 z 와 `rock_local` 을 따로 바꾸지 말 것.

## 5. 알려진 함정

- 렌더러는 자원 할당 실패를 **크래시 없이** 넘긴다. 죽은 카메라 관문을 믿을 것.
- `python.sh` 를 Ctrl+C 로 끊으면 kit 자식이 남는다 → `pkill -f train.py`.
- 백그라운드 `docker exec` 를 `| tail -N` 으로 받지 말 것(EOF 까지 버퍼링 → 정지로 오인).
- `pkill -f "gz sim"` 류는 자기 셸까지 죽인다 → `"gz[ ]sim"`.
