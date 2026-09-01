"""
step_3_NBV_BROV 환경 설정
===========================
step_1_NBV/env/envCfg.py(TSDF/coverage/시각 관측 파라미터)와
step_2_BROV/envs/vel_env_cfg.py(물리 시뮬레이션 레이트, DP 컨트롤러 게인)를
결합한다. 액션은 처음부터 연속(Stage 1+2 동시 착수, `.claude/plans/
kind-launching-kahan.md` 권장 실행순서 1번) — step_1의 6-way discrete argmax는
포팅하지 않는다.
"""

from __future__ import annotations

import math
import os
import sys
from os.path import join

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.scene_cfg import NBVBROVSceneCfg


@configclass
class VisualConfig:
    h: int = 84
    w: int = 84
    num_seq_actor: int = 6
    num_seq_critic: int = 6


@configclass
class TSDFCfg:
    vol_dim: tuple = (40, 40, 40)
    voxel_size: float = 0.05
    trunc_margin: float = 0.05   # voxel_size와 동일 유지 필수 (step_1 known issue)


@configclass
class NBVBROVEnvCfg(DirectRLEnvCfg):
    # ── 시뮬레이션 ── 물리 100Hz(step_2와 동일 레이트)
    #
    # decimation=500 → 정책(NBV) dt = 500×(1/100) = **5.0초**. 실기 ROS2 구조를
    # 시뮬레이션에 그대로 재현한 것: 실제 시스템에서 NBV 노드와 컨트롤러 노드는
    # 서로 독립된 두 루프로, NBV는 느리게 목표 pose를 publish하고 컨트롤러는
    # 자기 주기로 그 목표를 계속 추종한다. DirectRLEnv의 decimation이 정확히 이
    # 이중루프를 표현한다 — env.step() 1회 = NBV 결정 1회 = 그 아래서 물리+DP
    # 컨트롤러가 500틱(5초) 동안 같은 목표를 추종. (초기값 4는 step_2의 25Hz
    # 속도추종 RL에서 그대로 복사해온 값이라 NBV 태스크엔 안 맞았음 — 목표가
    # 0.04초마다 갱신돼 로봇이 도달할 기회조차 없었다.)
    #
    # 이 구조라 ManagerBasedRLEnv/CommandManager 이관은 불필요하다: 그건
    # "빠르게 도는 RL 에이전트 + 느리게 갱신되는 command"의 주기 불일치를 푸는
    # 장치인데, 여기서는 저수준이 RL이 아니라 classical DP PID라 빠르게 도는
    # 에이전트 자체가 없고, 유일한 에이전트(NBV)의 자연 주기가 곧 5초다.
    #
    # render_interval도 decimation과 일치시킨다 — TSDF/관측 샘플은 정책스텝당
    # 1회(아래 참조)만 필요하므로, 그보다 잦은 렌더링은 순수 낭비다.
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=500)
    decimation: int = 500

    # ── 씬 ──
    scene: NBVBROVSceneCfg = NBVBROVSceneCfg(num_envs=4, env_spacing=10.0)

    # ── 에피소드 ──
    # NBV 결정 횟수 단위로 잡는다: 50초 = 5초 × **10회 결정**.
    # (step_1의 0.8333초/50스텝은 순간이동 전제라 물리 실행 불가.)
    #
    # 관측 샘플링 정책(사용자 확정): 이동 중 카메라가 스쳐 지나가며 얻는
    # 데이터는 "정책이 의도적으로 고른 시점"이 아니므로 TSDF에 반영하지
    # 않는다 — 목표 pose에 도달한 시점의 데이터만으로 NBV를 동작시킨다.
    # 그래서 `_integrate_depth()`/관측버퍼 갱신은 `_get_rewards()`/
    # `_get_observations()`에서 정책스텝당 딱 1회만 호출하는 현재 구조를
    # 그대로 유지한다(물리 서브스텝마다 융합하지 않는다).
    # 단 주의: 5초 시점에도 자세오차가 완전히 0은 아니므로(Stage 1 실측
    # ~6-10°) "도달"은 근사다 — step_1의 완벽한 순간이동과 달리 실기에
    # 가까운 조건이며, PID 게인을 더 조이면 줄어든다.
    # 125.0 = 25결정 (2026-09-01 상한 측정으로 확정, 이전 50.0 = 10결정).
    #
    # 10결정은 **과제를 판별 불가능하게 만드는 길이**였다. 40결정 포화 곡선
    # 실측(eval_out/ceiling)에서 지각 없는 orbit과 완전 무작위 random이
    # 결정 9까지 구분되지 않는다:
    #
    #   결정      1      5      9     13     17     21     25     29     37
    #   orbit  0.206  0.366  0.455  0.485  0.491  0.495  0.497  0.498  0.498
    #   random 0.205  0.357  0.461  0.520  0.582  0.631  0.661  0.685  0.737
    #
    # 분리는 결정 13부터 시작된다. 10결정 학습에서 모든 정책이 0.50~0.57에
    # 몰리고 아무것도 아무것도 이기지 못한 것, coverage_terminal=0.65가 도달
    # 불가였던 것, 적응형 커리큘럼이 0.511에서 멈춘 것이 전부 이것 때문이다.
    #
    # 25결정을 고른 이유: orbit 0.497 vs random 0.661로 격차가 0.164까지 벌어져
    # 판별력이 충분하고, 150K env-step 기준 에피소드 수가 6,000으로 여전히
    # 넉넉하다(10결정일 때 15,000). 더 길수록 격차는 커지지만 에피소드 수가
    # 반비례로 줄어든다.
    #
    # quality가 max 누적이라 각 voxel은 **한 번만** 가까이서 보면 값이 남는다 —
    # 결정 수가 늘수록 접근 기회가 늘어 gt_full이 크게 오르는 구조다
    # (40결정 random에서 gt_full 0.422 vs orbit 0.024).
    episode_length_s: float = 125.0

    # ── RL 공간 ──
    visual: VisualConfig = VisualConfig()
    observation_space: tuple = (visual.num_seq_actor, visual.h, visual.w)
    state_space: tuple = (visual.num_seq_critic, visual.h, visual.w)
    num_scalar_obs: int = 3   # 구면좌표 (theta, phi, psi) 정규화
    action_space: int = 3     # 연속 (Δtheta, Δphi, Δpsi) ∈ [-1,1]^3

    use_visit_map: bool = False

    # ── 구면좌표 목표점 파라미터 (rock 중심 기준) ──
    # 이산 delta_theta/phi/psi(step_1) 대신 정책스텝당 최대 변화율 — 연속
    # action(∈[-1,1])에 이 값을 곱해 목표점을 갱신한다.
    max_rate_theta: float = math.radians(30)   # [rad/policy-step]
    max_rate_phi: float = math.radians(30)     # [rad/policy-step]
    max_rate_psi: float = 0.5                  # [m/policy-step]

    phi_min: float = math.radians(10)
    phi_max: float = math.radians(80)
    psi_min: float = 1.0
    psi_max: float = 4.5

    tsdf: TSDFCfg = TSDFCfg()
    mesh_root: str = join("isaac-sim", "extsUser", "OceanSim", "oceansim_asset", "collected_rock")

    # ── 보상 가중치 (2026-08-26 재보정 — step_1 실제 학습 이력 조사 반영) ──
    #
    # ⚠ 초기 이식 오류 정정: 처음에는 `step_1_NBV/env/envCfg.py`의 **기본값**을
    # 그대로 가져왔는데(k_x=0.02, c_step=0.1), 조사 결과 step_1의 모든 학습·평가
    # 스크립트가 그 기본값을 오버라이드하고 있어서 **실제 학습에 쓰인 적이 없는
    # 값**이었다. envCfg 기본값이 아니라 train/eval 스크립트의 값이 정본이다.
    #
    #   항목      envCfg 기본값   step_1 실제 학습값
    #   k_x       0.02            0.0   (train_GenNBV.py: "# 이동 패널티 금지")
    #   c_step    0.1             0.02  (quality 런; binary GenNBV 런은 0.1)
    #
    # k_x=0.0 근거: git 이력상 k_x가 딱 한 커밋 동안 1.0이었고(2026-05-13) 다음
    # 날 50배 축소, 이후 모든 런에서 0.0 고정 — 문서의 "k_x>0 금지 / 접근 후 대기
    # 병폐"는 그 k_x=1.0 시기의 경험이다. step_3에서는 시간 비용을 c_step이
    # 이미 담당하므로(decimation 고정 → 결정 1회 = 항상 5초) 이동거리 패널티는
    # 중복이기도 하다. `dist_moved`는 보상에 안 쓰이더라도 진단 지표로 유지.
    #
    # c_step=0.02 근거: step_1의 명시적 규칙 `c_step ≪ k_c × Δcov_max`
    # (문서 원문: "현재 0.02 vs 5×0.05≈0.25" — 최대 획득량의 8% 수준).
    # step_3 실측 Δcov_max≈0.049라 k_c×Δcov_max≈0.25로 step_1과 거의 동일한데,
    # c_step=0.1은 그 40%로 규칙 위반이었다. 0.02면 최소 실측치(Δcov=0.012)에서도
    # 0.06−0.02=+0.04로 순보상이 양수를 유지한다.
    #
    # k_c=5.0 유지(step_1 관례). 단 주의: step_1 실측 로그(step 460800) 기준
    # 보상 분해가 `cov_q=+0.052 / success=+0.293 / net=+0.321`로 **성공 보너스가
    # 전체의 91%**를 차지했다. step_3는 에피소드당 결정이 50→10회로 줄어 shaping이
    # 더 약해지므로, 학습에서 정체가 보이면 k_c 상향을 우선 검토할 것.
    # k_c=20.0 / coverage_bonus=10.0 (2026-08-26 2차 재보정 — 학습 스모크 실측 반영):
    # 1차 재보정(c_step 0.1→0.02) 후 스모크를 돌려보니 보상의 **87~99%가 성공
    # 보너스**로, step_1의 91%보다 오히려 나빴다(value loss 2468 폭발, KL 초과로
    # 매 롤아웃 early_stop → 사실상 1에포크만 학습). 균형 계산:
    #   bonus=10 단독:  shaping 5×0.5=2.5  vs bonus 10 → 보너스 80% (여전히 지배)
    #   k_c=20 병행:    shaping 20×0.5=10  vs bonus 10 → 50:50 (균형)
    # step_1은 보너스 91% 구조였고 **그 최종 정책이 정체**했으므로(위 참조) 같은
    # 균형을 복제하면 같은 실패를 복제할 위험이 있다. 비교 가능성은 인코더/PPO
    # 하이퍼파라미터 동일성으로 이미 확보돼 있고, 보상 가중치는 환경마다
    # 재보정하는 것이 정상이라 판단.
    # bonus=10은 step_1 `algo_UW_NBV.py` 문서의 권장값이기도 하다(실제로는 30이
    # 쓰였지만 — 조사에서 확인된 문서/코드 불일치 중 하나).
    k_c: float = 20.0
    k_x: float = 0.0
    c_step: float = 0.02
    coverage_terminal: float = 0.65
    coverage_bonus: float = 10.0

    # ── 커리큘럼 (2026-08-26 사용자 확정: coverage_terminal 임계값 상향 방식) ──
    # 성공 임계값을 낮게 시작해 학습 진행에 따라 목표치까지 선형으로 올린다.
    # 초반에 성공 보상(coverage_bonus)을 실제로 받아볼 수 있어야 정책이 무엇을
    # 최적화해야 하는지 배울 수 있고, 그 뒤 난이도를 올리는 구조.
    # step_1에도 같은 축으로 조정한 이력이 있음(0.52→0.65→0.82, step_1 CLAUDE.md).
    #
    # 당초 후보였던 "DR off→on" 커리큘럼은 plant DR 자체를 제외하면서 소멸.
    # `curriculum_total_steps`는 train.py가 총 학습 스텝으로 설정해야 하며,
    # 0이면 커리큘럼 비활성(= coverage_terminal 고정) — vel_env.py의
    # `action_envelope_curriculum_total_steps` 패턴과 동일한 관례.
    # start=0.45 근거(2026-08-26 실측 기반 재조정): 초기값 0.30은 감으로 정한
    # 값이었는데, 실측 coverage 곡선상 **첫 결정 하나만으로 0.207**에 도달하고
    # 미숙한 정책도 3~4결정이면 0.30을 넘어(스모크에서 종료 시점 coverage가
    # 0.38~0.40으로 측정됨) 정책 품질과 무관하게 성공 보너스가 터졌다 = 학습
    # 신호 없음. 0.45는 미숙한 정책의 도달선(0.40)보다 위라 "잘해야 성공"이 된다.
    # end=0.65는 순수 azimuth 공전 12결정 결과(0.53)보다 높아 적절히 어렵다.
    # (step_1이 "wandb 분포에서 max window mean ≈ 0.507 → 0.52"처럼 데이터로
    #  임계값을 정했던 방식을 따른 것.)
    curriculum_enabled: bool = True
    # 2026-09-01 재조정 — 25결정 지평의 실측 베이스라인에 맞춤.
    # start 0.55: 결정 13 근처에서 어떤 정책이든 도달하는 수준(0.49~0.52)
    #   바로 위라, 초반에 성공 보상을 받아보면서도 공짜는 아니다.
    # end 0.80: 결정 25에서 random이 0.661이므로 상한이 그보다 충분히 위에
    #   있어야 "random을 넘어라"는 압박이 걸린다. 이전 0.65는 random에도
    #   못 미쳐 커리큘럼이 정책을 압박할 수 없었다.
    curriculum_coverage_terminal_start: float = 0.55
    curriculum_coverage_terminal_end: float = 0.80
    curriculum_total_steps: int = 0

    # ── 적응형 커리큘럼 (2026-08-27) ────────────────────────────────────────
    # 스텝 기반 선형 상향은 정책 성능과 무관하게 진행되므로 정책을 앞지를 수
    # 있고, 9.3시간 런에서 실제로 3.3배 앞질렀다(임계값 +0.199 vs coverage
    # +0.061). 최근 성공률이 게이트를 넘을 때만 올리면 정의상 앞지를 수 없다.
    #
    # 게이트 0.7 근거: 성공률이 이 정도면 "현재 난이도를 익혔다"고 보기 충분하고,
    # 동시에 학습 신호(성공/실패 대비)가 남아 있는 수준. 1.0에 가깝게 두면
    # 난이도가 거의 안 오르고, 0.5면 절반만 성공해도 올려 앞지르기가 재발한다.
    #
    # rate 0.002 = "num_envs개 에피소드가 끝날 때마다의 상승폭"(비례 환산은
    # `_update_curriculum()` 주석 참조). 150K 스텝 / 에피소드 8결정 / 64 env면
    # 약 293 배치이므로, 계속 성공한다는 가정에서 최대 +0.59까지 오를 수 있다 —
    # start 0.45에서 end 0.65까지 도달하고도 충분한 여유.
    #
    # False면 기존 스텝 기반 선형 상향(A/B 비교용).
    curriculum_adaptive: bool = True
    curriculum_success_gate: float = 0.7
    # 0.005 — 에피소드가 25결정으로 길어지면서 총 에피소드가 15,000 → 6,000,
    # num_envs 배치가 234 → 94로 줄었다. 기존 0.002면 최대 상승이 0.188뿐이라
    # start 0.45에서 0.637까지밖에 못 올라가 **random의 0.661에도 못 미친다**.
    # 0.005면 0.469까지 오를 수 있어 상한 0.80에 도달 가능하다. 성공률
    # 게이트가 걸려 있으므로 빠른 상승률 자체는 위험하지 않다(자기 제한).
    curriculum_rate: float = 0.005
    curriculum_success_ema_alpha: float = 0.05

    # ── Quality-weighted coverage (2026-08-27 이식) ──────────────────────────
    # step_1의 연구 기여인 **거리 기반 관측 품질**(Beer-Lambert)을 step_3에 이식.
    # 초기 step_3는 step_1의 **부모** 클래스 `env_GenNBV.py`(binary)만 이식하고
    # 서브클래스 `env_GenNBV_quality.py`의 오버라이드를 빠뜨려서, "5 m 밖에서
    # 스친 voxel"과 "1 m 앞에서 정면으로 본 voxel"이 동일하게 계산됐다 =
    # 가까이 갈 유인이 목적함수에 없었다. (`_get_vox_actor()`를 빠뜨렸던 것과
    # 같은 유형의 누락.)
    #
    #   q(v) = exp(-μ·d)    d = 카메라 ↔ voxel 중심 거리
    #   Q_vol = max(Q_vol, q)            (반복 방문 시 최근접 품질만 기록)
    #   coverage_q = Σ(Q_vol × surf_vol) / total_surf_voxels
    #
    # False로 두면 기존 binary coverage로 돌아간다 — 2026-08-26 9.3시간 런이
    # binary 기준 baseline이라 A/B 비교가 가능하도록 남겨둔다.
    use_quality_coverage: bool = True

    # μ와 Q_sat은 **설정하지 않는다** — 카메라의 실제 `atten_coeff` 채널 평균에서
    # 유도한다(step_1 `env_GenNBV_quality.py::_reset_idx`와 동일 메커니즘).
    # 렌더러가 실제로 적용하는 감쇠와 품질 모델이 어긋나면 coverage_q가 이미지에
    # 없는 것을 재는 지표가 되기 때문. step_3 `scene_cfg.py`의
    # atten_coeff=(0.05,0.05,0.20) → μ=0.10 → Q_sat=exp(-0.10×psi_min)=0.905.
    #
    # 주의: step_1은 DR이 꺼져 있으면 μ가 초기값 0.1에 머무는데 Q_sat은 cfg의
    # 0.80을 그대로 써서 둘이 불일치했다(step_1 CLAUDE.md §13 "Q_sat 설계 불일치").
    # step_3는 DR 여부와 무관하게 항상 둘을 함께 유도해 이 불일치를 없앤다.
    #
    # 종료·커리큘럼·보상은 모두 **정규화값 `coverage_q / Q_sat` (0~1)** 을 쓴다.
    # 그래야 ⓐ `coverage_terminal=0.65`가 step_1과 같은 "달성 가능 상한의 65%"
    # 의미를 유지하고 ⓑ 위에서 실측으로 맞춰둔 k_c/c_step/coverage_bonus 보정이
    # binary와 같은 스케일에서 그대로 유효하다.
    lambda_q: float = 0.1
    k_still: float = 0.05
    stall_thr: float = 1e-4
    k_explore: float = 0.0

    # ── 조명 (step_1_NBV/env/envCfg.py와 동일 값) ──
    # scene_cfg.py의 SphereLight는 intensity=0.0으로 선언되고, 런타임에
    # `env.py::_update_light_intensity()`가 매 리셋마다
    # `light_level × light_intensity_per_level`(= 7 × 200,000 = 1,400,000)로
    # 설정한다. 초기 구현에서 이 호출을 빠뜨려 차량 조명이 계속 꺼진 채
    # 돌고 있었음(2026-08-26 수정).
    light_level_init: int = 7
    light_intensity_per_level: float = 200_000.0

    # ── 카메라 viewport 시각화 (GUI에서 실제 획득 이미지를 창으로 확인) ──
    # UWCameraCfg.enable_viewport를 켜면 수중 감쇠까지 적용된 uw_rgb를
    # 별도 ui.Window로 띄운다. 카메라 prim이 씬 생성 시점에 스폰되므로
    # `env.__init__`에서 `super().__init__()` **이전에** 설정해야 한다.
    enable_camera_viewport: bool = False
    camera_viewport_env_id: int = 0

    # ── 소나 (기본 비활성, 2026-08-26) ──
    # step_1 `sceneCfg.py`에서 딸려온 ImagingSonar는 **관측/보상 어디에도 쓰이지
    # 않으면서** 매 프레임 렌더링된다(step_1에서도 동일하게 죽은 파이프라인이었음).
    # 해상도가 615×4000 = 2.46M 픽셀로 **카메라(240×320=76.8K)의 32배**라
    # VRAM/렌더 비용이 크다 — 실측상 step_3 학습의 VRAM 병목은 RL(0.15 GB)이
    # 아니라 전적으로 센서 렌더링이므로, 이 죽은 센서를 끄는 것이 유일하게
    # 의미 있는 절감 레버다.
    # 배선은 남겨둠 — Stage 3에서 TRIDENT 단안 depth 정확도가 부족할 경우
    # 소나 융합이 2차 대안으로 계획돼 있다([[project_step3_roadmap]] Stage 3).
    enable_sonar: bool = False

    # ── 카메라 수중 DR (step_1과 동일, 기본 비활성) ──
    jerlov_dr_enabled: bool = False
    jerlov_types: tuple = ("IB", "II", "III", "1C", "3C", "5C")

    # ── DP(Dynamic Positioning) PID 게인 — control/dp_controller.py ──
    # tau_max는 vel_env_cfg.py의 f_max와 동일 출처(von Benzon et al. 2022 Table 4).
    #
    # dp_ki_att/dp_integral_att_limit 실측 근거(2026-08-24, isaac-lab-base 컨테이너
    # 스모크테스트): 초기값(ki_att=0.5, limit=2.0)은 실측 CoB 오프셋(~10mm, DR
    # 0 인 nominal 조차)의 트림모멘트를 못 이겨 목표 미부여(zero action) 상태에서도
    # 자세오차가 100스텝(4초) 동안 0°→165°까지 무한정 발산 — CoB를 정확히 0으로
    # 강제하면 동일 조건에서 완벽히 정지(오차 <1e-5°)함을 확인해 원인을 CoB
    # 트림모멘트 하나로 특정. `[[project_step2_brov_retrain_spec]]`의 pitch windup
    # 사가와 동일한 클래스의 문제(단, 여기는 classical PID라 "학습분포 이탈"이
    # 아니라 순수 정상상태오차 제거력 부족).
    #
    # ⚠ 부분 개선, 완전 해결 아님 (2026-08-24): nominal CoB(DR 없음) zero-action
    # 홀드 80스텝 기준으로 kp_att/kd_att 10종 스윕(같은 Isaac Sim 세션 안에서
    # env.reset()+DPController 교체로 재실행, 매번 재구동 안 함) 실시. 원래값
    # (kp=5,kd=3)은 후반 평균 자세오차 117.5°(지속 성장)였는데, **kd_att를 kp_att
    # 대비 ~2:1 비율로 크게 올릴수록 일관되게 개선**됨 — 절대 크기보다 댐핑비가
    # 지배적. 최선(kp=20,kd=40)이 후반 평균 10.7°로 약 11배 개선. 그래도 완전한
    # steady-state 수렴은 아직 아님(느리게 계속 자람, GROWING 추세는 10종 전부
    # 동일) — "정밀 튜닝은 불필요, 대략 개선"이라는 요청 범위 안에서 채택한
    # 값이며, 완전 해결하려면 bottom_up.py류 축별 스텝응답 테스트로 더 큰
    # kd_att(예: 60~80)나 Kp_att 자체를 추가로 올려보는 후속 튜닝이 필요.
    # Kd 하향 확정 (2026-08-26, 통제조건 스윕 — DR off + eval_mode 고정스폰으로
    # 후보 간 공정 비교). 실기 IMU/DVL 노이즈는 미분/속도 궤환에서 증폭되므로
    # Kd를 낮추는 게 목표였는데, 측정 결과 **낮은 Kd가 모든 지표에서 우세**했다:
    #   kd_att=40: 최종 0.346°, 궤적 3.96→3.88→3.35→2.35→1.21→0.35 (과감쇠, 느림)
    #   kd_att=10: 최종 0.370°, 궤적 3.81→2.06→1.70→0.69→0.40→0.37
    #   kd_att= 5: 최종 0.188°, 궤적 3.32→2.14→1.69→0.56→0.46→0.19  ← 채택
    #   kd_att= 3: 최종 0.253°, kd_att=2: 최종 0.295° (진동/불안정 징후 없음)
    # 원래값 40은 자세 P항 부호 버그(2026-08-25 수정)를 Kd로 억누르려다 부풀려진
    # 값이었고, 부호가 정상인 지금은 과감쇠로 수렴만 느리게 만든다.
    # 5.0 채택 근거: 최종오차 최저 + 40 대비 노이즈 증폭 8배 감소 + 2~3보다
    # 감쇠 여유가 있어 향후 노이즈 유입 시 진동 마진 확보.
    dp_kp_pos: tuple = (15.0, 15.0, 15.0)
    dp_ki_pos: tuple = (1.5, 1.5, 1.5)
    dp_kd_pos: tuple = (5.0, 5.0, 5.0)
    dp_kp_att: tuple = (20.0, 20.0, 20.0)
    dp_ki_att: tuple = (2.5, 2.5, 2.5)
    dp_kd_att: tuple = (5.0, 5.0, 5.0)
    dp_tau_max: tuple = (85.0, 85.0, 120.0, 26.0, 14.0, 22.0)
    # integral_pos_limit: 적분항이 상수 외란(순부력 불균형)을 전부 상쇄할 수
    # 있을 만큼은 크되, 불필요하게 크면 노이즈 유입 시 windup 위험이 된다.
    #
    # 이력: DR이 켜져 있던 동안은 volume(±10%)+mass(±5%) 조합으로 순힘이
    # 최대 ±22 N까지 벌어져 한도 20.0이 필요했다(초기값 2.0으로는 적분이
    # 한계에 걸려 0.18~0.26 m 정상상태오차가 남는 걸 실측). 2026-08-26에
    # step_3 DR을 끄면서 nominal 기준 순부력 불균형은 부력 143.4 N vs 무게
    # 143.6 N ≈ 0.2 N로 거의 중성이 됐으므로, 한도를 5.0으로 되돌린다
    # (ki_pos=1.5 기준 7.5 N까지 낼 수 있어 여전히 충분한 여유).
    dp_integral_pos_limit: float = 5.0
    dp_integral_att_limit: float = 5.0

    # ── 안전 경계 (env origin 기준, 초과 시 terminated) ──
    max_bound: float = 20.0

    # ── 물리 도메인 랜덤화 — **step_3에서는 기본 비활성** (2026-08-26 사용자 확정) ──
    #
    # 이 DR(mass/volume/CoB/added-mass)은 step_2 Sim2Swim의 **RL 속도컨트롤러**가
    # 플랜트 변동에 강건해져 sim2real 전이를 하기 위한 장치였다. step_3의 저수준은
    # classical DP PID라 "플랜트 변동에 강건해져야 할 학습 정책"이 저수준에 없고,
    # 남는 효과는 에피소드마다 컨트롤러 정착 성능을 무작위로 흔들어 **NBV 정책의
    # credit assignment를 흐리는 것**뿐이다(같은 관측+같은 액션인데 도달 pose가
    # 달라짐). 그래서 step_3에서는 제외한다.
    #
    # 주의: "랜덤화"만 끄는 것이지 실제 물리 특성을 지우는 게 아니다 —
    # brov2_heavy.yaml의 nominal volume/CoB(≈10mm 오프셋)/added-mass는 그대로
    # 적용된다(dr_cob_radius=0.0은 nominal 주변 랜덤 샘플 반경이 0이라는 뜻).
    #
    # 학습 기반 저수준 제어는 NBV 실기 배포 프로젝트 이후에 진행될 예정이며,
    # 그 시점에 이 플래그들을 다시 켜면 된다(배선은 그대로 남겨둠).
    #
    # ★ 제외 범위는 **plant DR(아래 5개)뿐**이다. NBV/인지 쪽 DR은 그대로 유지:
    #   - `_randomize_rock_pose()` (env.py): 타겟 rock의 자세(yaw 0~360°,
    #     roll/pitch ±30°)와 스케일(0.8~1.5배)을 매 리셋마다 랜덤화 — NBV
    #     정책이 일반화해야 할 대상이므로 필수.
    #   - `jerlov_dr_enabled` (아래): 수중 광학 특성 DR — 인지 쪽 변동이라 유지
    #     (현재 기본 False지만 그건 step_1에서 넘어온 기본값이지 제외 결정이 아님).
    dr_enable_mass: bool = False
    dr_mass_scale_range: tuple = (1.0, 1.0)
    dr_volume_range: tuple = (0.014665, 0.014665)   # brov2_heavy.yaml nominal
    dr_cob_radius: float = 0.0
    dr_added_mass_rot_range: tuple = (1.0, 1.0)

    # ── 디버그 시각화 ──
    debug_vis: bool = False
    debug_vis_env_id: int = -1

    # ── 평가 모드 (고정 시작 구면좌표) ──
    eval_mode: bool = False
    eval_theta: float = 0.0
    eval_phi: float = math.radians(45)
    eval_psi: float = 4.5
