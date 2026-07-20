"""
BROV2 6DOF 속도/자세 컨트롤러 환경 설정
==========================================
DirectRLEnvCfg 서브클래스. Sim2Swim(arXiv:2512.08656) 방식 저수준 RL 환경 —
경로 추종이 아니라 body-frame 속도(v_d^b) + 자세(q_d) 명령 추종만 학습한다.
경로 추종은 이 환경 밖(3D LOS guidance, 추후 구현)에서 고전 제어로 처리한다.

관측 벡터 (16-dim)
------------------
q_e(4)      : 쿼터니언 오차 q̄_d ⊗ q [w,x,y,z]
v_e_b(3)    : body-frame 선속도 오차 v^b - v_d^b [m/s]
ω_b(3)      : body-frame 각속도 [rad/s]
z_v(3)      : v_e_b 적분 상태
z_q(3)      : q_e vector part 적분 상태

행동 벡터 (6-dim)
-----------------
surge,sway,heave,roll,pitch,yaw 스케일 [-1, 1] — F_max로 스케일 후
할당행렬 B의 pseudo-inverse로 8-thruster PWM에 할당한다 (hydrodynamics.py 참조).
"""

import os
import sys

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

sys.path.insert(0, os.path.dirname(__file__))
from sceneCfg import BROVSceneCfg


@configclass
class BROVVelEnvCfg(DirectRLEnvCfg):
    """BROV2 6DOF 속도/자세 컨트롤러 강화학습 환경 설정."""

    # ── 시뮬레이션 ──────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(dt=1 / 100)

    # ── 씬 ─────────────────────────────────────────────────────────────────────
    scene: BROVSceneCfg = BROVSceneCfg(num_envs=512, env_spacing=5.0)

    # ── 에피소드 ────────────────────────────────────────────────────────────────
    episode_length_s: float = 5.0      # Sim2Swim: 짧은 에피소드 + 대규모 병렬화
    decimation      : int   = 4        # 정책 dt = 4 × (1/100) = 0.04 s

    # ── RL 공간 ─────────────────────────────────────────────────────────────────
    observation_space: int = 16   # q_e(4) + v_e_b(3) + ω_b(3) + z_v(3) + z_q(3)
    action_space     : int = 6    # surge,sway,heave,roll,pitch,yaw

    # ── 초기 수심 ───────────────────────────────────────────────────────────────
    starting_depth: float = 10.0   # 경로가 없어 경계에서 충분히 떨어진 곳에서 시작

    # ── 액션 할당 (hydrodynamics.build_allocation_matrix + inverse_thrust) ──────
    # F_max — von Benzon et al. 2022 (JMSE 10,1898) Table 4 실측 최대추력값.
    # (surge, sway, heave, roll, pitch, yaw) = [N, N, N, N·m, N·m, N·m]
    f_max: tuple = (85.0, 85.0, 120.0, 26.0, 14.0, 22.0)

    # ── 속도 명령 프로파일 (Sim2Swim Eq.9) ──────────────────────────────────────
    # v_d^b(t) = q_cmd ⊗ [a, b·sin(ωt), c·cos(ωt)], q_cmd는 에피소드마다 랜덤 샘플
    cmd_omega : float = 0.2
    cmd_coeffs: tuple = (0.5, 0.5, 0.3)   # [a, b, c]

    # ── 보상 가중치 (Sim2Swim Table 1, Eq.5-8) ─────────────────────────────────
    # r = w_quat·exp(-‖q_e_vec‖²) + w_vel·exp(-‖v_e_b‖²) + w_omega·exp(-‖ω_b‖²)
    #   + w_quat·exp(-∠(q_d,q))                                    (Eq.7, 별도 항, 제곱 없음)
    #   + w_action·exp(-‖a‖)
    rew_w_quat  : float = 0.4    # orientation error — Eq.6의 q_e 항과 Eq.7의 r_q 항 둘 다에 재사용
    rew_w_vel   : float = 0.2
    rew_w_omega : float = 0.05
    rew_w_action: float = 0.3

    # ── 안전 경계 (env origin 기준, 초과 시 terminated) ─────────────────────────
    # 경로 추종이 없는 태스크라 waypoint 기반 경계 대신 널찍한 안전판만 둔다.
    max_bound: float = 20.0

    # ── 도메인 랜덤화 1단계 (mass는 2단계로 연기, project_step2_brov_sim2swim 참조) ──
    dr_volume_range         : tuple = (0.01320, 0.01613)   # [m^3] nominal 0.014665 ±10%
    dr_cob_radius           : float = 0.015                 # [m] 구 반경 (부피 기준 균등 샘플)
    dr_added_mass_rot_range : tuple = (0.60, 1.40)           # nominal 대비 배율 (±40%, Kṗ/Mq̇/Nṙ)
