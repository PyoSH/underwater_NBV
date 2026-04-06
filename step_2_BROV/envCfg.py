"""
BROV2 궤적 추종 환경 설정
==========================
DirectRLEnvCfg 서브클래스.
시뮬레이션 · 씬 · RL 공간 · 궤적 파라미터 · 수중 동역학 파라미터 ·
보상 가중치를 한 곳에서 관리한다.

관측 벡터 (17-dim)
------------------
pos_env(3)  : env origin 기준 로봇 위치 [m]
quat(4)     : 로봇 방향 쿼터니언 [w, x, y, z]
lin_vel_b(3): body-frame 선속도 [m/s]
ang_vel_b(3): body-frame 각속도 [rad/s]
wp_dir_b(3) : 다음 waypoint 방향 단위벡터 (body frame)
wp_dist(1)  : 다음 waypoint까지 거리 [m]

행동 벡터 (8-dim)
-----------------
8개 추진기 PWM 명령 [-1, 1]
"""

import os
import sys

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

sys.path.insert(0, os.path.dirname(__file__))
from sceneCfg import BROVSceneCfg


@configclass
class BROVTrajEnvCfg(DirectRLEnvCfg):
    """BROV2 궤적 추종 강화학습 환경 설정."""

    # ── 시뮬레이션 ──────────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(dt=1 / 100)
    # 물리 스텝 : 100 Hz  /  정책 스텝 : 25 Hz (decimation=4)

    # ── 씬 ─────────────────────────────────────────────────────────────────────
    scene: BROVSceneCfg = BROVSceneCfg(num_envs=4, env_spacing=20.0)

    # ── 에피소드 ────────────────────────────────────────────────────────────────
    episode_length_s: float = 60.0
    decimation      : int   = 4        # 정책 dt = 4 × (1/100) = 0.04 s

    # ── RL 공간 ─────────────────────────────────────────────────────────────────
    # obs: pos_env(3) + quat(4) + lin_vel_b(3) + ang_vel_b(3)
    #      + wp_dir_b(3) + wp_dist(1) = 17
    observation_space: int = 17
    # act: 8채널 추진기 PWM [-1, 1]
    action_space     : int = 8

    # ── 초기 수심 ───────────────────────────────────────────────────────────────
    starting_depth: float = 5.0    # [m] world Z 좌표 (brov_joint.py 기본값과 일치)

    # ── 궤적 파라미터 ──────────────────────────────────────────────────────────
    trajectory_type  : str   = "circle"  # "circle" | "helix"
    num_waypoints    : int   = 12        # 궤적 분할 개수
    trajectory_radius: float = 3.0      # [m] 원형 반경
    trajectory_height: float = 2.0      # [m] helix 전체 고도 변화 (circle 시 무시)
    # waypoint 도달 판정 거리 [m]
    waypoint_reach_threshold: float = 0.5

    # ── 경계 조건 (env origin 기준, 초과 시 terminated) ─────────────────────────
    max_bound_x: float = 12.0
    max_bound_y: float = 12.0
    max_bound_z: float = 10.0

    # ── 수중 동역학 파라미터 (MARINEGYM 방식) ──────────────────────────────────
    water_density: float = 997.0          # [kg/m³]
    volume       : float = 0.022747843    # [m³]  BROV2 Heavy 체적 (중성 부력 기준)
    cob_offset   : float = 0.01          # [m]   COM → 부력 중심 (+Z body)

    # ── 보상 가중치 ─────────────────────────────────────────────────────────────
    rew_scale_progress   : float =  1.0    # 이전보다 waypoint 에 가까워질 때
    rew_scale_waypoint   : float = 10.0    # waypoint 도달 보너스
    rew_scale_action     : float =  0.05   # 액션 크기 페널티 (에너지 효율)
    rew_scale_upright    : float =  0.3    # 자세 유지 (body Z ↑ 정렬)
    rew_scale_terminated : float = -5.0    # 경계 이탈 페널티
