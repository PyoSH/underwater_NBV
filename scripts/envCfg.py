"""
envCfg.py
---------
강화학습 환경 설정.

행동 공간 (12차원 연속):
    [cam_vx, cam_vy, cam_vz,  cam_wx, cam_wy, cam_wz,
     light_vx, light_vy, light_vz,  light_wx, light_wy, light_wz]
    → 정규화 [-1, 1], max_velocity / max_angular_velocity 로 스케일

관측 공간 (17차원):
    [cam_pos(3), cam_orient(4),        # 카메라 리그 위치 + 쿼터니언 [w,x,y,z]
     light_pos(3), light_orient(4),    # 조명 리그 위치 + 쿼터니언
     cam_to_rock(3)]                   # 카메라→rock 벡터

보상:
    - 카메라가 rock 에서 optimal_cam_dist 거리일 때 최대  (w_distance)
    - 카메라 시선이 rock 을 향할수록 최대               (w_direction)
    - 카메라-조명 baseline 이 범위를 벗어나면 페널티     (w_baseline)

종료 조건:
    - 카메라가 workspace_radius 밖으로 이탈  → terminated
    - episode_length_s 초과                 → truncated
"""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from sceneCfg import OceanSceneCfg


@configclass
class OceanNBVEnvCfg(DirectRLEnvCfg):
    # ── 시뮬레이션 ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=1)

    # ── 씬 설정 (InteractiveSceneCfg 서브클래스) ──────────────────────────
    scene: OceanSceneCfg = OceanSceneCfg(num_envs=1, env_spacing=5.0)

    # ── 에피소드 ─────────────────────────────────────────────────────────────
    episode_length_s: float = 20.0

    # ── RL 공간 크기 ─────────────────────────────────────────────────────────
    # cam_pos(3) + cam_orient(4) + light_pos(3) + light_orient(4) + cam_to_rock(3) = 17
    observation_space: int = 17
    # cam_vel(3) + cam_angvel(3) + light_vel(3) + light_angvel(3) = 12
    action_space:      int = 12

    # 시뮬레이션 스텝당 정책 업데이트 횟수 (policy dt = decimation * sim dt)
    decimation: int = 2

    # ── 물리 파라미터 ────────────────────────────────────────────────────────
    max_velocity:         float = 0.1    # 선속도 스케일 [m/s]
    max_angular_velocity: float = 1.0   # 각속도 스케일 [rad/s] ≈ 57°/s
    optimal_cam_dist:     float = 1.5   # rock 에서 최적 거리 [m]
    workspace_radius:     float = 3.0   # 이 거리를 넘으면 에피소드 종료 [m]
    baseline_min:         float = 0.05  # 카메라-조명 최소 baseline [m]
    baseline_max:         float = 5.0   # 카메라-조명 최대 baseline [m]

    # ── 리셋 랜덤화 (카메라 초기 위치 구면 좌표 범위) ────────────────────────
    reset_radius_min: float = 1.0   # rock 에서 최소 거리 [m]
    reset_radius_max: float = 2.5   # rock 에서 최대 거리 [m]
    reset_theta_min:  float = 0.35  # 최소 앙각 [rad] ≈ 20°
    reset_theta_max:  float = 1.22  # 최대 앙각 [rad] ≈ 70°
    light_baseline:   float = 0.15  # 리셋 시 카메라-조명 baseline [m]

    # ── 보상 가중치 ──────────────────────────────────────────────────────────
    w_distance:  float = 1.0   # 거리 보상
    w_direction: float = 0.8   # 카메라 시선이 rock 을 향하는 정도
    w_baseline:  float = 0.5   # baseline 페널티

    # ── 카메라 센서 ──────────────────────────────────────────────────────────
    cam_height:        int  = 480    # 이미지 세로 해상도 [px]
    cam_width:         int  = 640    # 이미지 가로 해상도 [px]
    cam_save_interval: int  = 200    # N 스텝마다 이미지 저장 (0 = 비활성화)
    cam_save_dir:      str  = "/tmp/ocean_rl_images"  # 저장 경로
    cam_realtime_vis:  bool = True  # cv2 실시간 윈도우 표시

    # ── 디버그 시각화 ─────────────────────────────────────────────────────────
    debug_vis:        bool = True  # 방향 마커 활성화 여부
    debug_vis_env_id: int  = -1     # -1: 전체 env, 0~N-1: 특정 env 만 시각화
