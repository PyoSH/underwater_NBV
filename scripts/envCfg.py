from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from dataclasses import field
from sceneCfg import OceanSceneCfg

@configclass
class WaterParamRangeCfg:
    """수중 파라미터 DR 범위 설정."""

    # Backscatter value (veiling light) RGB 범위
    backscatter_value_min: tuple = (0.0,  0.20, 0.15)
    backscatter_value_max: tuple = (0.05, 0.40, 0.35)

    # Attenuation coefficient RGB 범위
    atten_coeff_min: tuple = (0.03, 0.03, 0.10)
    atten_coeff_max: tuple = (0.15, 0.10, 0.40)

    # Backscatter coefficient RGB 범위
    backscatter_coeff_min: tuple = (0.02, 0.02, 0.02)
    backscatter_coeff_max: tuple = (0.15, 0.12, 0.10)

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
    baseline_min:         float = 0.5  # 카메라-조명 최소 baseline [m]
    baseline_max:         float = 5.0   # 카메라-조명 최대 baseline [m]

    # ── 리셋 랜덤화 (카메라 초기 위치 구면 좌표 범위) ────────────────────────
    reset_radius_min: float = 1.0   # rock 에서 최소 거리 [m]
    reset_radius_max: float = 3.5   # rock 에서 최대 거리 [m]
    reset_theta_min:  float = 0.35  # 최소 앙각 [rad] ≈ 20°
    reset_theta_max:  float = 1.22  # 최대 앙각 [rad] ≈ 70°
    light_baseline:   float = 0.5  # 리셋 시 카메라-조명 baseline [m]

    # ── 보상 가중치 ──────────────────────────────────────────────────────────
    w_distance:  float = 1.0   # 거리 보상
    w_direction: float = 0.8   # 카메라 시선이 rock 을 향하는 정도
    w_baseline:  float = 0.5   # baseline 페널티

    # ── 카메라 센서 ──────────────────────────────────────────────────────────
    water_dr:           WaterParamRangeCfg = WaterParamRangeCfg()
    water_dr_enabled:   bool = False
      
    cam_height:        int  = 480    # 이미지 세로 해상도 [px]
    cam_width:         int  = 640    # 이미지 가로 해상도 [px]

    sonar_min_range:    float = 0.2
    sonar_max_range:    float = 10.0
    sonar_hori_fov:     float = 130.0
    sonar_vert_fov:     float = 20.0
    sonar_hori_res:     int = 1024

    # ── 디버그 시각화 ─────────────────────────────────────────────────────────
    debug_vis:        bool = True  # 방향 마커 활성화 여부
    debug_vis_env_id: int  = -1     # -1: 전체 env, 0~N-1: 특정 env 만 시각화