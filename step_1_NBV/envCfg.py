from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
# from dataclasses import field
from sceneCfg import OceanSceneCfg
from collections import deque
import math
from os.path import join

@configclass
class VisualConfig:
    h:  int = 84
    w:  int = 84
    k:  int = 5 # num of past frames
    num_seq_actor:     int = 6
    num_seq_critic:    int = 6

@configclass
class TSDFCfg:
    vol_dim:        tuple = (40,40,40)
    voxel_size:     float = 0.05
    trunc_margin:   float = 0.25

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
class OceanEnvCfg(DirectRLEnvCfg):
    # ── 시뮬레이션 ───────────────────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=1)

    # ── 씬 설정 (InteractiveSceneCfg 서브클래스) ──────────────────────────
    scene: OceanSceneCfg = OceanSceneCfg(num_envs=1, env_spacing=5.0)

    # ── 에피소드 ─────────────────────────────────────────────────────────────
    episode_length_s: float = 20000.0

    # ── RL 공간 크기 ─────────────────────────────────────────────────────────
    
    
    visual: VisualConfig = VisualConfig()
    observation_space:  tuple = (visual.num_seq_actor, visual.h, visual.w)     # 2D gray image sequence for actor
    state_space:        tuple = (visual.num_seq_critic, visual.h, visual.w)    # depth map sequence for critic

    # 총 5개 = 구면좌표계 3개 (azimuth, elevation, distance) + contrast 1개 + 조명 밝기 단계 1개  
    num_scalar_obs: int = 5

    # 6개 (구면좌표계 3개에 대해 이산 +-) + 3개 (조명 3가지 이산 +,0,-)
    action_space:   int = 9

    # 시뮬레이션 스텝당 정책 업데이트 횟수 (policy dt = decimation * sim dt)
    decimation: int = 2

    # ── 물리 파라미터 ────────────────────────────────────────────────────────
    delta_theta:    float = math.radians(15)
    delta_phi:      float = math.radians(15)
    delta_psi:      float = 0.20

    phi_min:        float = math.radians(10)
    phi_max:        float = math.radians(80)
    psi_min:        float = 0.5
    psi_max:        float = 6.0

    light_level_init:           int = 3
    light_intensity_per_level:  float = 2_000_000.0

    tsdf:       TSDFCfg = TSDFCfg()
    mesh_root:  str = join("isaac-sim", "extsUser","OceanSim", "oceansim_asset", "collected_rock")

    # ── 보상 가중치 ──────────────────────────────────────────────────────────
    k_c:                float = 1.0
    lambda_q:           float = 1.0
    k_x:                float = 0.02
    c_step:             float = 2.0
    coverage_terminal:  float = 0.96

    # ── 카메라 센서 ──────────────────────────────────────────────────────────
    water_dr:           WaterParamRangeCfg = WaterParamRangeCfg()
    water_dr_enabled:   bool = False
    

    # ── 디버그 시각화 ─────────────────────────────────────────────────────────
    debug_vis:        bool = True  # 방향 마커 활성화 여부
    debug_vis_env_id: int  = -1     # -1: 전체 env, 0~N-1: 특정 env 만 시각화