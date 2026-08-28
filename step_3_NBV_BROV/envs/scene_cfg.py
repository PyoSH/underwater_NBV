"""
step_3_NBV_BROV 씬 구성
========================
BROV2 물리 바디(robots/assets/brov_rigid.py::BROV_RIGID_CFG) 위에
step_1_NBV의 카메라/소나/조명을 로봇 동체의 자식 프림으로 고정 부착한다 —
step_1처럼 독립된 sensor_rig를 `write_root_state_to_sim()`으로 순간이동시키는
방식이 아니라, step_2_BROV의 물리 루프(Fossen 동역학+8-스러스터) 위에서
카메라가 실제로 로봇과 함께 이동/회전한다.

카메라/조명 오프셋은 `robots/data/BROV2/brov2_spec.md` §7의 실측 USD 프레임
좌표(Camera_frame/Light_L_frame/Light_R_frame)를 그대로 사용한다 — 임의값
아님. Camera_frame 자체는 USD에 위치만 있는 참조용 Xform(회전 항등원)이라
실제 IsaacLab Camera 센서는 이 프레임을 그대로 재사용하지 않고 동일 오프셋을
`UWCameraCfg.offset`으로 직접 지정해 새로 스폰한다 — thruster_pos_dir_ned()가
thruster_N_frame을 직접 쓰지 않고 YAML 미러값을 쓰는 것과 동일한 경계 원칙
("USD=PhysX가 읽는 값, YAML/코드=그 외 전부").

포함 에셋
---------
robot                : BROV2 Heavy Articulation
seafloor/walls/rock  : step_1_NBV/env/sceneCfg.py의 OceanSceneCfg 패턴 재사용
camera/sonar/lights  : robot 동체 자식 프림으로 고정 부착(오프셋 고정)
"""

import math
import os
import sys

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from robots.assets.brov_rigid import BROV_RIGID_CFG
from sensors.UWCamera.UW_Camera_cfg import UWCameraCfg
from sensors.ImagingSonar.ImagingSonarCfg import ImagingSonarCfg

OCEANSIM_DIR = "/isaac-sim/extsUser/OceanSim"
ASSET_DIR = os.path.join(OCEANSIM_DIR, "oceansim_asset")
ROCK_USD = os.path.join(ASSET_DIR, "collected_rock/rock.usd")

# rock 45° Z 회전 쿼터니언 [w, x, y, z] (step_1_NBV/env/sceneCfg.py와 동일)
_ROT_45Z = (math.cos(math.radians(22.5)), 0.0, 0.0, math.sin(math.radians(22.5)))

_FLOOR_DEPTH = -3.25
_WALL_HEIGHT = 5.5
_WALL_WIDTH = 0.01
_WALL_LENGTH = 10.0

_wall_material = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.01, 0.01, 0.01),
    metallic=0.0,
    roughness=1.0,
)

# brov2_spec.md §7 실측 USD 프레임 좌표 (world-frame USD 검증 완료, robot body frame 기준)
_CAMERA_FRAME_POS = (0.1575, 0.0053, 0.0678)
_LIGHT_L_FRAME_POS = (0.1962, 0.1918, -0.0486)
_LIGHT_R_FRAME_POS = (0.1962, -0.1848, -0.0486)


@configclass
class NBVBROVSceneCfg(InteractiveSceneCfg):
    """BROV2 물리 + NBV 카메라/소나 통합 씬."""

    robot: ArticulationCfg = BROV_RIGID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    seafloor: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Seafloor",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 10.0, 0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.50, 0.50)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, _FLOOR_DEPTH)),
    )

    wall_north: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall1",
        spawn=sim_utils.CuboidCfg(size=(_WALL_LENGTH, _WALL_WIDTH, _WALL_HEIGHT), visual_material=_wall_material),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, _WALL_LENGTH / 2, _FLOOR_DEPTH + _WALL_HEIGHT / 2)),
    )
    wall_south: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall2",
        spawn=sim_utils.CuboidCfg(size=(_WALL_LENGTH, _WALL_WIDTH, _WALL_HEIGHT), visual_material=_wall_material),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -_WALL_LENGTH / 2, _FLOOR_DEPTH + _WALL_HEIGHT / 2)),
    )
    wall_east: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall3",
        spawn=sim_utils.CuboidCfg(size=(_WALL_WIDTH, _WALL_LENGTH, _WALL_HEIGHT), visual_material=_wall_material),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(_WALL_LENGTH / 2, 0.0, _FLOOR_DEPTH + _WALL_HEIGHT / 2)),
    )
    wall_west: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Wall4",
        spawn=sim_utils.CuboidCfg(size=(_WALL_WIDTH, _WALL_LENGTH, _WALL_HEIGHT), visual_material=_wall_material),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-_WALL_LENGTH / 2, 0.0, _FLOOR_DEPTH + _WALL_HEIGHT / 2)),
    )

    rock: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=ROCK_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0), rot=_ROT_45Z),
    )

    # ── 카메라/소나/조명: 로봇 동체 자식 프림으로 고정 부착 ──
    # convention="world" + 항등원 회전 = 카메라가 부모(Robot) body frame의 +X를
    # 바라봄 (step_1_NBV/env/sceneCfg.py의 동일 설정이 sensor_rig에서 그렇게
    # 동작했던 것과 같은 컨벤션). BROV2의 +X도 선수(forward) 방향이라 별도
    # 회전 없이 그대로 맞는다 — brov2_spec.md §7이 지적한 "Camera_frame 회전
    # 미지정" 문제를 여기서 명시적으로 해결.
    camera: UWCameraCfg = UWCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Camera",
        update_period=0,
        height=240,
        width=320,
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 20.0)),
        offset=UWCameraCfg.OffsetCfg(
            pos=_CAMERA_FRAME_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        backscatter_value=(0.05, 0.31, 0.24),
        # Jerlov IB (외해 최청정) — `utils_NBV/jerlov_presets.py`의 표준값.
        #
        # 2026-08-28 변경. 이전 값 (0.05,0.05,0.20)은 채널평균 μ=0.100으로
        # **자연 해수 중 가장 맑은 IB보다도 2.3배 맑아** 물리적으로 비현실적이었고,
        # 그보다 심각하게는 quality coverage가 제 역할을 못 하게 만들었다:
        # 평가 실측(eval_out/run01)에서 2.88 m → 1.92 m 접근 시 voxel당 품질
        # 이득은 +10%인데 멀리서 얻는 관측 범위 이득이 +20%라, **후퇴가 순이득**
        # 이었다. 깨끗한 정책끼리 비교해도 같다 — orbit(2.22 m)이 random(1.92 m)
        # 보다 멀리 있으면서 cov_q가 더 높았다(0.568 vs 0.518).
        # 품질이 범위를 이기려면 μ > 0.190이 필요하고, IB의 μ=0.233이 이를 넘는다.
        #
        # 주의: 이 값은 렌더링 감쇠와 quality 모델이 **함께** 참조한다
        # (`env.py::_sync_quality_water()`가 여기서 μ를 유도). 한쪽만 바꾸면
        # coverage_q가 이미지에 없는 것을 재게 되므로 반드시 여기만 수정할 것.
        atten_coeff=(0.325835, 0.196346, 0.177762),
        backscatter_coeff=(0.05, 0.05, 0.05),
    )

    # 이미징 소나 (step_1_NBV/env/sceneCfg.py의 Oculus M750d 설정 그대로 재사용,
    # 위치만 카메라와 인접한 실측 위치로 조정 — 정확한 소나 마운트 실측치는
    # 아직 없어 카메라 프레임 인근으로 근사, 추후 보정 필요)
    sonar: ImagingSonarCfg = ImagingSonarCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Sonar",
        update_period=0,
        height=615,
        width=4000,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=103.0,
            clipping_range=(0.05, 6.0),
        ),
        offset=ImagingSonarCfg.OffsetCfg(
            pos=_CAMERA_FRAME_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        hori_res=4000.0,
        hori_fov=130.0,
        vert_fov=20.0,
        min_range=0.1,
        max_range=5.0,
        range_res=0.0025,
        angular_res=0.6,
        attenuation=0.1,
        binning_method="sum",
        normalizing_method="range",
        enable_viewport=False,
    )

    sphere_light_l: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/SphereLight_L",
        spawn=sim_utils.SphereLightCfg(intensity=0.0, radius=0.05, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_LIGHT_L_FRAME_POS, rot=(0.7071, 0.0, -0.7071, 0.0),
        ),
    )
    sphere_light_r: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/SphereLight_R",
        spawn=sim_utils.SphereLightCfg(intensity=0.0, radius=0.05, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_LIGHT_R_FRAME_POS, rot=(0.7071, 0.0, -0.7071, 0.0),
        ),
    )

    # ※ DomeLight 없음 — step_1_NBV/env/sceneCfg.py와 동일 조건.
    # 초기 구현에는 step_2_BROV/envs/scene_cfg.py에서 딸려온 dome light
    # (intensity=1200, 청색)가 있었으나, step_1은 전역 조명 없이 차량에 달린
    # SphereLight 2개만으로 대상을 비춘다(그게 수중 NBV의 전제 조건 — 조명이
    # 닿는 범위가 곧 관측 가능 범위). 조명 조건이 달라지면 인지 파이프라인의
    # 동작이 달라지므로 제거함(2026-08-26, 사용자가 GUI로 확인 후 지적).
    #
    # 위 SphereLight의 intensity는 여기서 0.0으로 선언하고 런타임에
    # `env.py::_update_light_intensity()`가 매 리셋마다
    # `light_level × light_intensity_per_level`로 설정한다(step_1과 동일 구조).
