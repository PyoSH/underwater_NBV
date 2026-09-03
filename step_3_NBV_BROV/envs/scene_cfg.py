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
# Jerlov 수질 프리셋은 step_1의 표를 정본으로 쓴다 — 값을 복사해 오면 세
# 파라미터가 따로 놀 수 있고, 실제로 그렇게 두 번 틀렸다(아래 WATER 주석).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "step_1_NBV"))
from utils_NBV.jerlov_presets import JERLOV_PRESETS
from robots.assets.brov_rigid import BROV_RIGID_CFG
from sensors.UWCamera.UW_Camera_cfg import UWCameraCfg
from sensors.ImagingSonar.ImagingSonarCfg import ImagingSonarCfg

OCEANSIM_DIR = "/isaac-sim/extsUser/OceanSim"
ASSET_DIR = os.path.join(OCEANSIM_DIR, "oceansim_asset")
ROCK_USD = os.path.join(ASSET_DIR, "collected_rock/rock.usd")

# ── 수질 (2026-09-02 정정) ────────────────────────────────────────────────────
# **세 파라미터를 반드시 한 세트로** 쓴다. 여기서 두 번 틀렸다:
#
# 1) step_1 이식 시 `atten_coeff`와 `backscatter_coeff`가 **전치**됐다.
#    OceanSim은 UW_param 배열을 [value(0:3), back_coeff(3:6), atten_coeff(6:9)]
#    순으로 넘기는데(UW_Camera.py:101-103, colorpicker ui_builder.py의 슬라이더
#    라벨과 YAML 저장부가 모두 일치), 이름 필드로 풀면서 순서를 뒤집어
#    atten=(0.05,0.05,0.20) / back=(0.05,0.05,0.05)이 됐다. OceanSim 의도는
#    그 반대다. 그 결과 **청색이 가장 강하게 감쇠**되어 거리가 멀수록 이미지가
#    적황색으로 변했다 — 실제 수중(적색이 먼저 흡수, 청색이 가장 멀리)과
#    정반대다. step_1의 모든 학습·평가가 이 조건이었다.
#
# 2) 2026-08-28에 `atten_coeff`만 Jerlov IB로 바꾸고 나머지 둘을 그대로 뒀다.
#    감쇠 방향은 우연히 바로잡혔지만 후방산란이 정본 대비 3~4배 약해져
#    "IB만큼 감쇠하는데 거의 맑은 물처럼 흐림이 없는" 비물리적 조합이 됐다.
#
# 그래서 개별 값을 쓰지 않고 프리셋 dict를 통째로 펼친다. 한쪽만 고치는 실수를
# 구조적으로 막는다.
#
# 주의: `atten_coeff`는 렌더링과 quality 모델이 **함께** 참조한다
# (`env.py::_sync_quality_water()`가 여기서 μ를 유도). 반면 backscatter는
# 렌더 이미지에만 영향을 주고 coverage/TSDF 계산에는 관여하지 않는다
# (TSDF는 GT depth를 쓰므로) — 즉 이번 정정은 **actor의 시각 입력 분포**를
# 바꾸지 신뢰 지표를 바꾸지 않는다.
_WATER = JERLOV_PRESETS["IB"]   # 외해 최청정. μ=0.233 (채널평균)

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

    # 대상 물체. `mesh_pool`이 비어 있으면 기존 단일 rock, 아니면 env마다 다른
    # USD를 스폰한다(Stage 4). 다중 메쉬는 `replicate_physics=False`를 요구하며
    # — IsaacLab 문서가 "separate assets in the environments"의 공식 경로로
    # 명시한다 — 비용은 **startup time에만** 붙고 런타임 처리량에는 영향이 없다.
    #
    # `random_choice=False`로 두는 이유: env↔메쉬 대응이 매 실행 고정돼야
    # 평가에서 정책 간 비교가 같은 조건에서 이뤄진다. 무작위면 실행마다 배치가
    # 달라져 비교가 흔들린다.
    rock: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=ROCK_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -3.0), rot=_ROT_45Z),
    )

    def use_mesh_pool(self, usd_paths: list[str]) -> None:
        """대상 물체를 다중 USD 풀로 교체한다. 씬 생성 **이전**에 호출할 것."""
        if not usd_paths:
            return
        self.rock.spawn = sim_utils.MultiUsdFileCfg(
            usd_path=list(usd_paths), random_choice=False,
        )
        self.replicate_physics = False

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
        # decimation=500이라 `update()`마다 수중 렌더를 돌리면 같은 프레임을
        # 499번 더 계산한다. `env.py::_get_rewards()`가 결정당 1회
        # `refresh_uw()`를 부른다 (2026-09-03: 결정당 8.31초 → 1.71초).
        defer_uw_render=True,
        offset=UWCameraCfg.OffsetCfg(
            pos=_CAMERA_FRAME_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        # 세 값 모두 동일 프리셋에서 — 개별 수정 금지(위 _WATER 주석 참조)
        backscatter_value=_WATER["backscatter_value"],
        atten_coeff=_WATER["atten_coeff"],
        backscatter_coeff=_WATER["backscatter_coeff"],
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
