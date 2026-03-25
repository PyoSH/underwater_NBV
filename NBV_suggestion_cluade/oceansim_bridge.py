"""
oceansim_bridge.py
------------------
OceanSim Isaac Sim Extension의 센서 출력을 NumPy 배열로 래핑한다.

OceanSim 제공 센서:
  - UnderwaterCamera: Eq.1 수중 광학 효과가 적용된 RGB + Depth
  - ImagingSonar:     GPU 레이트레이싱 기반 포인트클라우드

Isaac Sim 환경이 없을 경우 SyntheticBridge를 사용하여
수식 기반 합성 데이터를 생성한다 (테스트/개발용).

OceanSim Eq.1:
  I_c = J · exp(-β_attn · d) + B∞ · (1 - exp(-β_bs · d))
  J: 실제 장면 복사, d: 깊이, β_attn: 감쇠, β_bs: 후방산란, B∞: 원거리 배경
"""

import numpy as np
import json
from pathlib import Path


# ── Isaac Sim 환경 확인 ───────────────────────────────────────────────────────
try:
    import omni.isaac.sensor
    from omni.isaac.core.utils.numpy.rotations import quats_to_rot_matrices
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 공개 API: 팩토리 함수
# ─────────────────────────────────────────────────────────────────────────────

def create_bridge(config_path: str,
                  camera_prim_path: str = "/World/ROV/LeftArm/tool0/Camera",
                  sonar_prim_path:  str = "/World/ROV/RightArm/tool0/Sonar",
                  synthetic: bool = False):
    """
    환경에 맞는 Bridge 인스턴스를 반환한다.

    Args:
        config_path:       water_params.json 경로
        camera_prim_path:  Isaac Sim 카메라 prim 경로
        sonar_prim_path:   Isaac Sim 소나 prim 경로
        synthetic:         True이면 Isaac Sim 없이 합성 데이터 생성

    Returns:
        IsaacSimBridge 또는 SyntheticBridge 인스턴스
    """
    params = _load_water_params(config_path)

    if synthetic or not ISAAC_SIM_AVAILABLE:
        return SyntheticBridge(params)
    else:
        return IsaacSimBridge(params, camera_prim_path, sonar_prim_path)


# ─────────────────────────────────────────────────────────────────────────────
# IsaacSimBridge: 실제 OceanSim 연동
# ─────────────────────────────────────────────────────────────────────────────

class IsaacSimBridge:
    """
    Isaac Sim OceanSim Extension에서 센서 데이터를 취득한다.
    """

    def __init__(self, params: dict,
                 camera_prim_path: str,
                 sonar_prim_path: str):
        self.params           = params
        self.camera_prim_path = camera_prim_path
        self.sonar_prim_path  = sonar_prim_path
        self._setup_sensors()

    def _setup_sensors(self):
        """OceanSim 센서 prim에 접근하는 핸들을 초기화한다."""
        import omni.replicator.core as rep
        self._camera_rp = rep.create.render_product(
            self.camera_prim_path, resolution=(640, 480))
        self._rgb_anno   = rep.AnnotatorRegistry.get_annotator("rgb")
        self._depth_anno = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        self._rgb_anno.attach([self._camera_rp])
        self._depth_anno.attach([self._camera_rp])

    def get_rgb(self) -> np.ndarray:
        """
        Returns:
            (H, W, 3) uint8 RGB 이미지 (OceanSim 수중 효과 포함)
        """
        data = self._rgb_anno.get_data()
        return data[:, :, :3]

    def get_depth(self) -> np.ndarray:
        """
        Returns:
            (H, W) float64 깊이 맵 (미터)
        """
        return self._depth_anno.get_data().astype(np.float64)

    def get_sonar_pointcloud(self) -> np.ndarray:
        """
        ImagingSonar에서 포인트클라우드를 취득한다.

        Returns:
            (P, 3) float64 포인트 좌표 (월드 좌표계, 미터)
        """
        import omni.isaac.sensor as sensor_ext
        sonar = sensor_ext.get_sensor(self.sonar_prim_path)
        data  = sonar.get_current_frame()
        return np.array(data['point_cloud'], dtype=np.float64)

    def get_water_params(self) -> dict:
        return self.params


# ─────────────────────────────────────────────────────────────────────────────
# SyntheticBridge: Isaac Sim 없이 수식 기반 합성 데이터
# ─────────────────────────────────────────────────────────────────────────────

class SyntheticBridge:
    """
    Isaac Sim 없이 OceanSim Eq.1을 직접 적용하여 합성 센서 데이터를 생성한다.
    개발, 테스트, 독립 실행에 사용한다.

    OceanSim Eq.1:
      I_c = J · exp(-β_attn · d) + B∞ · (1 - exp(-β_bs · d))
    """

    def __init__(self, params: dict,
                 image_shape: tuple = (480, 640)):
        self.params      = params
        self.image_shape = image_shape   # (H, W)

        # 현재 카메라/조명 상태 (arm_controller가 업데이트)
        self._cam_pos   = np.array([0.0, 0.0, 1.0])
        self._light_pos = np.array([0.2, 0.0, 0.8])

        # 합성 장면: 평면 (z=0)
        self._scene_depth = 1.0   # 미터

    def set_pose(self, cam_pos: np.ndarray, light_pos: np.ndarray):
        """arm_controller에서 pose가 바뀔 때 호출된다."""
        self._cam_pos   = cam_pos.copy()
        self._light_pos = light_pos.copy()
        self._scene_depth = abs(cam_pos[2])

    def get_rgb(self) -> np.ndarray:
        """
        OceanSim Eq.1을 적용한 합성 RGB 이미지를 반환한다.

        Returns:
            (H, W, 3) uint8
        """
        H, W = self.image_shape
        d = self._scene_depth

        beta_attn = self.params['beta_attn']
        beta_bs   = self.params['beta_bs']
        B_inf     = self.params['B_inf']

        # 균일한 장면 복사율 J (albedo = 0.5 가정)
        J = 0.5 * np.ones((H, W), dtype=np.float64)

        # OceanSim Eq.1
        I_c = J * np.exp(-beta_attn * d) + B_inf * (1.0 - np.exp(-beta_bs * d))

        # 노이즈 추가
        I_c += np.random.normal(0, 0.005, I_c.shape)
        I_c  = np.clip(I_c, 0.0, 1.0)

        # RGB 변환 (수중 청록색 색조 적용)
        rgb = np.zeros((H, W, 3), dtype=np.float64)
        rgb[:, :, 0] = I_c * 0.6   # R 채널 감쇠 (수중 적색 흡수)
        rgb[:, :, 1] = I_c * 0.9   # G
        rgb[:, :, 2] = I_c * 1.0   # B

        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    def get_depth(self) -> np.ndarray:
        """
        합성 깊이 맵을 반환한다.

        Returns:
            (H, W) float64 (미터)
        """
        H, W = self.image_shape
        base  = self._scene_depth * np.ones((H, W), dtype=np.float64)
        noise = np.random.normal(0, 0.01, base.shape)
        return np.clip(base + noise, 0.01, None)

    def get_sonar_pointcloud(self) -> np.ndarray:
        """
        평면 장면의 합성 소나 포인트클라우드를 반환한다.

        Returns:
            (P, 3) float64
        """
        n_pts = 500
        xs = np.random.uniform(-1.0, 1.0, n_pts)
        ys = np.random.uniform(-1.0, 1.0, n_pts)
        zs = np.random.normal(0.0, 0.02, n_pts)   # 평면 z=0에 노이즈
        return np.column_stack([xs, ys, zs])

    def get_water_params(self) -> dict:
        return self.params


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _load_water_params(config_path: str) -> dict:
    """water_params.json을 로드하여 dict로 반환한다."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없음: {config_path}")

    with open(path, 'r') as f:
        raw = json.load(f)

    # _comment 키 제거
    return {k: v for k, v in raw.items() if not k.startswith('_')}


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    config_path = os.path.join(os.path.dirname(__file__),
                               '..', 'config', 'water_params.json')

    bridge = create_bridge(config_path, synthetic=True)

    bridge.set_pose(
        cam_pos=np.array([0.0, 0.0, 1.0]),
        light_pos=np.array([0.2, 0.0, 0.8])
    )

    rgb   = bridge.get_rgb()
    depth = bridge.get_depth()
    pts   = bridge.get_sonar_pointcloud()

    print(f"RGB shape:   {rgb.shape},  dtype: {rgb.dtype}")
    print(f"Depth shape: {depth.shape}, mean: {depth.mean():.3f} m")
    print(f"Sonar pts:   {pts.shape}")
    print(f"Water params: {bridge.get_water_params()}")
