"""
arm_controller.py
-----------------
UR5e 듀얼암의 end-effector를 목표 위치로 이동시키고
Isaac Sim USD prim을 동기화한다.

구성:
  왼팔  (LeftArm):  카메라 (UnderwaterCamera prim)
  오른팔 (RightArm): 조명   (UsdLux.SphereLight prim)

Isaac Sim 없을 때:
  pose를 NumPy 배열로만 관리 (시뮬레이션 상태 추적)
  oceansim_bridge.SyntheticBridge.set_pose()와 연동

Isaac Sim 있을 때:
  1. LulaKinematicsSolver로 IK 계산
  2. Articulation API로 joint 이동
  3. USD API로 카메라/조명 prim 위치 업데이트
"""

import numpy as np


# ── Isaac Sim 환경 확인 ───────────────────────────────────────────────────────
try:
    import omni.usd
    from pxr import UsdGeom, UsdLux, Gf
    from omni.isaac.core.articulations import Articulation
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False


class ArmController:
    """
    UR5e 듀얼암 컨트롤러.

    Isaac Sim 환경이면 실제 USD prim을 제어하고,
    없으면 pose 상태만 NumPy로 관리한다.
    """

    def __init__(self,
                 cam_prim_path:   str = "/World/ROV/LeftArm/tool0/Camera",
                 light_prim_path: str = "/World/ROV/RightArm/tool0/Light",
                 left_arm_path:   str = "/World/ROV/LeftArm",
                 right_arm_path:  str = "/World/ROV/RightArm",
                 bridge=None):
        """
        Args:
            cam_prim_path:   카메라 USD prim 경로
            light_prim_path: 조명 USD prim 경로
            left_arm_path:   왼팔 Articulation prim 경로
            right_arm_path:  오른팔 Articulation prim 경로
            bridge:          SyntheticBridge 인스턴스 (Isaac Sim 없을 때 사용)
        """
        self.cam_prim_path   = cam_prim_path
        self.light_prim_path = light_prim_path
        self.left_arm_path   = left_arm_path
        self.right_arm_path  = right_arm_path
        self.bridge          = bridge

        # 현재 pose 상태 (NumPy)
        self._cam_pos   = np.array([0.0, 0.0, 1.0])
        self._light_pos = np.array([0.1, 0.0, 0.8])

        # Isaac Sim 핸들
        self._left_arm  = None
        self._right_arm = None

        if ISAAC_SIM_AVAILABLE:
            self._setup_isaac_sim()

    def _setup_isaac_sim(self):
        """Isaac Sim Articulation 핸들을 초기화한다."""
        self._left_arm  = Articulation(self.left_arm_path)
        self._right_arm = Articulation(self.right_arm_path)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def move_to(self, cam_pos: np.ndarray, light_pos: np.ndarray) -> bool:
        """
        카메라와 조명을 목표 위치로 이동한다.

        Args:
            cam_pos:   (3,) 목표 카메라 위치 (미터, 월드 좌표계)
            light_pos: (3,) 목표 조명 위치 (미터, 월드 좌표계)

        Returns:
            bool: 이동 성공 여부 (IK 실패 시 False)
        """
        if ISAAC_SIM_AVAILABLE:
            success = self._move_isaac_sim(cam_pos, light_pos)
        else:
            success = True   # Synthetic 모드: 항상 성공

        if success:
            self._cam_pos   = cam_pos.copy()
            self._light_pos = light_pos.copy()

            # SyntheticBridge에 pose 전달
            if self.bridge is not None:
                self.bridge.set_pose(cam_pos, light_pos)

        return success

    def get_cam_pos(self) -> np.ndarray:
        """현재 카메라 위치를 반환한다."""
        return self._cam_pos.copy()

    def get_light_pos(self) -> np.ndarray:
        """현재 조명 위치를 반환한다."""
        return self._light_pos.copy()

    def get_baseline(self) -> float:
        """현재 카메라-조명 baseline 거리를 반환한다."""
        return float(np.linalg.norm(self._cam_pos - self._light_pos))

    # ── Isaac Sim 내부 구현 ───────────────────────────────────────────────────

    def _move_isaac_sim(self, cam_pos: np.ndarray,
                         light_pos: np.ndarray) -> bool:
        """
        Isaac Sim: IK 계산 → joint 이동 → USD prim 업데이트.

        Returns:
            bool: IK 성공 여부
        """
        # 왼팔 IK (카메라)
        cam_joints = self._solve_ik(self._left_arm, cam_pos)
        if cam_joints is None:
            return False

        # 오른팔 IK (조명)
        light_joints = self._solve_ik(self._right_arm, light_pos)
        if light_joints is None:
            return False

        # Joint 적용
        self._left_arm.set_joint_positions(cam_joints)
        self._right_arm.set_joint_positions(light_joints)

        # USD prim 위치 동기화
        self._update_prim_pos(self.cam_prim_path,   cam_pos)
        self._update_prim_pos(self.light_prim_path, light_pos)

        return True

    def _solve_ik(self, arm, target_pos: np.ndarray):
        """
        LulaKinematicsSolver로 IK를 풀어 joint 각도를 반환한다.

        Returns:
            np.ndarray or None: joint 각도 배열, IK 실패 시 None
        """
        # TODO: Isaac Sim 연동 시 실제 LulaKinematicsSolver 구현
        # 현재는 placeholder (모두 0 joint 반환)
        return np.zeros(6)

    def _update_prim_pos(self, prim_path: str, pos: np.ndarray):
        """
        USD API를 사용하여 prim의 위치를 업데이트한다.

        Args:
            prim_path: USD prim 경로
            pos:       (3,) 월드 좌표 위치 (미터)
        """
        stage = omni.usd.get_context().get_stage()
        prim  = stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            print(f"경고: 유효하지 않은 prim 경로: {prim_path}")
            return

        xformable = UsdGeom.Xformable(prim)

        # 기존 translate op이 있으면 재사용, 없으면 새로 추가
        ops = xformable.GetOrderedXformOps()
        translate_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()

        # 미터 → USD 단위 (Isaac Sim은 미터 기준)
        translate_op.Set(Gf.Vec3d(float(pos[0]),
                                   float(pos[1]),
                                   float(pos[2])))


# ─────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctrl = ArmController()

    print(f"초기 cam_pos:   {ctrl.get_cam_pos()}")
    print(f"초기 light_pos: {ctrl.get_light_pos()}")
    print(f"초기 baseline:  {ctrl.get_baseline():.3f} m")

    success = ctrl.move_to(
        cam_pos=np.array([0.1, 0.0, 1.2]),
        light_pos=np.array([0.25, 0.0, 1.0])
    )

    print(f"\nmove_to 성공: {success}")
    print(f"이동 후 cam_pos:   {ctrl.get_cam_pos()}")
    print(f"이동 후 light_pos: {ctrl.get_light_pos()}")
    print(f"이동 후 baseline:  {ctrl.get_baseline():.3f} m")
