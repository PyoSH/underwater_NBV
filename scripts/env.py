"""
env.py
------
OceanNBVEnv: DirectRLEnv 기반 수중 카메라·조명 탐색 RL 환경.

씬 구성 (OceanSceneCfg → InteractiveScene 자동 생성):
    {ENV_REGEX_NS}/Seafloor     - 정적 해저면
    {ENV_REGEX_NS}/Rock         - 대상 물체
    {ENV_REGEX_NS}/CameraRig    - 카메라 강체 (RigidObject)
      └── Camera                - Pinhole 카메라 (_setup_scene 에서 추가)
    {ENV_REGEX_NS}/LightRig     - 조명 강체 (RigidObject)
      └── SphereLight           - 스포트라이트 (_setup_scene 에서 추가)
omni.isaac.debug_draw           - 카메라·조명 3축 화살표 (뷰포트 오버레이, 센서 이미지 미포함)

행동 적용 방식:
    action ∈ [-1, 1]^12  → RigidObject.write_root_velocity_to_sim()
    물리 엔진이 위치를 적분 → 충돌 자동 처리
    RigidObject.data.root_pos_w / root_quat_w 로 상태 읽기

병렬 환경:
    InteractiveSceneCfg 의 {ENV_REGEX_NS} 패턴으로 num_envs 개 자동 복제
    rock_pos_w = scene.env_origins + (0, 0, -3)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

import omni.usd
from pxr import Gf, UsdGeom, UsdLux

from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils

from envCfg import OceanNBVEnvCfg


class OceanNBVEnv(DirectRLEnv):
    """카메라와 조명을 이동시키며 대상 물체를 탐색하는 병렬 RL 환경."""

    cfg: OceanNBVEnvCfg

    # ── 초기화 ───────────────────────────────────────────────────────────────

    def __init__(self, cfg: OceanNBVEnvCfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode)

        # RigidObject 핸들 (InteractiveScene 이 자동 생성)
        self._cam_rig   = self.scene["camera_rig"]
        self._light_rig = self.scene["light_rig"]

        # rock 월드 위치: env_origins + 로컬 offset (0, 0, -3)
        # sceneCfg.rock init_state.pos = (0, 0, -3) 이 env_origin 기준 상대 좌표
        rock_local = torch.tensor([0.0, 0.0, -3.0], device=self.device)
        self.rock_pos = self.scene.env_origins + rock_local  # (num_envs, 3)

        # 행동 버퍼
        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        # 방향 시각화 마커 (PhysX runtime 위치·자세를 매 스텝 직접 업데이트)
        self._setup_vis_markers()

        # 이미지 저장 카운터
        self._step_count = 0

    # ── Camera 센서 ───────────────────────────────────────────────────────────

    def _setup_camera_sensor(self) -> None:
        """Isaac Lab Camera 센서 등록.

        _add_camera_child() 에서 생성된 USD Camera 프림을 재사용 (spawn=None).
        InteractiveScene.sensors 에 등록하면 scene.update() 시 자동 갱신됨.

        렌더링 파이프라인:
            RTX renderer (OceanSim 수중 효과 포함)
                → Camera 센서 픽셀 캡처
                → data.output["rgb"]  (H×W×4 uint8, RGBA)
        """
        from isaaclab.sensors import Camera, CameraCfg

        cam_cfg = CameraCfg(
            prim_path="/World/envs/env_.*/CameraRig/Camera",
            update_period=0,                          # 매 스텝 갱신
            height=self.cfg.cam_height,
            width=self.cfg.cam_width,
            data_types=["rgb"],
            spawn=None,                               # 기존 USD 프림 사용
        )
        self._camera = Camera(cam_cfg)
        self.scene.sensors["camera"] = self._camera   # scene.update() 에 편입

    def _show_camera_realtime(self) -> None:
        """cv2.imshow 로 카메라 이미지를 실시간 표시.

        Isaac Sim GUI 모드에서만 동작. 헤드리스 모드에서는 자동 비활성화.
        센서 픽셀 데이터 (OceanSim 수중 렌더링 포함) 를 그대로 표시.
        """
        try:
            import cv2
        except ImportError:
            print("[Camera] cv2 없음 → cam_realtime_vis 비활성화")
            self.cfg.cam_realtime_vis = False
            return

        rgb = self._camera.data.output["rgb"]
        if rgb is None or rgb.shape[0] == 0:
            return

        img_np  = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)  # (H,W,3) RGB
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)           # OpenCV BGR
        try:
            cv2.imshow("OceanNBV - Camera Sensor", img_bgr)
            cv2.waitKey(1)  # 이벤트 루프 유지 (non-blocking)
        except cv2.error:
            print("[Camera] cv2.imshow 미지원 (GUI 없는 빌드) → cam_realtime_vis 비활성화")
            self.cfg.cam_realtime_vis = False

    def _save_camera_image(self) -> None:
        """env_0 카메라 이미지를 PNG 로 저장.

        data.output["rgb"]: (num_envs, H, W, 4) uint8 (RGBA)
        OceanSim 수중 렌더링이 포함된 실제 센서 출력.
        """
        import os
        import numpy as np
        from PIL import Image

        rgb = self._camera.data.output["rgb"]          # (N, H, W, 4) uint8
        if rgb is None or rgb.shape[0] == 0:
            return

        save_dir = self.cfg.cam_save_dir
        os.makedirs(save_dir, exist_ok=True)

        img_np = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)   # (H, W, 3)
        path   = os.path.join(save_dir, f"step_{self._step_count:06d}.png")
        Image.fromarray(img_np).save(path)
        print(f"[Camera] 저장: {path}")

    # ── 방향 시각화 마커 ──────────────────────────────────────────────────────

    def _setup_vis_markers(self) -> None:
        """debug_draw 인터페이스 초기화 (Isaac Sim 버전별 경로 순차 시도).

        debug_draw 는 뷰포트 오버레이에만 선을 그림.
        USD 프림을 생성하지 않으므로 카메라 센서 이미지에 영향 없음.
        모듈을 찾지 못하면 self._draw = None 으로 시각화 비활성화.
        """
        self._draw = None

        _candidates = [
            # Isaac Sim 2023.x
            lambda: __import__(
                "omni.isaac.debug_draw", fromlist=["_debug_draw"]
            )._debug_draw.acquire_debug_draw_interface(),
            # Isaac Sim 4.x (isaacsim 패키지)
            lambda: __import__(
                "isaacsim.util.debug_draw", fromlist=["_debug_draw"]
            )._debug_draw.acquire_debug_draw_interface(),
            # Kit 106+
            lambda: __import__(
                "omni.debugdraw", fromlist=["get_debug_draw_interface"]
            ).get_debug_draw_interface(),
        ]

        for _try in _candidates:
            try:
                self._draw = _try()
                return
            except (ModuleNotFoundError, AttributeError):
                continue

        print("[OceanNBVEnv] debug_draw 모듈 없음 → 방향 시각화 비활성화")

    def _update_vis_markers(self) -> None:
        """카메라·조명의 X/Y/Z 3축을 뷰포트에 선분으로 실시간 표시.

        cfg.debug_vis_env_id:
            -1 → 전체 num_envs
            ≥0 → 해당 인덱스 환경만
        """
        if self._draw is None:
            return

        self._draw.clear_lines()

        L   = 0.30   # 화살표 길이 [m]
        W   = 3.0    # 선 굵기

        # 표준 RGB 축 색상: +X=빨강, +Y=초록, +Z=파랑 (카메라·조명 공통)
        AXIS_COLS = [
            (1.0, 0.0, 0.0, 1.0),  # +X: 빨강
            (0.0, 1.0, 0.0, 1.0),  # +Y: 초록
            (0.0, 0.0, 1.0, 1.0),  # +Z: 파랑
        ]
        # 로컬 축 벡터 (body frame)
        AXES = [
            torch.tensor([[1., 0., 0.]], device=self.device),  # +X
            torch.tensor([[0., 1., 0.]], device=self.device),  # +Y
            torch.tensor([[0., 0., 1.]], device=self.device),  # +Z
        ]

        eid = self.cfg.debug_vis_env_id
        indices = [eid] if eid >= 0 else list(range(self.num_envs))

        starts, ends, colors, widths = [], [], [], []

        for i in indices:
            cp = self.cam_pos[i].cpu()
            cq = self.cam_orient[i:i+1]
            lp = self.light_pos[i].cpu()
            lq = self.light_orient[i:i+1]

            for ax, col in zip(AXES, AXIS_COLS):
                # 카메라 축
                aw = self._rotate_vec_by_quat(cq, ax)[0].cpu()
                starts.append(cp.tolist())
                ends.append((cp + aw * L).tolist())
                colors.append(col); widths.append(W)

                # 조명 축
                aw = self._rotate_vec_by_quat(lq, ax)[0].cpu()
                starts.append(lp.tolist())
                ends.append((lp + aw * L).tolist())
                colors.append(col); widths.append(W)

        if starts:
            self._draw.draw_lines(starts, ends, colors, widths)

    # ── 씬 구성 ──────────────────────────────────────────────────────────────

    def _setup_scene(self) -> None:
        """Camera, SphereLight, DirectionCone 자식 프림을 각 환경에 추가.

        OceanSceneCfg 의 에셋(Seafloor, Rock, CameraRig, LightRig)은
        InteractiveScene 이 이미 스폰했으므로 추가 자식만 생성.
        """
        stage = omni.usd.get_context().get_stage()

        for env_idx in range(self.num_envs):
            env_ns = f"/World/envs/env_{env_idx}"
            self._add_camera_child(stage, f"{env_ns}/CameraRig")
            self._add_light_children(stage, f"{env_ns}/LightRig")

        # Camera 센서를 _setup_scene() 에서 등록해야 sim.reset() (PLAY 이벤트) 전에
        # _initialize_callback 이 등록됨 → _is_initialized = True 보장
        self._setup_camera_sensor()

    def _add_camera_child(self, stage, rig_path: str) -> None:
        """CameraRig 아래에 Pinhole 카메라 프림 추가 (ROS 관례).

        ROS body frame: +X = 전방, +Y = 좌, +Z = 상
        USD Camera:     -Z = 전방 (기본값)

        USD -Z 를 body +X 에 정렬하려면 Y축 -90° 회전 필요:
          R_y(-90°) · (0,0,-1) = (1,0,0)  →  카메라가 body +X 방향을 바라봄

        위치: CameraRig X 반폭(0.05m) + 여유(0.01m) = +X 앞면 바깥
        """
        cam_prim = stage.DefinePrim(f"{rig_path}/Camera", "Camera")
        # standardize_xform_ops: XformPrimView 가 요구하는 [translate, orient, scale] 정규화
        # translation: +X 앞면 바깥 (CameraRig X 반폭 0.05m + 여유 0.01m)
        # orientation: Y축 -90°  USD -Z 전방 → body +X (ROS 전방)  [w, x, y, z]
        sim_utils.standardize_xform_ops(
            cam_prim,
            translation=(0.06, 0.0, 0.0),
            orientation=(0.7071, 0.0, -0.7071, 0.0),
        )
        cam = UsdGeom.Camera(cam_prim)
        cam.GetFocalLengthAttr().Set(24.0)                        # mm
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))    # near/far [m]

    def _add_light_children(self, stage, rig_path: str) -> None:
        """LightRig 아래에 SphereLight 추가 (ROS 관례).

        LightRig body frame: +X = 전방 (빛 방출 방향), +Y = 좌, +Z = 상
        USD SphereLight ShapingAPI: 로컬 -Z 방향으로 빛 방출 (USD 기본값)

        -Z → +X 정렬: Y축 -90° 회전
          R_y(-90°) · (0,0,-1) = (1,0,0)  →  빛이 body +X 로 방출됨
        """
        # ── SphereLight + ShapingAPI ──────────────────────────────────────
        light = UsdLux.SphereLight.Define(stage, f"{rig_path}/SphereLight")
        light.GetIntensityAttr().Set(10000000.0)
        light.GetRadiusAttr().Set(0.05)
        light.GetColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.GetShapingConeAngleAttr().Set(40.0)   # 반각 [도]
        shaping.GetShapingConeSoftnessAttr().Set(0.1)

        # Y축 -90° 회전: USD -Z 방출 → LightRig body +X (ROS 전방)
        # [w, x, y, z] = [cos(-45°), 0, sin(-45°), 0]
        # 카메라와 동일한 회전 (USD -Z 전방 관례 통일)
        # UsdGeom.Xformable(light.GetPrim()).AddOrientOp().Set(
        #     Gf.Quatf(0.7071, 0.0, -0.7071, 0.0)
        # )
        sim_utils.standardize_xform_ops(
            light.GetPrim(),
            translation = (0.0,0.0,0.0),
            orientation= (0.7071, 0.0, -0.7071, 0.0)
        )

        # # ── 방향 시각화 원뿔 (흰색, guide 용도) ──────────────────────────
        # cone_half_angle_deg = 20.0
        # cone_h = 0.15
        # cone_r = cone_h * math.tan(math.radians(cone_half_angle_deg))
        # cone = UsdGeom.Cone.Define(stage, f"{rig_path}/DirectionCone")
        # cone.GetHeightAttr().Set(cone_h)
        # cone.GetRadiusAttr().Set(cone_r)
        # cone.GetAxisAttr().Set("Z")
        # cone.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 1.0)])
        # cone.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        # cone_xf = UsdGeom.Xformable(cone)
        # cone_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cone_h / 2))

    # ── 행동 적용 ─────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """행동 클리핑 후 저장 (물리 스텝 직전)."""
        self._actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        """RigidObject GPU tensor API 로 선속도·각속도 설정.

        action[0:3]  = 카메라 선속도  [vx, vy, vz]
        action[3:6]  = 카메라 각속도  [wx, wy, wz]
        action[6:9]  = 조명 선속도    [vx, vy, vz]
        action[9:12] = 조명 각속도    [wx, wy, wz]
        """
        lv = self.cfg.max_velocity
        av = self.cfg.max_angular_velocity

        # (num_envs, 6): [lin_vel(3), ang_vel(3)]
        cam_vel = torch.cat([
            self._actions[:, 0:3] * lv,
            self._actions[:, 3:6] * av,
        ], dim=-1)
        light_vel = torch.cat([
            self._actions[:, 6:9]  * lv,
            self._actions[:, 9:12] * av,
        ], dim=-1)

        self._cam_rig.write_root_velocity_to_sim(cam_vel)
        self._light_rig.write_root_velocity_to_sim(light_vel)

    # ── 상태 프로퍼티 ─────────────────────────────────────────────────────────

    @property
    def cam_pos(self) -> torch.Tensor:
        """카메라 리그 월드 위치 (num_envs, 3)."""
        return self._cam_rig.data.root_pos_w

    @property
    def cam_orient(self) -> torch.Tensor:
        """카메라 리그 월드 자세 쿼터니언 [w,x,y,z] (num_envs, 4)."""
        return self._cam_rig.data.root_quat_w

    @property
    def light_pos(self) -> torch.Tensor:
        """조명 리그 월드 위치 (num_envs, 3)."""
        return self._light_rig.data.root_pos_w

    @property
    def light_orient(self) -> torch.Tensor:
        """조명 리그 월드 자세 쿼터니언 [w,x,y,z] (num_envs, 4)."""
        return self._light_rig.data.root_quat_w

    # ── 관측 ─────────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """
        관측 벡터 (17차원):
            cam_pos(3) | cam_orient(4) | light_pos(3) | light_orient(4) | cam_to_rock(3)
        """
        cam_to_rock = self.rock_pos - self.cam_pos
        obs = torch.cat(
            [self.cam_pos, self.cam_orient,
             self.light_pos, self.light_orient,
             cam_to_rock],
            dim=-1,
        )

        # 방향 마커 실시간 업데이트 (PhysX runtime 데이터 사용)
        if self.cfg.debug_vis:
            self._update_vis_markers()

        # 카메라 이미지 실시간 표시 / 주기적 저장
        # self._step_count += 1
        # if self.cfg.cam_realtime_vis:
        #     self._show_camera_realtime()
        # interval = self.cfg.cam_save_interval
        # if interval > 0 and self._step_count % interval == 0:
        #     self._save_camera_image()

        return {"policy": obs}

    # ── 보상 ─────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """
        거리 보상  r_dist = 1 - |dist - optimal| / optimal          (w_distance)
        방향 보상  r_dir  = dot(camera_look_world, cam_to_rock_unit) (w_direction)
        baseline  r_bl   = -1 if baseline ∉ [min, max]              (w_baseline)
        """
        cfg = self.cfg

        # ── 거리 보상 ────────────────────────────────────────────────────────
        dist_cam = torch.norm(self.cam_pos - self.rock_pos, dim=-1)
        r_dist   = (1.0 - torch.abs(dist_cam - cfg.optimal_cam_dist) / cfg.optimal_cam_dist).clamp(-1.0, 1.0)

        # ── 방향 보상 ────────────────────────────────────────────────────────
        # ROS body frame 카메라 전방: +X = (1, 0, 0)
        look_default    = torch.tensor([[1.0, 0.0, 0.0]], device=self.device).expand(self.num_envs, -1)
        cam_look_world  = self._rotate_vec_by_quat(self.cam_orient, look_default)
        cam_to_rock_dir = torch.nn.functional.normalize(self.rock_pos - self.cam_pos, dim=-1)
        r_dir = (cam_look_world * cam_to_rock_dir).sum(dim=-1)  # cosine [-1, 1]

        # ── baseline 페널티 ──────────────────────────────────────────────────
        baseline  = torch.norm(self.cam_pos - self.light_pos, dim=-1)
        out_range = (baseline < cfg.baseline_min) | (baseline > cfg.baseline_max)
        r_bl      = torch.where(out_range,
                                torch.full_like(baseline, -1.0),
                                torch.zeros_like(baseline))

        return cfg.w_distance * r_dist + cfg.w_direction * r_dir + cfg.w_baseline * r_bl

    # ── 종료 조건 ─────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        terminated: 카메라가 workspace_radius 밖으로 이탈
        truncated:  에피소드 길이 초과
        """
        dist_cam   = torch.norm(self.cam_pos - self.rock_pos, dim=-1)
        terminated = dist_cam > self.cfg.workspace_radius
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── 리셋 ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        """
        지정된 환경을 배치 리셋.
        - 카메라 위치: rock 중심 반구 내 균등 샘플링 (구면 좌표)
        - 카메라 방향: rock 을 바라보도록 look-at 쿼터니언 적용
        - 조명 위치:   카메라에서 look-at 수직 방향으로 baseline 오프셋
        """
        super()._reset_idx(env_ids)

        cfg = self.cfg
        n   = len(env_ids)

        # rock 월드 위치 (해당 env 들만)
        rock_np = self.rock_pos[env_ids].cpu().numpy()  # (n, 3)

        # ── 카메라 위치: 구면 좌표 샘플링 ─────────────────────────────────
        r     = np.random.uniform(cfg.reset_radius_min, cfg.reset_radius_max, n)
        theta = np.random.uniform(cfg.reset_theta_min,  cfg.reset_theta_max,  n)
        phi   = np.random.uniform(0.0, 2.0 * np.pi, n)

        offsets = r[:, None] * np.stack([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], axis=1)  # (n, 3)

        cam_np = rock_np + offsets  # (n, 3) — 월드 좌표

        # ── 조명 위치: look-at 방향의 수직 오프셋 ──────────────────────────
        look_dirs = rock_np - cam_np
        look_dirs /= np.linalg.norm(look_dirs, axis=1, keepdims=True) + 1e-8

        ref = np.tile([0.0, 0.0, 1.0], (n, 1))
        perp = np.cross(look_dirs, ref)
        perp_norm = np.linalg.norm(perp, axis=1)
        fallback = perp_norm < 1e-6
        if fallback.any():
            fallback_ref = np.tile([1.0, 0.0, 0.0], (fallback.sum(), 1))
            perp[fallback] = np.cross(look_dirs[fallback], fallback_ref)
        perp /= np.linalg.norm(perp, axis=1, keepdims=True) + 1e-8
        light_np = cam_np + perp * cfg.light_baseline  # (n, 3)

        # ── 카메라 look-at 쿼터니언 ─────────────────────────────────────────
        quats = np.stack([self._look_at_quat(cam_np[i], rock_np[i]) for i in range(n)])  # (n, 4)

        # ── RigidObject 상태 쓰기: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13 ──
        cam_state   = torch.zeros(n, 13, device=self.device)
        light_state = torch.zeros(n, 13, device=self.device)

        cam_state[:, 0:3] = torch.tensor(cam_np,  dtype=torch.float32, device=self.device)
        cam_state[:, 3:7] = torch.tensor(quats,   dtype=torch.float32, device=self.device)
        # 속도는 0 (zeros_like 기본값)

        light_state[:, 0:3] = torch.tensor(light_np, dtype=torch.float32, device=self.device)
        light_state[:, 3:7] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device
        ).expand(n, -1)

        env_ids_t = torch.tensor(list(env_ids), dtype=torch.int64, device=self.device)
        self._cam_rig.write_root_state_to_sim(cam_state,   env_ids=env_ids_t)
        self._light_rig.write_root_state_to_sim(light_state, env_ids=env_ids_t)

    # ── 유틸리티 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """배치 쿼터니언 곱셈 q1 ⊗ q2. 입력·출력: [..., 4] = [w, x, y, z]"""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    @staticmethod
    def _rotate_vec_by_quat(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """쿼터니언 q [w,x,y,z] 로 벡터 v [x,y,z] 를 회전한다. (N,4),(N,3) → (N,3)"""
        w   = q[:, 0:1]
        xyz = q[:, 1:]
        t   = 2.0 * torch.linalg.cross(xyz, v)
        return v + w * t + torch.linalg.cross(xyz, t)

    @staticmethod
    def _look_at_quat(cam_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        USD 카메라 기본 시선 방향(-Z)이 target 을 향하도록 하는
        쿼터니언 [w, x, y, z] 반환.
        """
        look_dir = target_pos - cam_pos
        norm = np.linalg.norm(look_dir)
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0])
        look_dir /= norm

        default_look = np.array([1.0, 0.0, 0.0])  # ROS body frame 전방: +X
        axis = np.cross(default_look, look_dir)
        axis_norm = np.linalg.norm(axis)
        dot = float(np.dot(default_look, look_dir))

        if axis_norm < 1e-6:
            if dot > 0:
                return np.array([1.0, 0.0, 0.0, 0.0])   # identity (이미 정렬)
            else:
                return np.array([0.0, 0.0, 0.0, 1.0])   # 180° around Z

        axis /= axis_norm
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        w   = float(np.cos(angle / 2))
        xyz = np.sin(angle / 2) * axis
        return np.array([w, xyz[0], xyz[1], xyz[2]])
