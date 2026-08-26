"""
step_1_NBV/env/env_utils.py를 이식 — TSDF voxel화/좌표변환/메쉬 로딩 로직은
로봇 종류와 무관해 대부분 무수정 재사용. 유일한 실질 변경은 `_build_cam_pose()`:
step_1은 카메라가 독립 `sensor_rig`(RigidObject)에 실려 있어 `self.cam_pos`/
`self.cam_orient`(=sensor_rig의 root pose)를 읽었지만, step_3는 카메라가
로봇 동체에 물리적으로 고정 부착된 IsaacLab Camera 센서라 센서 자체의
`data.pos_w`/`data.quat_w`(오프셋+회전까지 반영된 실제 카메라 월드 pose)를
직접 읽는다 — 로봇이 이동/회전하면 자동으로 갱신되므로 별도 추적 로직 불필요.

`_look_at_quat()`은 그대로 재사용 — step_1처럼 매 스텝 카메라를 순간이동시키는
용도가 아니라, `env.py::_reset_idx()`에서 로봇의 "초기 스폰 자세"(구면좌표
목표점에서 rock을 바라보는 방향)를 계산하는 데 쓰인다.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Sequence
import omni.usd
import cv2


class EnvUtilsMixin:

    def _look_at_quat(self, from_pos: torch.Tensor, to_pos: torch.Tensor) -> torch.Tensor:
        """
        from_pos (N,3) → to_pos (N,3) 를 바라보는 쿼터니언 [w,x,y,z] (N,4) 반환.
        body frame: +X = forward, +Y = left, +Z = up.

        Shepperd method 4분기 완전 구현으로 수치 안정성 확보.
        """
        N = from_pos.shape[0]

        forward = to_pos - from_pos
        forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-8)

        up = torch.tensor([[0., 0., 1.]], device=self.device).expand(N, -1)
        dot = (forward * up).sum(dim=-1, keepdim=True).abs()
        fallback = torch.tensor([[0., 1., 0.]], device=self.device).expand(N, -1)
        up = torch.where(dot > 1.0 - 1e-6, fallback, up)

        right = torch.linalg.cross(forward, up)
        right = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
        up_ortho = torch.linalg.cross(right, forward)
        up_ortho = up_ortho / (up_ortho.norm(dim=-1, keepdim=True) + 1e-8)

        R = torch.stack([forward, -right, up_ortho], dim=-1)  # (N, 3, 3)

        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        w = torch.zeros(N, device=self.device)
        x = torch.zeros(N, device=self.device)
        y = torch.zeros(N, device=self.device)
        z = torch.zeros(N, device=self.device)

        m0 = trace > 0
        if m0.any():
            s = 0.5 / torch.sqrt((trace[m0] + 1.0).clamp(min=1e-8))
            w[m0] = 0.25 / s
            x[m0] = (R[m0, 2, 1] - R[m0, 1, 2]) * s
            y[m0] = (R[m0, 0, 2] - R[m0, 2, 0]) * s
            z[m0] = (R[m0, 1, 0] - R[m0, 0, 1]) * s

        m1 = (~m0) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
        if m1.any():
            s = 2.0 * torch.sqrt((1.0 + R[m1, 0, 0] - R[m1, 1, 1] - R[m1, 2, 2]).clamp(min=1e-8))
            w[m1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s
            x[m1] = 0.25 * s
            y[m1] = (R[m1, 0, 1] + R[m1, 1, 0]) / s
            z[m1] = (R[m1, 0, 2] + R[m1, 2, 0]) / s

        m2 = (~m0) & (~m1) & (R[:, 1, 1] > R[:, 2, 2])
        if m2.any():
            s = 2.0 * torch.sqrt((1.0 + R[m2, 1, 1] - R[m2, 0, 0] - R[m2, 2, 2]).clamp(min=1e-8))
            w[m2] = (R[m2, 0, 2] - R[m2, 2, 0]) / s
            x[m2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s
            y[m2] = 0.25 * s
            z[m2] = (R[m2, 1, 2] + R[m2, 2, 1]) / s

        m3 = (~m0) & (~m1) & (~m2)
        if m3.any():
            s = 2.0 * torch.sqrt((1.0 + R[m3, 2, 2] - R[m3, 0, 0] - R[m3, 1, 1]).clamp(min=1e-8))
            w[m3] = (R[m3, 1, 0] - R[m3, 0, 1]) / s
            x[m3] = (R[m3, 0, 2] + R[m3, 2, 0]) / s
            y[m3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s
            z[m3] = 0.25 * s

        quat = torch.stack([w, x, y, z], dim=-1)

        return quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)

    def _quat_to_rot_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        R = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ], dim=-1).reshape(-1, 3, 3)

        return R

    def _build_cam_pose(self) -> torch.Tensor:
        """카메라 extrinsic (world → OpenCV camera frame, 4x4).

        step_1과의 유일한 차이: `self.cam_pos`/`self.cam_orient`(독립 sensor_rig
        pose) 대신 카메라 센서 자체의 `data.pos_w`/`data.quat_w`를 직접 읽는다
        — 로봇에 물리적으로 고정 부착돼 있어 오프셋+회전까지 반영된 실제 월드
        pose가 매 스텝 자동 갱신된다.
        """
        from isaaclab.utils.math import quat_apply

        N = self.num_envs
        # 카메라 pose는 센서가 보고하는 `data.pos_w`/`data.quat_w_world`를 쓰지 않고
        # **로봇 root pose에서 직접 유도**한다 (2026-08-26 버그 수정).
        #
        # 실측: 카메라 prim이 Robot의 자식인데도 `data.pos_w`가 초기 스폰 pose에
        # 고정돼 갱신되지 않았다(로봇이 (2.47,0,-2.53)에 있는데 카메라는
        # (0.157,0.005,5.068) = 초기 pose ⊕ 오프셋을 보고). 그 결과 voxel이
        # 카메라 평면 위로 변환돼(cam_z≈±0.2) 투영좌표가 10^6 규모로 발산,
        # `in_bounds=0/328`이 되어 TSDF에 아무것도 융합되지 않았다.
        #
        # 단 **렌더링 자체는 정상**이다(결정마다 depth가 바뀜 = 실제 카메라는
        # 로봇을 따라감). 즉 결함은 pose *읽기*에만 있으므로, 렌더 경로를
        # 건드리는 `set_world_poses()` 대신 읽기만 교정한다 — 카메라 prim이
        # Robot의 자식이라 pose를 직접 쓰면 이중 변환 위험이 있다.
        #
        # `scene_cfg.py`가 선언한 오프셋(위치=_CAMERA_FRAME_POS, 회전=항등)과
        # 정확히 일치해야 한다 — 그쪽을 바꾸면 여기도 같이 바꿀 것.
        robot_pos = self._robot.data.root_pos_w          # (E,3)
        robot_quat = self._robot.data.root_quat_w        # (E,4) [w,x,y,z]
        cam_pos_w = robot_pos + quat_apply(
            robot_quat, self._camera_offset_pos.unsqueeze(0).expand(N, -1)
        )
        cam_quat_w = robot_quat   # scene_cfg의 offset.rot이 항등이므로 그대로

        R_wc = self._quat_to_rot_matrix(cam_quat_w)   # (E,3,3) cam→world
        R_cw = R_wc.transpose(1, 2)                    # (E,3,3) world→cam
        t_cw = -torch.bmm(R_cw, cam_pos_w.unsqueeze(-1)).squeeze(-1)  # (E,3)

        # world → body(X=fwd,Y=left,Z=up) → OpenCV(X=right,Y=down,Z=depth) 축 재배치
        # (step_1_NBV/env/env_utils.py의 P 행렬과 동일 — 카메라 body-frame 컨벤션
        # 자체는 UWCameraCfg.offset의 convention="world"로 그대로 유지되므로 불변)
        P = torch.tensor([
            [0., -1., 0.],
            [0., 0., -1.],
            [1., 0., 0.],
        ], device=self.device)
        P_batch = P.unsqueeze(0).expand(N, -1, -1)

        R_std = torch.bmm(P_batch, R_cw)
        t_std = torch.bmm(P_batch, t_cw.unsqueeze(-1)).squeeze(-1)

        pose = torch.eye(4, device=self.device).unsqueeze(0).expand(N, -1, -1).clone()
        pose[:, :3, :3] = R_std
        pose[:, :3, 3] = t_std
        return pose

    def _voxelize_gt_mesh(self, env_ids: Sequence[int]) -> None:
        vox = self.cfg.tsdf.voxel_size
        Nx, Ny, Nz = self.cfg.tsdf.vol_dim

        for env_id in env_ids:
            verts, faces = self._load_mesh(env_id)

            r1 = np.random.rand(len(faces), 1).astype(np.float32)
            r2 = np.random.rand(len(faces), 1).astype(np.float32)
            a = 1.0 - np.sqrt(r1)
            b = np.sqrt(r1) * (1.0 - r2)
            c = np.sqrt(r1) * r2

            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            pts = a * v0 + b * v1 + c * v2

            obj_min = pts.min(axis=0)
            obj_max = pts.max(axis=0)
            center = (obj_min + obj_max) / 2.0
            half_ext = np.array([Nx, Ny, Nz], dtype=np.float32) * vox / 2.0
            origin = center - half_ext

            self._vol_origin[env_id] = torch.tensor(origin, device=self.device)

            pts_t = torch.tensor(pts, device=self.device)
            orig_t = self._vol_origin[env_id]
            idx = ((pts_t - orig_t) / vox).long()

            in_bounds = (
                (idx[:, 0] >= 0) & (idx[:, 0] < Nx) &
                (idx[:, 1] >= 0) & (idx[:, 1] < Ny) &
                (idx[:, 2] >= 0) & (idx[:, 2] < Nz)
            )
            idx = idx[in_bounds]

            surf_vol = torch.zeros(Nx, Ny, Nz, dtype=torch.bool, device=self.device)
            surf_vol[idx[:, 0], idx[:, 1], idx[:, 2]] = True

            self._total_surf_voxels[env_id] = surf_vol.sum().float().clamp(min=1.0)
            self._tsdf_vol[env_id] = torch.zeros(Nx, Ny, Nz, device=self.device)
            self._weight_vol[env_id] = torch.zeros(Nx, Ny, Nz, device=self.device)
            self._surf_vol[env_id] = surf_vol

    def _load_mesh(self, env_id: int):
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim_path = f"/World/envs/env_{env_id}/Object"
        root_prim = stage.GetPrimAtPath(prim_path)

        mesh_prim = None
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdGeom.Mesh):
                mesh_prim = UsdGeom.Mesh(prim)
                break

        if mesh_prim is None:
            raise RuntimeError(f"No UsdGeom.Mesh found under: {prim_path}")

        points = mesh_prim.GetPointsAttr().Get()
        verts = np.array(points, dtype=np.float32)

        indices = np.array(mesh_prim.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        counts = np.array(mesh_prim.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        faces = self._triangulate(indices, counts)

        xform_cache = UsdGeom.XformCache()
        world_xform = xform_cache.GetLocalToWorldTransform(mesh_prim.GetPrim())
        ones = np.ones((len(verts), 1), dtype=np.float32)
        verts_h = np.hstack([verts, ones])
        mat = np.array(world_xform).reshape(4, 4).T.astype(np.float32)
        verts = (verts_h @ mat.T)[:, :3]

        stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
        verts = verts * float(stage_mpu)

        return verts, faces

    def _triangulate(self, indices: np.ndarray, counts: np.ndarray) -> np.ndarray:
        triangles = []
        offset = 0
        for n in counts:
            v0 = indices[offset]
            for j in range(1, n - 1):
                triangles.append([v0, indices[offset + j], indices[offset + j + 1]])
            offset += n
        return np.array(triangles, dtype=np.int64)

    def _randomize_rock_pose(self, env_ids: Sequence[int]) -> None:
        from pxr import UsdGeom, Gf

        stage = omni.usd.get_context().get_stage()

        for env_id in env_ids:
            prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Object")
            if not prim.IsValid():
                continue

            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()

            xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -3.0))

            yaw = float(np.random.uniform(0.0, 360.0))
            pitch = float(np.random.uniform(-30.0, 30.0))
            roll = float(np.random.uniform(-30.0, 30.0))
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(roll, pitch, yaw))

            scale = float(np.random.uniform(0.8, 1.5))
            xformable.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
