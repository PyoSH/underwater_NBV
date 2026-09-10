"""sim <-> 배포 공유 기하·TSDF. **양쪽이 같은 코드를 쓰게 하는 순수 함수.** 2026-09-10.

왜 분리하는가 (DEPLOY_3WEEK_PLAN.md §6): 배포 TSDF 노드가 `env_reward` 의 융합을
**복사하면 드리프트한다**. step_2 `sync_vendor.sh` 가 복제본 드리프트 + 정규식 버그로
`_BROV2_YAML_PATH` 를 깨뜨린 전례가 있다. 그래서 여기에 한 벌만 두고 양쪽이 import 한다.

의존은 torch 뿐이다 — isaaclab/omni 를 끌어오지 않으므로 ROS 컨테이너(torch 2.13 CPU)
에서도 그대로 돈다.

좌표 규약 (`env_utils._build_cam_pose` 와 동일):
    cam_pose = 4x4,  **world -> OpenCV 카메라 프레임** (extrinsic)
    body FLU(X 전방, Y 좌, Z 상) -> OpenCV(X 우, Y 하, Z 깊이) 재배치는
    P = [[0,-1,0],[0,0,-1],[1,0,0]]

깊이 규약 — **`fuse_depth` 는 언제나 z-depth 를 받는다.**
    융합식이 `sdf = depth - vox_z` 이므로 광축 방향 거리(z-depth)여야 한다.
    유클리드 거리(광학중심까지의 슬랜트 거리)를 넣으면 광축에서 벗어난 화소가
    1/cos(theta) 배로 과대 측정된다.

    두 양이 필요한 이유가 따로 있다:
      UW 렌더 감쇠 exp(-d*ac) 는 **광로 길이 = 유클리드**가 물리적으로 옳다
        (`sensors/UWCamera/UW_Camera_parallel.py`).
      TSDF 융합은 **z-depth** 가 옳다.
    Isaac 은 `UWCamera` 도입 커밋(1cb684b "dooomed :(", 2026-04-23)에서 한 키
    (`distance_to_camera`)를 두 용도가 나눠 쓰게 됐다. annotator 를 하나 더 켜는
    대신 `euclidean_to_z_depth()` 로 화소별 변환한다 — 렌더 비용이 0 이고,
    이 프로젝트는 이미 렌더 자원 한계에 두 번 부딪혔다(계획서 §3).
    Gazebo `rgbd_camera` 의 `depth_image` 는 처음부터 z-depth 라 변환이 필요 없다.
"""
from __future__ import annotations

import torch

# body FLU -> OpenCV optical. body +X -> optical +Z, body +Y -> optical -X, body +Z -> optical -Y
FLU_TO_OPTICAL = ((0.0, -1.0, 0.0),
                  (0.0, 0.0, -1.0),
                  (1.0, 0.0, 0.0))


def build_voxel_grid(vol_dim, voxel_size: float, *, device, dtype=torch.float32):
    """볼륨 로컬 좌표의 voxel **중심** 격자 (N_vox, 3).

    반 voxel 보정이 들어 있다 — 이게 빠지면 볼륨이 반 칸 밀린다
    (2026-09-03 `_voxel_offset` 에서 실제로 났던 버그).
    """
    nx, ny, nz = vol_dim
    xi = torch.arange(nx, device=device, dtype=dtype)
    yi = torch.arange(ny, device=device, dtype=dtype)
    zi = torch.arange(nz, device=device, dtype=dtype)
    gx, gy, gz = torch.meshgrid(xi, yi, zi, indexing="ij")
    half = voxel_size / 2.0
    return torch.stack([gx.flatten() * voxel_size + half,
                        gy.flatten() * voxel_size + half,
                        gz.flatten() * voxel_size + half], dim=-1)


def quat_wxyz_to_rot(quat: torch.Tensor) -> torch.Tensor:
    """(E,4) [w,x,y,z] -> (E,3,3) 회전행렬 (body -> world)."""
    q = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def camera_extrinsic(cam_pos_w: torch.Tensor, cam_quat_w: torch.Tensor) -> torch.Tensor:
    """카메라 world pose(FLU) -> extrinsic 4x4 (world -> OpenCV).

    `cam_quat_w` 는 [w,x,y,z], 카메라 body 축이 FLU 라고 본다
    (Isaac 은 `UWTiledCameraCfg.offset.convention="world"` + 항등 회전,
     Gazebo 는 센서 프레임이 +X 전방/+Y 좌/+Z 상 — 둘이 같은 규약이다).
    """
    e = cam_pos_w.shape[0]
    dev, dt = cam_pos_w.device, cam_pos_w.dtype
    r_wc = quat_wxyz_to_rot(cam_quat_w).to(dt)          # cam(FLU) -> world
    r_cw = r_wc.transpose(1, 2)                          # world -> cam(FLU)
    t_cw = -torch.bmm(r_cw, cam_pos_w.unsqueeze(-1)).squeeze(-1)
    p = torch.tensor(FLU_TO_OPTICAL, device=dev, dtype=dt).unsqueeze(0).expand(e, -1, -1)
    pose = torch.eye(4, device=dev, dtype=dt).unsqueeze(0).repeat(e, 1, 1)
    pose[:, :3, :3] = torch.bmm(p, r_cw)
    pose[:, :3, 3] = torch.bmm(p, t_cw.unsqueeze(-1)).squeeze(-1)
    return pose


def fuse_depth(depth_img: torch.Tensor, cam_pose: torch.Tensor,
               tsdf_vol: torch.Tensor, weight_vol: torch.Tensor,
               intrinsics: torch.Tensor, vol_origin: torch.Tensor,
               voxel_size: float, trunc_margin: float, vol_dim,
               vox_local: torch.Tensor | None = None):
    """depth 한 장씩을 배치 TSDF 볼륨에 융합한다. 입력 볼륨은 바꾸지 않는다.

    인자
        depth_img   (E,H,W) 또는 (E,H,W,1)  — **z-depth**(광축 방향 거리).
                    유클리드 거리라면 먼저 `euclidean_to_z_depth()` 를 통과시킬 것.
        cam_pose    (E,4,4)  world -> OpenCV extrinsic
        tsdf_vol    (E,Nx,Ny,Nz)
        weight_vol  (E,Nx,Ny,Nz)
        intrinsics  (E,3,3)
        vol_origin  (E,3)    볼륨 원점(월드)
    반환
        (tsdf_vol, weight_vol) 새 텐서

    `env_reward._fuse_depth` 원본과 한 곳 다르다: 원본은 `in_bounds` 판정에 카메라
    센서의 H/W 를, flatten 에는 depth_img 의 H/W 를 썼다. 둘은 Isaac 에서 항상 같아
    동작은 동일하지만, 여기서는 **depth_img 하나로 통일**한다(다르면 원본이 버그다).
    """
    if depth_img.dim() == 4:
        depth_img = depth_img.squeeze(-1)
    e = depth_img.shape[0]
    h, w = depth_img.shape[1], depth_img.shape[2]
    nx, ny, nz = vol_dim
    n_vox = nx * ny * nz

    if vox_local is None:
        vox_local = build_voxel_grid(vol_dim, voxel_size,
                                     device=depth_img.device, dtype=depth_img.dtype)

    fx = intrinsics[:, 0, 0].unsqueeze(1)
    fy = intrinsics[:, 1, 1].unsqueeze(1)
    cx = intrinsics[:, 0, 2].unsqueeze(1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)

    vox_world = vox_local.unsqueeze(0) + vol_origin.unsqueeze(1)        # (E,N,3)
    r = cam_pose[:, :3, :3]
    t = cam_pose[:, :3, 3]
    vox_cam = torch.bmm(r, vox_world.permute(0, 2, 1)) + t.unsqueeze(-1)
    vox_cam = vox_cam.permute(0, 2, 1)                                   # (E,N,3)

    vox_z = vox_cam[..., 2]
    valid_z = vox_z > 1e-4
    proj_u = (fx * vox_cam[..., 0] / vox_z.clamp(min=1e-4) + cx).long()
    proj_v = (fy * vox_cam[..., 1] / vox_z.clamp(min=1e-4) + cy).long()

    in_bounds = valid_z & (proj_u >= 0) & (proj_u < w) & (proj_v >= 0) & (proj_v < h)

    depth_flat = depth_img.reshape(e, -1)
    pixel_idx = proj_v.clamp(0, h - 1) * w + proj_u.clamp(0, w - 1)
    sampled_depth = torch.gather(depth_flat, 1, pixel_idx)

    sdf = sampled_depth - vox_z
    tsdf = (sdf / trunc_margin).clamp(-1.0, 1.0)
    update = in_bounds & (sdf >= -trunc_margin) & (sdf <= trunc_margin)

    w_old = weight_vol.reshape(e, n_vox)
    t_old = tsdf_vol.reshape(e, n_vox)
    w_new = w_old + update.float()
    t_new = torch.where(update, (t_old * w_old + tsdf) / w_new.clamp(min=1e-8), t_old)
    return t_new.reshape(e, nx, ny, nz), w_new.reshape(e, nx, ny, nz)


# ── NBV 구면 좌표 (sim 과 배포가 반드시 같아야 하는 변환) ────────────────────
# 규약(§12.0): 구면 중심 = 물체 **바닥면 원점**, (theta, phi, psi) 가 가리키는 점은
# **base_link**(카메라가 아니다). `env.py::_pre_physics_step` / `_get_observations` 원본.

def offset_from_spherical(theta: torch.Tensor, phi: torch.Tensor,
                          psi: torch.Tensor) -> torch.Tensor:
    """(theta, phi, psi) -> 구면 중심 기준 직교 오프셋 (E,3). `env.py:348` 과 동일."""
    return torch.stack([psi * torch.sin(phi) * torch.cos(theta),
                        psi * torch.sin(phi) * torch.sin(theta),
                        psi * torch.cos(phi)], dim=-1)


def spherical_from_offset(offset: torch.Tensor):
    """직교 오프셋 -> (theta, phi, psi). `env.py:444-446` 의 역변환과 동일.

    theta 는 [0, 2pi) 로 감는다 — 원본이 `% (2*pi)` 를 쓴다.
    """
    psi = offset.norm(dim=-1).clamp_min(1e-6)
    phi = torch.acos((offset[..., 2] / psi).clamp(-1.0, 1.0))
    theta = torch.atan2(offset[..., 1], offset[..., 0]) % (2 * torch.pi)
    return theta, phi, psi


def look_at_quat(from_pos: torch.Tensor, to_pos: torch.Tensor) -> torch.Tensor:
    """from -> to 를 바라보는 body FLU 자세, wxyz (E,4). `env_utils._look_at_quat` 와 동일.

    R 의 열 = [forward, -right, up_ortho],  right = forward x up_world.
    배포의 look_at heading mode 가 sim 과 **한 도라도 어긋나면** 관측이 달라진다.
    """
    fwd = to_pos - from_pos
    fwd = fwd / (fwd.norm(dim=-1, keepdim=True) + 1e-8)
    up = torch.tensor([[0.0, 0.0, 1.0]], device=fwd.device, dtype=fwd.dtype).expand_as(fwd)
    dot = (fwd * up).sum(dim=-1, keepdim=True).abs()
    fallback = torch.tensor([[0.0, 1.0, 0.0]], device=fwd.device, dtype=fwd.dtype).expand_as(fwd)
    up = torch.where(dot > 1.0 - 1e-6, fallback, up)
    right = torch.linalg.cross(fwd, up)
    right = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
    up_o = torch.linalg.cross(right, fwd)
    up_o = up_o / (up_o.norm(dim=-1, keepdim=True) + 1e-8)
    r = torch.stack([fwd, -right, up_o], dim=-1)
    return rot_to_quat_wxyz(r)


def rot_to_quat_wxyz(r: torch.Tensor) -> torch.Tensor:
    """(E,3,3) -> (E,4) [w,x,y,z]. Shepperd 4분기 (수치 안정)."""
    e = r.shape[0]
    q = torch.zeros(e, 4, device=r.device, dtype=r.dtype)
    tr = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    c0 = tr > 0
    c1 = (~c0) & (r[:, 0, 0] >= r[:, 1, 1]) & (r[:, 0, 0] >= r[:, 2, 2])
    c2 = (~c0) & (~c1) & (r[:, 1, 1] >= r[:, 2, 2])
    c3 = ~(c0 | c1 | c2)
    for mask, idx in ((c0, -1), (c1, 0), (c2, 1), (c3, 2)):
        if not bool(mask.any()):
            continue
        m = r[mask]
        if idx == -1:
            s = torch.sqrt(tr[mask] + 1.0) * 2
            q[mask] = torch.stack([0.25 * s,
                                   (m[:, 2, 1] - m[:, 1, 2]) / s,
                                   (m[:, 0, 2] - m[:, 2, 0]) / s,
                                   (m[:, 1, 0] - m[:, 0, 1]) / s], dim=-1)
        else:
            j, k = (idx + 1) % 3, (idx + 2) % 3
            s = torch.sqrt(1.0 + m[:, idx, idx] - m[:, j, j] - m[:, k, k]) * 2
            out = torch.zeros(m.shape[0], 4, device=r.device, dtype=r.dtype)
            out[:, 0] = (m[:, k, j] - m[:, j, k]) / s
            out[:, 1 + idx] = 0.25 * s
            out[:, 1 + j] = (m[:, j, idx] + m[:, idx, j]) / s
            out[:, 1 + k] = (m[:, k, idx] + m[:, idx, k]) / s
            q[mask] = out
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def z_depth_scale(height: int, width: int, intrinsics: torch.Tensor) -> torch.Tensor:
    """유클리드 거리 -> z-depth 변환 계수 (E 또는 1, H, W).

    화소 (u,v) 의 정규화 광선은 ((u-cx)/fx, (v-cy)/fy, 1) 이므로
        d_euclid = z * sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)
    따라서 z = d_euclid * scale,  scale = 1/sqrt(...).

    intrinsic 이 전 env 동일하면 (1,H,W) 를 반환해 브로드캐스트로 쓴다 — 96 env
    에서 (E,H,W) 를 들면 30 MB 를 그냥 버리는 셈이다.
    """
    dev, dt = intrinsics.device, intrinsics.dtype
    same = bool(torch.allclose(intrinsics, intrinsics[:1].expand_as(intrinsics)))
    k = intrinsics[:1] if same else intrinsics
    fx = k[:, 0, 0].view(-1, 1, 1)
    fy = k[:, 1, 1].view(-1, 1, 1)
    cx = k[:, 0, 2].view(-1, 1, 1)
    cy = k[:, 1, 2].view(-1, 1, 1)
    u = torch.arange(width, device=dev, dtype=dt).view(1, 1, -1)
    v = torch.arange(height, device=dev, dtype=dt).view(1, -1, 1)
    a = (u - cx) / fx
    b = (v - cy) / fy
    return torch.rsqrt(1.0 + a * a + b * b)


def euclidean_to_z_depth(depth_img: torch.Tensor, intrinsics: torch.Tensor,
                         scale: torch.Tensor | None = None) -> torch.Tensor:
    """유클리드 거리 영상 -> z-depth 영상. `scale` 을 넘기면 재계산하지 않는다."""
    squeezed = depth_img.dim() == 4
    d = depth_img.squeeze(-1) if squeezed else depth_img
    if scale is None:
        scale = z_depth_scale(d.shape[1], d.shape[2], intrinsics)
    z = d * scale
    return z.unsqueeze(-1) if squeezed else z


def vox_actor_channels(tsdf_vol: torch.Tensor, weight_vol: torch.Tensor,
                       quality_vol: torch.Tensor | None = None,
                       q_star: torch.Tensor | None = None) -> torch.Tensor:
    """정책 관측 voxel 3채널 (E,3,Nx,Ny,Nz). `env.py::_get_vox_actor` 와 동일.

    ch0 unknown   weight == 0
    ch1 free      weight > 0 and tsdf > 0
    ch2 occupied  품질 모드면 quality/q_star 를 clamp(0,1), 아니면 이진 점유
    """
    observed = weight_vol > 0
    if quality_vol is not None and q_star is not None:
        ch2 = (quality_vol / q_star).clamp(0.0, 1.0)
    else:
        ch2 = (observed & (tsdf_vol <= 0)).float()
    return torch.stack([
        (~observed).float(),
        (observed & (tsdf_vol > 0)).float(),
        ch2,
    ], dim=1)
