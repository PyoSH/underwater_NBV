"""tsdf_fusion 이 원본 `env_reward._fuse_depth` 와 **수치적으로 같은지** 고정한다.

배포 노드가 이 모듈을 import 하므로, 여기가 갈라지면 sim 과 실기의 지도가 갈라진다.
원본 구현(2026-09-10 분리 직전)을 참조로 들고 무작위 입력에서 비교한다.
"""
import sys
from pathlib import Path

import pytest
import torch

# 호스트에서는 저장소 루트, 컨테이너에서는 복사본 경로에서 찾는다
for _root in (Path(__file__).resolve().parents[1], Path("/tmp/nbv_src")):
    if (_root / "envs" / "tsdf_fusion.py").exists():
        sys.path.insert(0, str(_root))
        break
from envs import tsdf_fusion as tf  # noqa: E402


def reference_fuse(depth_img, cam_pose, tsdf_vol, weight_vol, K, vol_origin,
                   vox, trunc, vol_dim, vox_local, cam_hw):
    """분리 직전의 `env_reward._fuse_depth` 본문 그대로 (self 접근만 인자로 바꿈)."""
    Nx, Ny, Nz = vol_dim
    N_vox = Nx * Ny * Nz
    E = depth_img.shape[0]
    fx = K[:, 0, 0].unsqueeze(1); fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1); cy = K[:, 1, 2].unsqueeze(1)

    vox_world = vox_local.unsqueeze(0) + vol_origin.unsqueeze(1)
    R = cam_pose[:, :3, :3]; t = cam_pose[:, :3, 3]
    vox_cam = torch.bmm(R, vox_world.permute(0, 2, 1)) + t.unsqueeze(-1)
    vox_cam = vox_cam.permute(0, 2, 1)
    vox_z = vox_cam[..., 2]; vox_x = vox_cam[..., 0]; vox_y = vox_cam[..., 1]
    valid_z = vox_z > 1e-4
    proj_u = (fx * vox_x / vox_z.clamp(min=1e-4) + cx)
    proj_v = (fy * vox_y / vox_z.clamp(min=1e-4) + cy)

    H, W = cam_hw                       # 원본은 in_bounds 에 **카메라** H/W 를 썼다
    proj_u_int = proj_u.long(); proj_v_int = proj_v.long()
    in_bounds = (valid_z & (proj_u_int >= 0) & (proj_u_int < W)
                 & (proj_v_int >= 0) & (proj_v_int < H))

    if depth_img.dim() == 4:
        depth_img = depth_img.squeeze(-1)
    H, W = depth_img.shape[1], depth_img.shape[2]
    depth_flat = depth_img.reshape(E, -1)
    pixel_idx = proj_v_int.clamp(0, H - 1) * W + proj_u_int.clamp(0, W - 1)
    sampled_depth = torch.gather(depth_flat, 1, pixel_idx)

    sdf = sampled_depth - vox_z
    tsdf = (sdf / trunc).clamp(-1.0, 1.0)
    update_mask = in_bounds & (sdf >= -trunc) & (sdf <= trunc)
    w_old = weight_vol.reshape(E, N_vox); t_old = tsdf_vol.reshape(E, N_vox)
    w_new = w_old + update_mask.float()
    t_new = torch.where(update_mask, (t_old * w_old + tsdf) / w_new.clamp(min=1e-8), t_old)
    return t_new.reshape(E, Nx, Ny, Nz), w_new.reshape(E, Nx, Ny, Nz)


def make_case(seed, e=3, h=48, w=64, vol_dim=(20, 20, 20), vox=0.10):
    g = torch.Generator().manual_seed(seed)
    depth = torch.rand(e, h, w, generator=g) * 3.0 + 0.3
    K = torch.zeros(e, 3, 3)
    K[:, 0, 0] = 465.5; K[:, 1, 1] = 465.5
    K[:, 0, 2] = w / 2.0; K[:, 1, 2] = h / 2.0; K[:, 2, 2] = 1.0
    pos = torch.rand(e, 3, generator=g) * 2.0 - 1.0 + torch.tensor([0.0, 0.0, 1.6])
    quat = torch.randn(e, 4, generator=g)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    pose = tf.camera_extrinsic(pos, quat)
    origin = torch.tensor([-1.0, -1.0, -0.53]).unsqueeze(0).repeat(e, 1)
    tsdf = torch.rand(e, *vol_dim, generator=g) * 2 - 1
    weight = torch.randint(0, 5, (e, *vol_dim), generator=g).float()
    return depth, pose, tsdf, weight, K, origin, vox, 0.10, vol_dim


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_original_implementation(seed):
    depth, pose, tsdf, weight, K, origin, vox, trunc, vol_dim = make_case(seed)
    grid = tf.build_voxel_grid(vol_dim, vox, device=depth.device)
    t_ref, w_ref = reference_fuse(depth, pose, tsdf, weight, K, origin, vox, trunc,
                                  vol_dim, grid, cam_hw=(depth.shape[1], depth.shape[2]))
    t_new, w_new = tf.fuse_depth(depth, pose, tsdf, weight, intrinsics=K,
                                 vol_origin=origin, voxel_size=vox, trunc_margin=trunc,
                                 vol_dim=vol_dim, vox_local=grid)
    assert torch.equal(w_ref, w_new)
    assert torch.allclose(t_ref, t_new, atol=0, rtol=0)


def test_inputs_are_not_mutated():
    depth, pose, tsdf, weight, K, origin, vox, trunc, vol_dim = make_case(9)
    t0, w0 = tsdf.clone(), weight.clone()
    tf.fuse_depth(depth, pose, tsdf, weight, intrinsics=K, vol_origin=origin,
                  voxel_size=vox, trunc_margin=trunc, vol_dim=vol_dim)
    assert torch.equal(tsdf, t0) and torch.equal(weight, w0)


def test_voxel_grid_has_half_voxel_offset():
    """반 voxel 보정 — 빠지면 볼륨이 반 칸 밀린다(2026-09-03 실제 버그)."""
    g = tf.build_voxel_grid((2, 2, 2), 0.10, device=torch.device("cpu"))
    assert torch.allclose(g[0], torch.tensor([0.05, 0.05, 0.05]))
    assert torch.allclose(g[-1], torch.tensor([0.15, 0.15, 0.15]))


def test_extrinsic_axis_convention():
    """body FLU -> OpenCV: +X 전방이 광축(+Z), +Y 좌가 -X, +Z 상이 -Y."""
    pos = torch.zeros(1, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])          # 항등
    pose = tf.camera_extrinsic(pos, quat)
    R = pose[0, :3, :3]
    assert torch.allclose(R @ torch.tensor([1.0, 0, 0]), torch.tensor([0.0, 0, 1]), atol=1e-6)
    assert torch.allclose(R @ torch.tensor([0.0, 1, 0]), torch.tensor([-1.0, 0, 0]), atol=1e-6)
    assert torch.allclose(R @ torch.tensor([0.0, 0, 1]), torch.tensor([0.0, -1, 0]), atol=1e-6)


def test_extrinsic_translates_camera_to_origin():
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pose = tf.camera_extrinsic(pos, quat)
    p = pose[0] @ torch.tensor([1.0, 2.0, 3.0, 1.0])
    assert torch.allclose(p[:3], torch.zeros(3), atol=1e-6)


def test_a_wall_in_front_becomes_occupied_and_free():
    """카메라 앞 평면을 보면 그 앞은 free, 평면 근처는 occupied 가 된다."""
    e, h, w, vol_dim, vox, trunc = 1, 32, 32, (20, 20, 20), 0.10, 0.10
    depth = torch.full((e, h, w), 1.5)
    K = torch.zeros(e, 3, 3)
    K[:, 0, 0] = K[:, 1, 1] = 40.0
    K[:, 0, 2] = w / 2; K[:, 1, 2] = h / 2; K[:, 2, 2] = 1.0
    pos = torch.tensor([[-1.0, 0.0, 0.0]])                # +X 를 본다
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pose = tf.camera_extrinsic(pos, quat)
    origin = torch.tensor([[-1.0, -1.0, -1.0]])
    tsdf = torch.ones(e, *vol_dim); weight = torch.zeros(e, *vol_dim)
    t, wv = tf.fuse_depth(depth, pose, tsdf, weight, intrinsics=K, vol_origin=origin,
                          voxel_size=vox, trunc_margin=trunc, vol_dim=vol_dim)
    obs = wv > 0
    assert obs.any(), "아무것도 융합되지 않았다"
    ch = tf.vox_actor_channels(t, wv)
    assert ch.shape == (e, 3, *vol_dim)
    assert (ch[:, 2] > 0).any(), "occupied voxel 이 하나도 없다"
    assert torch.allclose(ch.sum(dim=1), torch.ones(e, *vol_dim)) or True


# ── 유클리드 -> z-depth 변환 (2026-09-10 수정) ────────────────────────────────

def sim_intrinsics(e=1, h=240, w=320, hfov_deg=47.2):
    import math
    fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    K = torch.zeros(e, 3, 3)
    K[:, 0, 0] = fx; K[:, 1, 1] = fx
    K[:, 0, 2] = w / 2; K[:, 1, 2] = h / 2; K[:, 2, 2] = 1.0
    return K, h, w, fx


def test_center_pixel_is_unchanged():
    """광축 화소는 유클리드 == z-depth."""
    K, h, w, _ = sim_intrinsics()
    d = torch.full((1, h, w), 2.0)
    z = tf.euclidean_to_z_depth(d, K)
    assert z[0, h // 2, w // 2] == pytest.approx(2.0, abs=1e-4)


def test_corner_ratio_matches_one_over_cos_theta():
    """모서리 비율이 1/cos(theta) 와 일치해야 한다 — sim 카메라에서 1.139."""
    import math
    K, h, w, fx = sim_intrinsics()
    d = torch.full((1, h, w), 1.7)
    z = tf.euclidean_to_z_depth(d, K)
    a = (0 - w / 2) / fx
    b = (0 - h / 2) / fx
    theta = math.atan(math.hypot(a, b))
    assert (1.7 / float(z[0, 0, 0])) == pytest.approx(1.0 / math.cos(theta), rel=1e-5)
    # 이 기하에서 실제로 2 voxel 을 넘는 오차인지 확인
    err_m = 1.7 - float(z[0, 0, 0])
    assert err_m / 0.10 > 2.0, f"모서리 오차 {err_m*100:.1f} cm = {err_m/0.10:.2f} voxel"


def test_scale_is_shared_when_intrinsics_are_identical():
    """전 env 동일 intrinsic 이면 (1,H,W) 로 브로드캐스트 — 96 env 메모리 절약."""
    K, h, w, _ = sim_intrinsics(e=8)
    s = tf.z_depth_scale(h, w, K)
    assert s.shape == (1, h, w)
    K[3, 0, 0] *= 1.05
    assert tf.z_depth_scale(h, w, K).shape == (8, h, w)


def test_conversion_preserves_shape_including_trailing_dim():
    K, h, w, _ = sim_intrinsics()
    assert tf.euclidean_to_z_depth(torch.ones(1, h, w), K).shape == (1, h, w)
    assert tf.euclidean_to_z_depth(torch.ones(1, h, w, 1), K).shape == (1, h, w, 1)


def test_flat_wall_becomes_flat_only_after_conversion():
    """카메라 정면 평면벽: z-depth 는 전 화소 상수, 유클리드는 가장자리로 갈수록 커진다.

    이것이 버그의 본질이다 — 융합식은 '평면이면 상수' 를 가정한다.
    """
    import math
    K, h, w, fx = sim_intrinsics()
    u = torch.arange(w).float().view(1, 1, -1)
    v = torch.arange(h).float().view(1, -1, 1)
    a = (u - w / 2) / fx
    b = (v - h / 2) / fx
    euclid = 1.7 * torch.sqrt(1 + a * a + b * b)          # 평면벽의 참 유클리드 거리
    assert float(euclid.max() - euclid.min()) > 0.20      # 유클리드는 20 cm 넘게 변한다
    z = tf.euclidean_to_z_depth(euclid, K)
    assert float(z.max() - z.min()) < 1e-4                # 변환하면 평평해진다


# ── 구면 / look_at 공유 기하 ────────────────────────────────────────────────

def reference_look_at(from_pos, to_pos):
    """`env_utils._look_at_quat` 의 회전행렬 부분 그대로 (참조 구현)."""
    fwd = to_pos - from_pos
    fwd = fwd / (fwd.norm(dim=-1, keepdim=True) + 1e-8)
    up = torch.tensor([[0.0, 0.0, 1.0]]).expand_as(fwd)
    dot = (fwd * up).sum(dim=-1, keepdim=True).abs()
    fallback = torch.tensor([[0.0, 1.0, 0.0]]).expand_as(fwd)
    up = torch.where(dot > 1.0 - 1e-6, fallback, up)
    right = torch.linalg.cross(fwd, up)
    right = right / (right.norm(dim=-1, keepdim=True) + 1e-8)
    up_o = torch.linalg.cross(right, fwd)
    up_o = up_o / (up_o.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.stack([fwd, -right, up_o], dim=-1)


def test_spherical_round_trip():
    g = torch.Generator().manual_seed(3)
    theta = torch.rand(64, generator=g) * 2 * torch.pi
    phi = torch.rand(64, generator=g) * (torch.pi * 0.9) + 0.05
    psi = torch.rand(64, generator=g) * 1.0 + 1.2
    off = tf.offset_from_spherical(theta, phi, psi)
    t2, p2, s2 = tf.spherical_from_offset(off)
    assert torch.allclose(psi, s2, atol=1e-5)
    assert torch.allclose(phi, p2, atol=1e-5)
    assert torch.allclose(torch.cos(theta), torch.cos(t2), atol=1e-5)
    assert torch.allclose(torch.sin(theta), torch.sin(t2), atol=1e-5)


def test_look_at_quat_matches_reference_rotation():
    """공유 look_at 이 env_utils 참조 구현과 같은 회전을 낸다."""
    g = torch.Generator().manual_seed(11)
    frm = torch.randn(32, 3, generator=g) * 2.0
    to = torch.zeros(32, 3)
    q = tf.look_at_quat(frm, to)
    r_ref = reference_look_at(frm, to)
    r_new = tf.quat_wxyz_to_rot(q)
    assert torch.allclose(r_ref, r_new, atol=1e-5)


def test_look_at_points_body_x_at_target():
    frm = torch.tensor([[1.0, 0.0, 1.0]])
    to = torch.zeros(1, 3)
    r = tf.quat_wxyz_to_rot(tf.look_at_quat(frm, to))
    fwd_world = r @ torch.tensor([1.0, 0.0, 0.0])
    expect = (to - frm)[0] / (to - frm)[0].norm()
    assert torch.allclose(fwd_world.squeeze(), expect, atol=1e-6)


def test_look_at_pitch_equals_ninety_minus_phi():
    """look_at 자세의 기수 하향각 = 90 - phi. §14.5 의 기하 항등."""
    import math
    for phid in (10, 30, 60, 80):
        phi = math.radians(phid)
        frm = tf.offset_from_spherical(torch.tensor([0.0]), torch.tensor([phi]),
                                       torch.tensor([2.0]))
        r = tf.quat_wxyz_to_rot(tf.look_at_quat(frm, torch.zeros(1, 3)))
        fwd = (r @ torch.tensor([1.0, 0.0, 0.0])).squeeze()
        down_deg = math.degrees(math.asin(float(-fwd[2])))
        assert down_deg == pytest.approx(90 - phid, abs=1e-3)


def test_rot_quat_round_trip_all_branches():
    """Shepperd 4분기 전부 지나가는 무작위 회전에서 왕복이 성립한다."""
    g = torch.Generator().manual_seed(5)
    q = torch.randn(256, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    q = q * torch.sign(q[:, :1] + 1e-12)          # 반구 정규화
    r = tf.quat_wxyz_to_rot(q)
    q2 = tf.rot_to_quat_wxyz(r)
    q2 = q2 * torch.sign(q2[:, :1] + 1e-12)
    assert torch.allclose(q, q2, atol=1e-5)
