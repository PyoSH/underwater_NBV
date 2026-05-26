"""
evaluate_utils.py
=================
evaluate_recon.py 에서 사용하는 유틸리티 함수 모음.
Isaac Sim / isaaclab 의존성 없음 — numpy / matplotlib 만 사용.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# 고해상도 TSDF 재융합 (episode 종료 후, RL env TSDF와 별도 실행)
# ─────────────────────────────────────────────────────────────────────────────
def fuse_highres_tsdf(
    depth_imgs: list,
    cam_poses: list,
    K_cache: tuple,
    vol_origin: np.ndarray,
    vol_dim: tuple = (80, 80, 80),
    voxel_size: float = 0.025,
    trunc_margin: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Episode 종료 후 수집된 depth 이미지를 고해상도 TSDF로 재융합.
    RL env의 TSDF와 독립적으로 실행 — 모델 입력과 무관.

    depth_imgs : list of (H,W) float32 ndarray (meters)
    cam_poses  : list of (R (3,3), t (3,)) — world→cam
    K_cache    : (fx, fy, cx, cy)
    vol_origin : (3,) world-space corner of volume (same as RL env)
    """
    Nx, Ny, Nz = vol_dim
    vox = voxel_size
    trunc = trunc_margin
    fx, fy, cx, cy = K_cache

    xi = (np.arange(Nx, dtype=np.float32) + 0.5) * vox
    yi = (np.arange(Ny, dtype=np.float32) + 0.5) * vox
    zi = (np.arange(Nz, dtype=np.float32) + 0.5) * vox
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')
    vox_world = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1) + vol_origin  # (N,3)

    tsdf_flat   = np.zeros(Nx * Ny * Nz, np.float32)
    weight_flat = np.zeros(Nx * Ny * Nz, np.float32)

    for depth_img, (R, t) in zip(depth_imgs, cam_poses):
        H, W = depth_img.shape
        vox_cam = vox_world @ R.T + t                          # (N,3)
        vox_z   = vox_cam[:, 2]
        valid_z = vox_z > 1e-4
        inv_z   = np.where(valid_z, 1.0 / np.maximum(vox_z, 1e-4), 0.0)
        pu = (fx * vox_cam[:, 0] * inv_z + cx).astype(np.int32)
        pv = (fy * vox_cam[:, 1] * inv_z + cy).astype(np.int32)
        in_bounds = valid_z & (pu >= 0) & (pu < W) & (pv >= 0) & (pv < H)
        su = np.clip(pu, 0, W - 1)
        sv = np.clip(pv, 0, H - 1)
        sampled = depth_img[sv, su]
        sdf = sampled - vox_z
        tsdf_new = np.clip(sdf / trunc, -1.0, 1.0)
        mask = in_bounds & (sdf >= -trunc) & (sdf <= trunc)
        w_new = weight_flat + mask.astype(np.float32)
        tsdf_flat = np.where(
            mask,
            (tsdf_flat * weight_flat + tsdf_new) / np.maximum(w_new, 1e-8),
            tsdf_flat,
        )
        weight_flat = w_new

    return tsdf_flat.reshape(Nx, Ny, Nz), weight_flat.reshape(Nx, Ny, Nz)


# ─────────────────────────────────────────────────────────────────────────────
# 고해상도 Quality 계산 (fuse_highres_tsdf 이후 호출)
# ─────────────────────────────────────────────────────────────────────────────
def fuse_highres_quality(
    cam_positions: list,
    vol_origin: np.ndarray,
    weight_hires: np.ndarray,
    mu: float = 0.217,
    voxel_size: float = 0.025,
) -> np.ndarray:
    """
    cam_positions : list of (3,) world-space camera positions (cam_traj)
    weight_hires  : (Nx, Ny, Nz) from fuse_highres_tsdf — observed mask
    반환          : quality_vol (Nx, Ny, Nz) — max exp(-μd) over steps
    """
    Nx, Ny, Nz = weight_hires.shape
    vox = voxel_size

    xi = (np.arange(Nx, dtype=np.float32) + 0.5) * vox
    yi = (np.arange(Ny, dtype=np.float32) + 0.5) * vox
    zi = (np.arange(Nz, dtype=np.float32) + 0.5) * vox
    gx, gy, gz = np.meshgrid(xi, yi, zi, indexing='ij')
    vox_world = (
        np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1) + vol_origin
    )  # (N, 3)

    observed_flat = (weight_hires > 0).ravel()
    quality_flat  = np.zeros(Nx * Ny * Nz, np.float32)

    for cam_pos in cam_positions:
        dist        = np.linalg.norm(vox_world - cam_pos[np.newaxis, :], axis=-1)
        quality_new = np.exp(-mu * dist).astype(np.float32)
        quality_flat = np.where(
            observed_flat,
            np.maximum(quality_flat, quality_new),
            quality_flat,
        )

    return quality_flat.reshape(Nx, Ny, Nz)


# ─────────────────────────────────────────────────────────────────────────────
# 3D 복원: TSDF Marching Cubes
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_mesh(tsdf_vol: np.ndarray,
                    weight_vol: np.ndarray,
                    vol_origin: np.ndarray,
                    voxel_size: float
                    ) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    미관측 복셀(weight==0)을 +1.0으로 마스킹 후 zero-crossing 등가면 추출.
    반환: (verts_world, faces) 또는 (None, None)
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        print("[recon] scikit-image 없음 → pip install scikit-image")
        return None, None

    tsdf_masked = tsdf_vol.copy()
    tsdf_masked[weight_vol == 0] = 1.0

    if tsdf_masked.min() >= 0.0 or tsdf_masked.max() <= 0.0:
        print("[recon] TSDF zero-crossing 없음 (커버리지 부족)")
        return None, None

    try:
        verts, faces, _, _ = marching_cubes(
            tsdf_masked, level=0.0, spacing=(voxel_size,) * 3
        )
    except Exception as e:
        print(f"[recon] marching_cubes 실패: {e}")
        return None, None

    return (verts + vol_origin).astype(np.float32), faces


# ─────────────────────────────────────────────────────────────────────────────
# 버텍스 컬러링: 메시 정점 → RGB 이미지 투영
# ─────────────────────────────────────────────────────────────────────────────
def color_mesh_vertices(verts_world: np.ndarray,
                        cam_poses: list,
                        rgb_imgs:  list,
                        fx: float, fy: float,
                        cx: float, cy: float) -> np.ndarray:
    """
    Marching Cubes 정점에 수집된 뷰들의 색상을 가중 평균으로 투영.

    cam_poses : list of (R (3,3), t (3,)) — world→cam (OpenCV 프레임)
                p_cam = p_world @ R.T + t
    rgb_imgs  : list of (H,W,3) uint8

    가중치 = cos(입사각) — 카메라 정면에서 본 정점에 높은 가중치.
    모든 뷰에서 보이지 않는 정점은 회색(128)으로 처리.

    반환: colors (V, 3) uint8
    """
    V = len(verts_world)
    color_accum  = np.zeros((V, 3), dtype=np.float64)
    weight_accum = np.zeros(V,      dtype=np.float64)

    for (R, t), rgb in zip(cam_poses, rgb_imgs):
        H, W = rgb.shape[:2]

        pts_cam = verts_world @ R.T + t[np.newaxis, :]  # (V, 3)

        z = pts_cam[:, 2]
        in_front = z > 0.05

        u = fx * pts_cam[:, 0] / z.clip(1e-6) + cx
        v = fy * pts_cam[:, 1] / z.clip(1e-6) + cy
        u_int = np.round(u).astype(np.int32)
        v_int = np.round(v).astype(np.int32)

        in_bounds = (
            in_front          &
            (u_int >= 0)      &
            (u_int <  W)      &
            (v_int >= 0)      &
            (v_int <  H)
        )

        dist  = np.linalg.norm(pts_cam, axis=-1).clip(1e-6)
        cos_w = (pts_cam[:, 2] / dist).clip(0.0, 1.0) * in_bounds.astype(np.float64)

        u_safe  = u_int.clip(0, W - 1)
        v_safe  = v_int.clip(0, H - 1)
        sampled = rgb[v_safe, u_safe].astype(np.float64)  # (V, 3)

        color_accum  += cos_w[:, np.newaxis] * sampled
        weight_accum += cos_w

    colors = np.full((V, 3), 128, dtype=np.uint8)
    valid  = weight_accum > 0
    colors[valid] = (
        color_accum[valid] / weight_accum[valid, np.newaxis]
    ).clip(0, 255).astype(np.uint8)

    return colors


# ─────────────────────────────────────────────────────────────────────────────
# PLY 저장
# ─────────────────────────────────────────────────────────────────────────────
def save_ply_points(path: str, pts: np.ndarray):
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, pts, fmt="%.6f")
    print(f"[save] cloud         → {path}  ({len(pts)} pts)")


def save_ply_mesh(path: str, verts: np.ndarray, faces: np.ndarray,
                colors: np.ndarray | None = None):
    """
    colors (V,3) uint8 이면 버텍스 컬러 포함 PLY 저장.
    None 이면 geometry only.
    """
    has_color = colors is not None
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        if has_color:
            combined = np.hstack([verts, colors.astype(np.float32)])
            np.savetxt(f, combined, fmt=["%.4f","%.4f","%.4f","%d","%d","%d"])
        else:
            np.savetxt(f, verts, fmt="%.4f")

        face_data = np.hstack([np.full((len(faces), 1), 3, dtype=int), faces])
        np.savetxt(f, face_data, fmt="%d")

    tag = "colored mesh" if has_color else "mesh"
    print(f"[save] {tag:12s} → {path}  ({len(verts)} verts, {len(faces)} faces)")


# ─────────────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────────────
def save_trajectory_plot(path: str, cam_traj: list,
                        rock_pos: np.ndarray,
                        verts_world: np.ndarray | None = None):
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection="3d")

    traj = np.array(cam_traj)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            "b-o", markersize=3, linewidth=1, label="camera path")
    ax.scatter(*traj[0],  c="g", s=80, zorder=5, label="start")
    ax.scatter(*traj[-1], c="r", s=80, zorder=5, label="end")
    ax.scatter(*rock_pos, c="k", s=120, marker="*", label="target")

    if verts_world is not None and len(verts_world) > 0:
        stride = max(1, len(verts_world) // 500)
        sub = verts_world[::stride]
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
                    c="orange", s=2, alpha=0.3, label="recon (sample)")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(fontsize=8)
    ax.set_title("Camera Trajectory & Reconstruction")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[save] trajectory    → {path}")


def save_coverage_plot(path: str, coverage_hist: list, coverage_q_hist: list | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(coverage_hist, linewidth=1.5, label="binary")
    if coverage_q_hist:
        ax.plot(coverage_q_hist, linewidth=1.5, linestyle="--", label="quality")
    ax.axhline(y=0.96, color="r", linestyle="--", linewidth=1, label="terminal")
    ax.set_xlabel("Step"); ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Coverage over Steps")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[save] coverage      → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 에피소드 결과 저장
# ─────────────────────────────────────────────────────────────────────────────
def save_episode_results(out_dir: Path, ep_idx: int, env_id: int,
                        tsdf_np: np.ndarray, weight_np: np.ndarray,
                        surf_np: np.ndarray, origin_np: np.ndarray,
                        voxel_size: float,
                        cam_traj: list, coverage_hist: list,
                        cam_poses: list, rgb_imgs: list,
                        K_cache: tuple | None,
                        rock_pos: np.ndarray,
                        coverage_q_hist: list | None = None,
                        tsdf_hires: np.ndarray | None = None,
                        weight_hires: np.ndarray | None = None,
                        voxel_hires: float | None = None):
    ep_dir = out_dir / f"ep_{ep_idx:03d}_env{env_id}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    vox = voxel_size

    # GT 표면 포인트 클라우드
    idx_gt = np.argwhere(surf_np)
    pts_gt = origin_np + (idx_gt + 0.5) * vox
    save_ply_points(str(ep_dir / "gt_surface.ply"), pts_gt)

    # 관측된 복셀 포인트 클라우드
    idx_obs = np.argwhere(weight_np > 0)
    pts_obs = origin_np + (idx_obs + 0.5) * vox
    save_ply_points(str(ep_dir / "observed_voxels.ply"), pts_obs)

    # Marching Cubes 복원 — hires TSDF 우선 사용
    use_hi = (tsdf_hires is not None) and (weight_hires is not None)
    tsdf_for_mesh   = tsdf_hires   if use_hi else tsdf_np
    weight_for_mesh = weight_hires if use_hi else weight_np
    vox_for_mesh    = voxel_hires  if (use_hi and voxel_hires) else vox

    verts_world, faces = reconstruct_mesh(tsdf_for_mesh, weight_for_mesh, origin_np, vox_for_mesh)

    if verts_world is not None:
        save_ply_mesh(str(ep_dir / "recon_mesh.ply"), verts_world, faces)

        if cam_poses and rgb_imgs and K_cache is not None:
            fx, fy, cx, cy = K_cache
            vert_colors = color_mesh_vertices(
                verts_world, cam_poses, rgb_imgs, fx, fy, cx, cy
            )
            save_ply_mesh(str(ep_dir / "recon_mesh_colored.ply"),
                        verts_world, faces, colors=vert_colors)

    # 플롯
    if cam_traj:
        save_coverage_plot(str(ep_dir / "coverage_curve.png"), coverage_hist, coverage_q_hist)
        save_trajectory_plot(str(ep_dir / "trajectory.png"),
                            cam_traj, rock_pos, verts_world)

    final_cov = coverage_hist[-1] if coverage_hist else 0.0
    print(f"[ep {ep_idx:03d}] done → {ep_dir}  final_cov={final_cov:.4f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 외부 뷰 동영상 저장
# ─────────────────────────────────────────────────────────────────────────────
def save_episode_video(path: str, frames: list, fps: int = 10):
    try:
        import imageio
    except ImportError:
        print("[video] imageio 없음 → pip install 'imageio[ffmpeg]'")
        return
    if not frames:
        return
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"[save] ext video     → {path}  ({len(frames)} frames)")