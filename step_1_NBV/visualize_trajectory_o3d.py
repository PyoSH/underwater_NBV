#!/usr/bin/env python3
"""
3D trajectory + mesh visualization using Open3D OffscreenRenderer.

Usage:
    /isaac-sim/python.sh visualize_trajectory_o3d.py \
        --episodes \
            "Manual:recon_output/basic_orbit/ep_000_env0" \
            "UW_NBV_5:recon_output/UW_NBV_5_0000143360/ep_000_env0" \
        --out analysis/traj_vis.png

    # Multiple episodes per algorithm (averaged trajectory from ep_000):
    --episodes "Manual:recon_output/basic_orbit/ep_000_env0" ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_trajectory_lineset(traj_xyz: np.ndarray, color=(0.9, 0.2, 0.1)):
    """N×3 array → LineSet (colored path)."""
    pts = o3d.utility.Vector3dVector(traj_xyz)
    lines = [[i, i + 1] for i in range(len(traj_xyz) - 1)]
    ls = o3d.geometry.LineSet(pts, o3d.utility.Vector2iVector(lines))
    ls.paint_uniform_color(color)
    return ls


def make_trajectory_arrows(traj_xyz: np.ndarray, arrow_step=6,
                            cone_radius=0.06, cone_height=0.14,
                            color=(0.95, 0.25, 0.1)):
    """N×3 → list of directional cone arrowheads along the trajectory."""
    meshes = []
    n = len(traj_xyz)
    indices = list(range(0, n - 1, arrow_step))
    for i in indices:
        j = min(i + 1, n - 1)
        direction = traj_xyz[j] - traj_xyz[i]
        length = np.linalg.norm(direction)
        if length < 1e-6:
            continue
        direction = direction / length

        cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius,
                                                      height=cone_height)
        cone.compute_vertex_normals()
        cone.paint_uniform_color(color)

        # create_cone points in +Z; rotate to movement direction
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(z, direction)
        s = float(np.linalg.norm(v))
        c = float(np.dot(z, direction))
        T = np.eye(4)
        if s < 1e-6:
            if c < 0:
                T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            T[:3, :3] = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

        T[:3, 3] = traj_xyz[i]  # base of cone at trajectory point
        cone.transform(T)
        meshes.append(cone)
    return meshes


def make_camera_spheres(traj_xyz: np.ndarray, radius=0.05,
                        start_color=(0.1, 0.8, 0.1),
                        end_color=(0.9, 0.2, 0.1),
                        mid_color=(0.2, 0.4, 0.9)):
    """N×3 → list of small colored spheres (start=green, end=red, mid=blue)."""
    n = len(traj_xyz)
    meshes = []
    for i, pt in enumerate(traj_xyz):
        if i == 0:
            c = start_color
        elif i == n - 1:
            c = end_color
        else:
            t = i / max(n - 1, 1)
            c = tuple(start_color[k] * (1 - t) + end_color[k] * t for k in range(3))
            c = mid_color  # uniform mid color for cleaner look
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(pt)
        s.paint_uniform_color(c)
        s.compute_vertex_normals()
        meshes.append(s)
    return meshes


def make_rock_marker(rock_pos: np.ndarray, radius=0.08):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    s.translate(rock_pos)
    s.paint_uniform_color([1.0, 0.8, 0.0])
    s.compute_vertex_normals()
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def compute_view_params(rock_pos: np.ndarray, traj_xyz: np.ndarray,
                        theta_deg=45.0, phi_deg=55.0, dist_factor=1.6,
                        fixed_radius: float | None = None):
    """
    Place viewer camera at fixed spherical offset from rock_pos.
    Returns (eye, center, up) in world coords.
    fixed_radius: if set, overrides per-episode radius computation.
    """
    if fixed_radius is not None:
        radius = fixed_radius
    else:
        all_pts = np.vstack([traj_xyz, rock_pos[None]])
        span = np.max(all_pts, axis=0) - np.min(all_pts, axis=0)
        radius = np.linalg.norm(span) * dist_factor

    theta = np.radians(theta_deg)
    phi   = np.radians(phi_deg)
    offset = np.array([
        radius * np.sin(phi) * np.cos(theta),
        radius * np.sin(phi) * np.sin(theta),
        radius * np.cos(phi),
    ])
    eye    = rock_pos + offset
    center = rock_pos
    up     = np.array([0.0, 0.0, 1.0])
    return eye, center, up


def render_scene(mesh_path: str, traj_xyz: np.ndarray, rock_pos: np.ndarray,
                 width=800, height=700,
                 theta_deg=45.0, phi_deg=55.0,
                 show_spheres=True, sphere_step=3,
                 dist_factor=0.9, fixed_radius: float | None = None,
                 mesh_only=False, arrow_step=6) -> np.ndarray:
    """
    Render mesh + trajectory to an RGBA image (numpy H×W×3 uint8).
    sphere_step: draw a sphere every N steps (avoid clutter).
    """
    rend = rendering.OffscreenRenderer(width, height)
    scene = rend.scene
    scene.set_background([0.12, 0.12, 0.15, 1.0])

    # ── 조명 ──────────────────────────────────────────────────────────────────
    scene.scene.enable_sun_light(True)
    scene.scene.set_sun_light([-1, -0.5, -1], [1.5, 1.5, 1.5], 75000)

    mat_mesh = rendering.MaterialRecord()
    mat_mesh.shader = "defaultUnlit"
    mat_mesh.base_color = [0.85, 0.82, 0.75, 1.0]

    mat_line = rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0
    mat_line.base_color = [0.95, 0.25, 0.1, 1.0]

    mat_sphere = rendering.MaterialRecord()
    mat_sphere.shader = "defaultLit"

    mat_rock = rendering.MaterialRecord()
    mat_rock.shader = "defaultLit"
    mat_rock.base_color = [1.0, 0.8, 0.0, 1.0]

    # ── 메시 로드 ──────────────────────────────────────────────────────────────
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0:
        print(f"[warn] empty mesh: {mesh_path}")
    else:
        mesh.compute_vertex_normals()
        scene.add_geometry("mesh", mesh, mat_mesh)

    if not mesh_only:
        # ── 궤적 라인 + 화살표 ────────────────────────────────────────────────
        if len(traj_xyz) >= 2:
            ls = make_trajectory_lineset(traj_xyz)
            scene.add_geometry("traj_line", ls, mat_line)

            mat_arrow = rendering.MaterialRecord()
            mat_arrow.shader = "defaultUnlit"
            mat_arrow.base_color = [0.95, 0.25, 0.1, 1.0]
            for k, arrow in enumerate(make_trajectory_arrows(traj_xyz,
                                                              arrow_step=arrow_step)):
                scene.add_geometry(f"arrow_{k}", arrow, mat_arrow)

        # ── 카메라 위치 마커 ──────────────────────────────────────────────────
        if show_spheres and len(traj_xyz) > 0:
            sel = traj_xyz[::sphere_step]
            if len(traj_xyz) > 0 and (len(traj_xyz) - 1) % sphere_step != 0:
                sel = np.vstack([sel, traj_xyz[[-1]]])
            n = len(sel)
            for i, pt in enumerate(sel):
                t = i / max(n - 1, 1)
                r = 0.1 + 0.9 * t
                g = 0.8 * (1 - t)
                b = 0.2
                s = o3d.geometry.TriangleMesh.create_sphere(radius=0.06)
                s.translate(pt)
                s.compute_vertex_normals()
                m = rendering.MaterialRecord()
                m.shader = "defaultLit"
                m.base_color = [r, g, b, 1.0]
                scene.add_geometry(f"cam_{i}", s, m)

        # ── rock 마커 ─────────────────────────────────────────────────────────
        rock_m = make_rock_marker(rock_pos)
        scene.add_geometry("rock", rock_m, mat_rock)

    # ── 카메라 뷰 설정 ────────────────────────────────────────────────────────
    eye, center, up = compute_view_params(rock_pos, traj_xyz,
                                          theta_deg, phi_deg, dist_factor,
                                          fixed_radius)
    rend.setup_camera(60.0,
                      center.astype(np.float32),
                      eye.astype(np.float32),
                      up.astype(np.float32))

    img = np.asarray(rend.render_to_image())   # H×W×3 uint8
    del rend
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side figure
# ─────────────────────────────────────────────────────────────────────────────

def add_label(img: np.ndarray, label: str, n_steps: int,
              coverage_q: float) -> np.ndarray:
    """Draw text label onto image using matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from io import BytesIO

    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100),
                           dpi=100)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{label}\nsteps={n_steps}  cov_q={coverage_q:.3f}",
                 fontsize=13, color="white", pad=6,
                 bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    buf.seek(0)
    out = np.frombuffer(buf.read(), dtype=np.uint8)
    import cv2
    out = cv2.imdecode(out, cv2.IMREAD_COLOR)[:, :, ::-1]
    return out


def stitch_horizontal(images: list, gap=10) -> np.ndarray:
    """Stack images horizontally with a dark gap."""
    h = max(im.shape[0] for im in images)
    strips = []
    for i, im in enumerate(images):
        if im.shape[0] < h:
            pad = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
            pad[:] = 30
            im = np.vstack([im, pad])
        strips.append(im)
        if i < len(images) - 1:
            strips.append(np.full((h, gap, 3), 30, dtype=np.uint8))
    return np.hstack(strips)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", required=True,
                        help="label:ep_dir  e.g. 'Manual:recon_output/basic_orbit/ep_000_env0'")
    parser.add_argument("--out", default="analysis/traj_vis.png")
    parser.add_argument("--width",  type=int, default=800)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--theta",  type=float, default=45.0,
                        help="viewer azimuth (deg)")
    parser.add_argument("--phi",    type=float, default=55.0,
                        help="viewer elevation from +Z (deg)")
    parser.add_argument("--sphere_step", type=int, default=3,
                        help="draw camera sphere every N steps")
    parser.add_argument("--dist_factor", type=float, default=0.9,
                        help="viewer distance multiplier (smaller = zoom in)")
    parser.add_argument("--mesh_only", action="store_true",
                        help="render mesh only, no trajectory or markers")
    parser.add_argument("--arrow_step", type=int, default=6,
                        help="place arrowhead every N trajectory steps")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 공통 반지름 계산: 모든 에피소드의 궤적 범위 합산 ────────────────────────
    all_trajs = []
    all_rocks = []
    episodes_data = []
    import csv as _csv

    for entry in args.episodes:
        label, ep_dir_str = entry.split(":", 1)
        ep_dir = Path(ep_dir_str)
        traj_path = ep_dir / "trajectory_xyz.npy"
        rock_path = ep_dir / "rock_pos.npy"
        mesh_path = ep_dir / "recon_mesh_colored.ply"

        if not traj_path.exists():
            print(f"[skip] {label}: trajectory_xyz.npy not found in {ep_dir}")
            print("       → re-run evaluate_recon.py to generate trajectory data")
            continue
        if not mesh_path.exists():
            print(f"[skip] {label}: recon_mesh_colored.ply not found")
            continue

        traj_xyz = np.load(str(traj_path))
        rock_pos = np.load(str(rock_path)) if rock_path.exists() else traj_xyz.mean(0)

        cov_q = 0.0
        step_log = ep_dir / "step_log.csv"
        if step_log.exists():
            with open(step_log) as f:
                rows = list(_csv.DictReader(f))
            if rows:
                cov_q = float(rows[-1]["coverage_q"])

        all_trajs.append(traj_xyz)
        all_rocks.append(rock_pos)
        episodes_data.append((label, str(mesh_path), traj_xyz, rock_pos, cov_q))

    if not episodes_data:
        print("[error] no valid episodes")
        sys.exit(1)

    all_pts = np.vstack(all_trajs + [r[None] for r in all_rocks])
    span    = np.max(all_pts, axis=0) - np.min(all_pts, axis=0)
    common_radius = np.linalg.norm(span) * args.dist_factor
    print(f"[info] common radius = {common_radius:.2f}m  (dist_factor={args.dist_factor})")

    # ── 렌더링 ────────────────────────────────────────────────────────────────
    rendered = []

    for label, mesh_path, traj_xyz, rock_pos, cov_q in episodes_data:
        n_steps = len(traj_xyz)
        print(f"[render] {label}  steps={n_steps}  cov_q={cov_q:.3f}  "
              f"mesh={Path(mesh_path).name}")

        img = render_scene(mesh_path, traj_xyz, rock_pos,
                           width=args.width, height=args.height,
                           theta_deg=args.theta, phi_deg=args.phi,
                           sphere_step=args.sphere_step,
                           dist_factor=args.dist_factor,
                           fixed_radius=common_radius,
                           mesh_only=args.mesh_only,
                           arrow_step=args.arrow_step)

        img = add_label(img, label, n_steps, cov_q)
        rendered.append(img)

    if not rendered:
        print("[error] no valid episodes rendered")
        sys.exit(1)

    combined = stitch_horizontal(rendered)

    import cv2
    cv2.imwrite(str(out_path), combined[:, :, ::-1])
    print(f"[save] {out_path}  ({combined.shape[1]}×{combined.shape[0]})")


if __name__ == "__main__":
    main()
