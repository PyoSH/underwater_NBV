"""렌더 해상도·env 수에 따른 처리량/VRAM 실측. 2026-09-10.

왜 필요한가: 학습 카메라를 실기와 일치시키며 320x240 -> 640x480 으로 올렸다(계획서
§15.5 "(나)"). 화소가 **4배**이고 UW 렌더 커널은 화소당 비용이다. 이 프로젝트는 이미
렌더 자원 한계에 두 번 부딪혔고(descriptor set 920건 / per-view 컬링 슬롯 4시간 정지),
그 실패는 **크래시 없이 조용히** 온다(§3). 그래서 결정 전에 잰다.

로컬 GPU 에서 재는 것은 **절대 상한이 아니라 스케일링**이다 — 해상도 간 결정당 시간 비와
env 당 VRAM 증분. 서버는 그 비율에 자기 VRAM 을 곱하면 된다.

사용법 (isaac-lab-base 컨테이너):
  /isaac-sim/python.sh -u tools/measure_render_scaling.py --headless --enable_cameras \\
      --num_envs 16 --resolution 640x480 --decisions 6
"""

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--resolution", type=str, default="640x480",
                    help="WxH. aperture 는 fx 465.518 이 되도록 자동 계산한다")
parser.add_argument("--decisions", type=int, default=6,
                    help="계측할 정책 스텝 수 (앞 --warmup 개는 버린다)")
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--mesh_pool", type=str, default=None)
parser.add_argument("--mesh_pool_limit", type=int, default=0)
parser.add_argument("--no_require_texture", action="store_true")
parser.add_argument("--fixed_object_pose", action="store_true")
parser.add_argument("--out", type=str, default=None, help="결과 JSON 추가 기록")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from envs.env_cfg import NBVBROVEnvCfg  # noqa: E402
from envs.env import NBVBROVEnv  # noqa: E402

FX_REAL = 465.5181034880913
FOCAL_MM = 24.0


def gpu_mb() -> float:
    """프로세스 전체 VRAM [MiB]. torch 만 보면 Kit/RTX 할당을 놓친다."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        import os
        mine = str(os.getpid())
        total = 0.0
        for line in out.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            total += float(mem)          # Kit 은 자식 프로세스를 쓰지 않는다
        return total
    except Exception:
        return float("nan")


def main() -> int:
    w, h = (int(v) for v in args.resolution.lower().split("x"))
    aperture = w * FOCAL_MM / FX_REAL

    cfg = NBVBROVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    cfg.use_tiled_camera = True
    cfg.scene.camera.width = w
    cfg.scene.camera.height = h
    cfg.scene.camera.spawn.horizontal_aperture = aperture
    if args.mesh_pool:
        cfg.mesh_pool_manifest = args.mesh_pool
        cfg.mesh_pool_limit = args.mesh_pool_limit
        cfg.mesh_pool_require_texture = not args.no_require_texture
        cfg.mesh_pool_split = "all"
    cfg.randomize_object_pose = not args.fixed_object_pose

    fx = w * FOCAL_MM / aperture
    print(f"[scale] {w}x{h}  aperture {aperture:.4f} mm -> fx {fx:.1f}  "
          f"env {args.num_envs}  화소 {w*h/1000:.1f}k")

    t0 = time.perf_counter()
    env = NBVBROVEnv(cfg=cfg)
    build_s = time.perf_counter() - t0
    obs, _ = env.reset()
    mem_after_build = gpu_mb()
    print(f"[scale] 씬 생성 {build_s:.1f} s,  VRAM {mem_after_build:.0f} MiB")

    action = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
    action[:, 0] = 1.0                    # orbit 과 같은 방위 이동 = 매 결정 새 렌더
    times = []
    for k in range(args.warmup + args.decisions):
        s = time.perf_counter()
        env.step(action)
        torch.cuda.synchronize()
        dt = time.perf_counter() - s
        if k >= args.warmup:
            times.append(dt)
        print(f"[scale]   결정 {k+1:2d}  {dt:6.3f} s" + ("  (warmup)" if k < args.warmup else ""))

    # 렌더 건전성 — 조용한 손상을 잡는 유일한 장치
    rgb = env._camera.data.output["uw_rgb"][..., :3].float()
    depth = env._camera.data.output["distance_to_camera"].float()
    if depth.dim() == 4:
        depth = depth.squeeze(-1)
    std = rgb.std(dim=(1, 2, 3))
    finite = (torch.isfinite(depth) & (depth > 0)).float().mean(dim=(1, 2))
    dead = int(((std < 1e-3) | (finite < 1e-3)).sum())

    mean_s = sum(times) / len(times)
    mem = gpu_mb()
    result = dict(width=w, height=h, pixels=w * h, aperture_mm=round(aperture, 4),
                  fx=round(fx, 2), num_envs=args.num_envs,
                  build_s=round(build_s, 2),
                  decision_s_mean=round(mean_s, 4),
                  decision_s_min=round(min(times), 4),
                  decision_s_max=round(max(times), 4),
                  vram_mib=round(mem, 1), dead_cameras=dead,
                  img_std_min=round(float(std.min()), 4),
                  depth_finite_min=round(float(finite.min()), 4))
    print(f"[scale] === 결정당 {mean_s:.3f} s (min {min(times):.3f} / max {max(times):.3f})")
    print(f"[scale] === VRAM {mem:.0f} MiB,  죽은 카메라 {dead}개")
    if dead:
        print("[scale] ✗ 렌더러가 일부 env 파이프라인을 만들지 못했다 — 이 env 수는 쓸 수 없다")
    print("[scale] JSON " + json.dumps(result))
    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = json.loads(p.read_text()) if p.exists() else []
        rows.append(result)
        p.write_text(json.dumps(rows, indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
