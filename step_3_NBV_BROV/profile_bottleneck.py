"""step_3 처리량 병목 프로파일 — 리셋 vs 스텝, 그리고 리셋 내부 세부 항목.

배경(2026-08-26 서버 실측): env-step당 소요시간이 env 수와 **무관하게** 0.63~0.70초로
고정이라 병렬화 이득이 없었고, 128 env에서는 2시간 넘게 첫 롤아웃도 못 끝냈다.
그때 프로세스 상태가 **GPU 1% / CPU 109%(단일코어) / RSS 46.7 GB** 였다 —
GPU 병목이 아니라 **Python 병목**이라는 직접 증거.

유력 용의자는 리셋 시 env마다 도는 Python 루프들이다:
  - `_voxelize_gt_mesh()`: env마다 `_load_mesh()` → USD 메쉬 재읽기 +
    `_triangulate()`의 **면 단위 순수 Python 루프**
  - `_randomize_rock_pose()`: env마다 USD Xform 조작
  - `_update_light_intensity()`: env마다 USD 속성 쓰기

추측으로 고치지 않기 위해 **실제로 어디에 시간이 쓰이는지** 측정한다.

사용법 (컨테이너 안)
--------------------
python.sh -u profile_bottleneck.py --headless --enable_cameras --num_envs 16
python.sh -u profile_bottleneck.py --headless --enable_cameras --num_envs 64
"""

import argparse
import sys
import time
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--decisions", type=int, default=6,
                    help="측정할 정책 결정 수(리셋을 포함하려면 에피소드 길이 이상)")
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

from envs.env_cfg import NBVBROVEnvCfg
from envs.env import NBVBROVEnv

_timings = defaultdict(list)


def _wrap(obj, name: str, key: str):
    """메서드를 감싸 호출 시간을 기록한다(원본 동작 불변)."""
    original = getattr(obj, name)

    def timed(*a, **kw):
        t0 = time.perf_counter()
        out = original(*a, **kw)
        _timings[key].append(time.perf_counter() - t0)
        return out

    setattr(obj, name, timed)
    return original


def _mesh_size(env) -> None:
    """바위 메쉬 규모 — `_triangulate()` Python 루프의 반복 횟수를 좌우한다."""
    try:
        verts, faces = env._load_mesh(0)
    except Exception as exc:                                  # noqa: BLE001
        print(f"[prof] 메쉬 크기 측정 실패: {exc}")
        return
    print(f"[prof] 바위 메쉬: 정점 {len(verts):,} / 삼각형 {len(faces):,}")
    print(f"[prof]   → _triangulate() Python 루프가 리셋 1회·env 1개당 "
          f"약 {len(faces):,}회 반복 (env {args.num_envs}개면 "
          f"{len(faces)*args.num_envs:,}회)")


def _report(total_wall: float, n_env: int, n_dec: int):
    print("\n" + "=" * 72)
    print(f"{'구간':<28} {'호출':>6} {'총시간(s)':>10} {'평균(ms)':>10} {'비중':>7}")
    print("-" * 72)
    for key in sorted(_timings, key=lambda k: -sum(_timings[k])):
        v = _timings[key]
        tot = sum(v)
        print(f"{key:<28} {len(v):>6} {tot:>10.2f} {tot/len(v)*1000:>10.1f} "
              f"{tot/total_wall*100:>6.1f}%")
    print("-" * 72)
    env_steps = n_env * n_dec
    print(f"[prof] 전체 wall {total_wall:.2f}s / env-step {env_steps} "
          f"→ **env-step당 {total_wall/env_steps:.3f}s**, 결정당 {total_wall/n_dec:.2f}s")
    print("[prof] GPU util이 낮고 특정 구간이 지배적이면 그 구간이 Python 병목이다.")


def main() -> int:
    cfg = NBVBROVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    # 에피소드를 짧게 만들어 측정 구간 안에 **리셋이 반드시 포함**되게 한다
    cfg.episode_length_s = 3 * (cfg.sim.dt * cfg.decimation)
    cfg.curriculum_enabled = False

    print(f"[prof] num_envs={args.num_envs} decimation={cfg.decimation} "
          f"(결정당 물리 서브스텝 {cfg.decimation}회), 에피소드=3결정")

    t_build0 = time.perf_counter()
    env = NBVBROVEnv(cfg)
    print(f"[prof] env 생성 {time.perf_counter()-t_build0:.2f}s")

    # 계측 대상 — 리셋 내부 세부 항목까지 분해
    _wrap(env, "_reset_idx", "_reset_idx (전체)")
    _wrap(env, "_voxelize_gt_mesh", "  └ _voxelize_gt_mesh")
    _wrap(env, "_load_mesh", "     └ _load_mesh(USD읽기)")
    _wrap(env, "_triangulate", "        └ _triangulate(Python루프)")
    _wrap(env, "_randomize_rock_pose", "  └ _randomize_rock_pose")
    _wrap(env, "_update_light_intensity", "  └ _update_light_intensity")
    _wrap(env, "_apply_action", "_apply_action (물리 서브스텝)")
    _wrap(env, "_get_observations", "_get_observations (렌더/버퍼)")
    _wrap(env, "_get_rewards", "_get_rewards (TSDF융합)")

    try:
        t_reset0 = time.perf_counter()
        env.reset()
        print(f"[prof] 최초 reset {time.perf_counter()-t_reset0:.2f}s "
              f"(env {args.num_envs}개 전체 복셀화 포함)")

        _mesh_size(env)

        # 최초 reset과 _mesh_size 호출은 성격이 달라 정상 루프 측정에서 제외
        for k in list(_timings):
            _timings[k].clear()

        act = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        t0 = time.perf_counter()
        for i in range(args.decisions):
            act.uniform_(-1.0, 1.0)
            env.step(act)
        total = time.perf_counter() - t0

        _report(total, args.num_envs, args.decisions)

        free, tot = torch.cuda.mem_get_info()
        print(f"[prof] VRAM 사용 {(tot-free)/1024**3:.2f} / {tot/1024**3:.2f} GB")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
