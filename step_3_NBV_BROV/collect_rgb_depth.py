"""TRIDENT 검증용 RGB / GT depth 쌍 수집 (Stage 3 선행).

왜 필요한가
-----------
Stage 3는 `env.py`의 `distance_to_camera`(PhysX 참값)를 단안 RGB 기반 추정치로
교체하는 단계다. 그런데 그 전에 **TRIDENT가 metric depth를 내는지** 확인해야
한다. 단안 depth는 통상 scale ambiguity가 있고, metric이 아니면 TSDF 적분과
coverage 계산이 성립하지 않는다 — 그 경우 Stage 3의 설계 자체가 달라진다
(스케일 복원 경로가 추가로 필요).

지연시간은 이미 측정됐다(Isaac 카메라 해상도·배치 16에서 env당 1.48 ms).
남은 미지수가 metric 여부다.

수집 설계
---------
- 입력은 **`uw_rgb`**(수중 감쇠가 적용된 렌더 결과)를 쓴다. TRIDENT는 흐린
  수중 영상을 받아 dehazing과 depth를 함께 추정하는 모델이므로, 깨끗한 RGB가
  아니라 이쪽이 실제 사용 조건이다.
- 매 결정마다 랜덤 액션으로 시점을 바꾸고 리셋마다 바위 자세도 바뀌므로,
  **장면 간 스케일 일관성**을 검증할 수 있는 표본이 모인다. metric 여부는
  한 장면 안이 아니라 장면들 사이에서 갈린다.

사용법
------
python.sh -u collect_rgb_depth.py --headless --num_envs 8 --num_samples 200 \
    --out /workspace/.../trident_pairs.npz
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TRIDENT 검증용 RGB/GT depth 수집")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_samples", type=int, default=200)
parser.add_argument("--out", type=str, default="trident_pairs.npz")
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.env_cfg import NBVBROVEnvCfg
from envs.env import NBVBROVEnv


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = NBVBROVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum_enabled = False      # 수집 중 난이도 변화는 무의미

    env = NBVBROVEnv(cfg)
    E, A = env.num_envs, cfg.action_space
    gen = torch.Generator(device=env.device).manual_seed(args.seed)

    rgb_list: list[np.ndarray] = []
    dep_list: list[np.ndarray] = []
    meta: list[tuple[float, float, float]] = []

    try:
        obs, _ = env.reset()
        while len(rgb_list) < args.num_samples:
            act = torch.rand((E, A), generator=gen, device=env.device) * 2.0 - 1.0
            env.step(act)

            # 관측은 목표 pose에 도달한 시점(결정 경계)에서만 읽는다 — 학습이
            # TSDF를 융합하는 시점과 동일하게 맞춰야 검증이 대표성을 갖는다.
            rgb = env._camera.data.output["uw_rgb"][:, :, :, :3]      # (E,H,W,3)
            dep = env._camera.data.output["distance_to_camera"]        # (E,H,W,1)
            if dep.ndim == 4:
                dep = dep.squeeze(-1)

            rgb_np = rgb.detach().to(torch.uint8).cpu().numpy()
            dep_np = dep.detach().float().cpu().numpy()
            for i in range(E):
                if len(rgb_list) >= args.num_samples:
                    break
                # 유효 depth가 거의 없는 프레임(물체가 화면 밖)은 스케일 추정에
                # 노이즈만 더하므로 제외한다.
                d = dep_np[i]
                if np.isfinite(d).mean() < 0.5:
                    continue
                rgb_list.append(rgb_np[i])
                dep_list.append(d)
                meta.append((
                    float(env._sph_theta[i].item()),
                    float(env._sph_phi[i].item()),
                    float(env._sph_psi[i].item()),
                ))
            print(f"\r[collect] {len(rgb_list)}/{args.num_samples}", end="", flush=True)
    finally:
        env.close()

    rgb_arr = np.stack(rgb_list)
    dep_arr = np.stack(dep_list)
    np.savez_compressed(
        args.out, rgb=rgb_arr, depth=dep_arr, sph=np.array(meta, dtype=np.float32),
        atten_coeff=np.array(env.cfg.scene.camera.atten_coeff, dtype=np.float32),
    )
    finite = np.isfinite(dep_arr)
    print(f"\n[collect] 저장: {args.out}")
    print(f"[collect] rgb {rgb_arr.shape} {rgb_arr.dtype}, depth {dep_arr.shape}")
    print(f"[collect] GT depth 범위 {dep_arr[finite].min():.2f} ~ "
          f"{dep_arr[finite].max():.2f} m, 중앙값 {np.median(dep_arr[finite]):.2f} m")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
