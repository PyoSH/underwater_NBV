"""TSDF 기울기 법선이 GT 법선을 얼마나 재현하는가 — 관측 ch2의 배포 가능성.

왜 필요한가
-----------
nbuv/pixel 품질 모델은 입사각 cos(th)를 쓴다. 보상은 학습 전용이라 GT 법선을
써도 되지만, **관측 ch2는 배포에서 계산되어야** 하므로 GT를 쓰면 정책이
실기에 없는 신호에 의존한다(imitation gap). 실기에서 얻을 수 있는 법선은
재구성 자체에서 나오는 ∇TSDF뿐이다. 그런데 우리 해상도(voxel 10 cm, 물체
6~11 voxel)에서 기울기가 쓸 만한지는 재본 적이 없다.

측정
----
psi=1.6 m(nbuv 최적 거리)에서 방위를 옮기며 관측을 누적하고, 매 시점 후
관측된 GT 표면 voxel에서 ∇TSDF(중앙차분)와 GT 법선의 각도 오차를 잰다.
|cos| 기준(면의 어느 쪽인지 무관)이라 각도는 0~90도.
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--mesh_pool", type=str, required=True)
parser.add_argument("--max_solidity", type=float, default=0.5)
parser.add_argument("--psi", type=float, default=1.6)
parser.add_argument("--n_view", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, ".")
from envs.env_cfg import NBVBROVEnvCfg   # noqa: E402
from envs.env import NBVBROVEnv          # noqa: E402


def tsdf_normals(tsdf: torch.Tensor, weight: torch.Tensor):
    """중앙차분 ∇TSDF. 이웃 6개가 모두 관측된 voxel만 신뢰(valid)."""
    g = torch.zeros(*tsdf.shape, 3, device=tsdf.device)
    g[:, 1:-1, :, :, 0] = tsdf[:, 2:, :, :] - tsdf[:, :-2, :, :]
    g[:, :, 1:-1, :, 1] = tsdf[:, :, 2:, :] - tsdf[:, :, :-2, :]
    g[:, :, :, 1:-1, 2] = tsdf[:, :, :, 2:] - tsdf[:, :, :, :-2]
    obs = weight > 0
    v = torch.zeros_like(obs)
    v[:, 1:-1, 1:-1, 1:-1] = (obs[:, 2:, 1:-1, 1:-1] & obs[:, :-2, 1:-1, 1:-1]
                              & obs[:, 1:-1, 2:, 1:-1] & obs[:, 1:-1, :-2, 1:-1]
                              & obs[:, 1:-1, 1:-1, 2:] & obs[:, 1:-1, 1:-1, :-2])
    n = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # ── 완화형: 축마다 관측된 이웃이 한쪽이라도 있으면 한쪽차분 ──
    # 10 cm voxel(물체 6~11칸)에서 6-이웃 완비는 관측 표면의 5~21%뿐이라
    # (2026-09-08 실측) ch2에 쓰기엔 너무 좁다. 한쪽차분은 잡음이 크지만
    # 가용 범위가 넓다 — 둘 다 재서 정확도/가용성 교환을 본다.
    gr = torch.zeros_like(g)
    vr = torch.ones_like(obs)
    for ax in range(3):
        fwd = torch.zeros_like(tsdf); bwd = torch.zeros_like(tsdf)
        of = torch.zeros_like(obs);   ob = torch.zeros_like(obs)
        sl_c = [slice(None)] * 4; sl_f = [slice(None)] * 4; sl_b = [slice(None)] * 4
        sl_c[ax + 1] = slice(1, -1); sl_f[ax + 1] = slice(2, None); sl_b[ax + 1] = slice(None, -2)
        fwd[tuple(sl_c)] = tsdf[tuple(sl_f)] - tsdf[tuple(sl_c)]
        bwd[tuple(sl_c)] = tsdf[tuple(sl_c)] - tsdf[tuple(sl_b)]
        of[tuple(sl_c)] = obs[tuple(sl_f)]
        ob[tuple(sl_c)] = obs[tuple(sl_b)]
        both = of & ob
        d = torch.where(both, 0.5 * (fwd + bwd), torch.where(of, fwd, bwd))
        gr[..., ax] = d
        vr = vr & (of | ob)
    vr = vr & obs
    nr = gr / gr.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return n, v, obs, nr, vr


def main() -> int:
    cfg = NBVBROVEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.curriculum_enabled = False
    cfg.coverage_terminal = 1.1
    cfg.use_tiled_camera = False
    cfg.mesh_pool_manifest = args.mesh_pool
    cfg.mesh_pool_split = "train"
    cfg.mesh_pool_max_solidity = args.max_solidity
    cfg.mesh_pool_require_texture = False
    cfg.quality_model = "nbuv"
    env = NBVBROVEnv(cfg)
    env.reset()
    env._sph_phi[:] = (cfg.phi_min + cfg.phi_max) / 2
    env._sph_psi[:] = args.psi
    act = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

    print(f"\n[normals] psi={args.psi} m, voxel {cfg.tsdf.voxel_size} m, "
          f"물체 {getattr(env,'_n_mesh_objects',1)}종 × env {env.num_envs}")
    print(f"{'시점':>4}{'관측surf':>10}  | 완비(6이웃 중앙차분) / 완화(한쪽차분 허용): n(관측 대비) 각오차 중앙 <30°비율")
    for k in range(args.n_view):
        env._sph_theta[:] = 2 * math.pi * k / args.n_view
        env.step(act)
        n_est, valid, obs, n_rel, valid_rel = tsdf_normals(env._tsdf_vol, env._weight_vol)
        surf = env._surf_vol
        m_obs = obs & surf
        line = f"{k+1:>4}{int(m_obs.sum()):>10}"
        for tag, nn, vv in (("완비", n_est, valid), ("완화", n_rel, valid_rel)):
            m = vv & surf
            cos = (nn * env._surf_normal).sum(-1).abs().clamp(max=1.0)
            ang = torch.rad2deg(torch.acos(cos))[m]
            if ang.numel() == 0:
                line += f" | {tag}: 없음"; continue
            q = lambda p: float(torch.quantile(ang, p))
            line += (f" | {tag} n={int(m.sum()):>5} ({int(m.sum())/max(int(m_obs.sum()),1)*100:>3.0f}%)"
                     f" 중앙 {q(0.5):>5.1f}° <30° {float((ang<30).float().mean())*100:>3.0f}%")
        print(line, flush=True)
    print("\n  기준: 각오차 중앙값 <30°이고 이웃완비 비율이 관측의 절반 이상이면 "
          "관측 ch2에 쓸 만하다. 무작위 법선의 |cos| 각오차 중앙값은 60°.")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    app.close()
    sys.exit(code)
