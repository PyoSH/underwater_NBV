"""
step_3 NBV 학습 런처 (연속 액션 PPO)
=======================================
`step_1_NBV/train/train_GenNBV_quality.py` 구조를 따르되 step_3 환경/연속 액션에
맞춰 조정했다.

step_1 대비 주요 차이
---------------------
1. **환경**: `NBVBROVEnv`(물리 기반) — 정책스텝 1회 = NBV 결정 1회 = 5초
   (decimation=500). step_1은 순간이동이라 정책스텝이 1/60초였다.
2. **액션**: 연속 3-dim (Δθ,Δφ,Δψ). `make_env_action()`(이산→one-hot) 불필요.
3. **env_cfg를 명시적으로 오버라이드하지 않는다** — step_1은 envCfg 기본값을
   train 스크립트가 전부 덮어쓰는 구조라 "envCfg 기본값 ≠ 실제 학습값"이라는
   함정이 있었다(2026-08-26 조사에서 이 함정 때문에 내가 폐기된 값을 step_3에
   이식했던 것이 드러남). step_3는 **`envs/env_cfg.py`가 유일한 정본**이고
   여기서는 CLI로 넘어온 것만 조정한다.

사용법 (isaac-lab-base 컨테이너 안)
-----------------------------------
python.sh train.py --headless --num_envs 8 --total_steps 50000
python.sh train.py --headless --num_envs 4 --total_steps 2000 --smoke   # 스모크
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="step_3 NBV 연속 액션 PPO")
parser.add_argument("--num_envs",       type=int,   default=8)
parser.add_argument("--total_steps",    type=int,   default=50_000,
                    help="총 env-step 수(= 정책 결정 수 × num_envs)")
parser.add_argument("--rollout_steps",  type=int,   default=32,
                    help="롤아웃 길이(정책 결정 단위). step_3는 1결정=5초라 "
                         "step_1(512)보다 훨씬 짧게 잡아야 한다")
parser.add_argument("--ppo_epochs",     type=int,   default=6)
parser.add_argument("--minibatch_size", type=int,   default=32,
                    help="작을수록 롤아웃당 gradient step이 늘어난다. VRAM에도 직결 — "
                         "3D CNN 활성값이 커서 128로 돌렸을 때 학습 중 14.7GB까지 "
                         "올라가 GPU를 독점했다(실측)")
parser.add_argument("--lr",             type=float, default=3e-4)
parser.add_argument("--gamma",          type=float, default=0.99)
parser.add_argument("--gae_lambda",     type=float, default=0.95)
parser.add_argument("--clip_eps",       type=float, default=0.2)
parser.add_argument("--ent_coef",       type=float, default=0.03)
parser.add_argument("--vf_coef",        type=float, default=0.5)
parser.add_argument("--max_grad_norm",  type=float, default=0.5)
parser.add_argument("--target_kl",      type=float, default=0.05,
                    help="step_1은 0.02였지만 그건 **이산 6-way** 정책 기준이다. "
                         "연속 3차원 tanh-Gaussian은 log-prob 스케일이 달라 KL이 "
                         "자연히 크다 — 1차 학습 시도에서 실측 KL이 0.025~0.047로 "
                         "매 롤아웃 early_stop이 걸려 1에포크(미니배치 4개)만 돌았고 "
                         "학습이 전혀 진행되지 않았다")
parser.add_argument("--log_std_init",   type=float, default=-0.5)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--ckpt_dir",       type=str,   default="checkpoints")
parser.add_argument("--save_interval",  type=int,   default=20, help="N 롤아웃마다 저장")
parser.add_argument("--smoke", action="store_true",
                    help="스모크 모드: 소규모로 몇 롤아웃만 돌고 종료, 이상치 검사 강화")

AppLauncher.add_app_launcher_args(parser)
if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 이후 import ───────────────────────────────────────────────────
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envs.env_cfg import NBVBROVEnvCfg
from envs.env import NBVBROVEnv
from algorithm.algo_nbv_continuous import (
    Actor, Critic, RolloutBuffer, PPOConfig, explained_variance, ppo_update,
)


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # env_cfg는 envs/env_cfg.py가 정본 — 여기서 보상/물리 가중치를 덮어쓰지 않는다.
    env_cfg = NBVBROVEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = NBVBROVEnv(cfg=env_cfg)
    device = env.device
    E = env.num_envs
    H, W = env_cfg.visual.h, env_cfg.visual.w
    M = 2                                   # img_semantic 채널(최근 2프레임)
    Nx, Ny, Nz = env_cfg.tsdf.vol_dim
    A = env_cfg.action_space
    T = args.rollout_steps

    actor  = Actor(img_ch=M, scalar_dim=3, action_dim=A, H=H, W=W,
                   log_std_init=args.log_std_init).to(device)
    critic = Critic(img_ch=M, scalar_dim=4, H=H, W=W).to(device)
    opt_a = torch.optim.Adam(actor.parameters(),  lr=args.lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)

    ppo_cfg = PPOConfig(
        ppo_epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm, gamma=args.gamma, lam=args.gae_lambda,
        target_kl=args.target_kl,
    )
    buf = RolloutBuffer(T, E, Nx, Ny, Nz, M, H, W,
                        scalar_dim=3, scalar_dim_critic=4,
                        action_dim=A, device=device)

    # 커리큘럼 총 스텝을 알려준다(0이면 비활성) — env_cfg.curriculum_* 참조
    if env_cfg.curriculum_enabled and env_cfg.curriculum_total_steps <= 0:
        env.cfg.curriculum_total_steps = args.total_steps
        print(f"[train] curriculum_total_steps = {args.total_steps} (자동 설정)")

    obs, _ = env.reset()
    ep_return = torch.zeros(E, device=device)
    ep_len = torch.zeros(E, dtype=torch.long, device=device)
    done_returns: list[float] = []
    done_covs: list[float] = []

    os.makedirs(args.ckpt_dir, exist_ok=True)
    n_rollouts = max(1, args.total_steps // (T * E))
    if args.smoke:
        n_rollouts = min(n_rollouts, 3)
    print(f"[train] envs={E} rollout={T} → 롤아웃당 {T*E} env-step, 총 {n_rollouts} 롤아웃")

    t0 = time.time()
    for it in range(n_rollouts):
        buf.reset()
        for _t in range(T):
            with torch.no_grad():
                a, u, logp, _ent = actor.sample(
                    obs["vox_actor"], obs["img_semantic"], obs["extra_info"])
                value = critic(
                    obs["vox_actor"], obs["img_semantic"], obs["critic_scalar"])

            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = (terminated | truncated).float()

            buf.add(obs["vox_actor"], obs["img_semantic"],
                    obs["extra_info"], obs["critic_scalar"],
                    u, logp, reward, done, value)

            ep_return += reward
            ep_len += 1
            for eid in (terminated | truncated).nonzero(as_tuple=True)[0].tolist():
                done_returns.append(ep_return[eid].item())
                # `curr_coverage`가 아니라 `terminal_coverage`를 읽어야 한다 —
                # step()이 반환될 시점엔 done env의 curr_coverage가 이미 0으로
                # 리셋돼 있다(envs/env.py `_reset_idx()` 주석 참조).
                done_covs.append(env.terminal_coverage[eid].item())
                ep_return[eid] = 0.0
                ep_len[eid] = 0

            obs = next_obs

        with torch.no_grad():
            last_v = critic(obs["vox_actor"], obs["img_semantic"], obs["critic_scalar"])
        buf.compute_gae(last_v, ppo_cfg.gamma, ppo_cfg.lam)
        stats = ppo_update(actor, critic, opt_a, opt_c, buf, ppo_cfg)

        flat = buf.flat()
        ev = explained_variance(flat["old_values"], flat["returns"])
        rew_mean = buf.rewards.mean().item()
        terms = env.last_reward_terms
        term_abs = {k: v.abs().mean().item() for k, v in terms.items()}
        tot = sum(term_abs.values()) or 1.0

        print(
            f"[{it:03d}] rew={rew_mean:+.4f} "
            f"ret={np.mean(done_returns[-20:]) if done_returns else float('nan'):+.3f} "
            f"cov={np.mean(done_covs[-20:]) if done_covs else float('nan'):.3f} "
            f"pl={stats['policy_loss']:+.4f} vl={stats['value_loss']:.3f} "
            f"ent={stats['entropy']:+.3f} kl={stats['approx_kl']:.4f} "
            f"ev={ev:+.3f} early_stop={stats['early_stop']} "
            f"| success={term_abs['success']/tot*100:.0f}% "
            f"dist={env.last_dist_moved.mean().item():.3f} "
            f"logstd={actor.log_std.mean().item():+.2f} "
            f"({time.time()-t0:.0f}s)"
        )

        # ── 이상치 검사 (스모크 모드에서 특히 중요) ──
        bad = []
        if not np.isfinite(rew_mean):
            bad.append("reward NaN/Inf")
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl"):
            if not np.isfinite(stats[k]):
                bad.append(f"{k} NaN/Inf")
        if not torch.isfinite(actor.log_std).all():
            bad.append("log_std NaN/Inf")
        if bad:
            print(f"[train] FAIL — {', '.join(bad)}")
            env.close()
            return 1

        if (it + 1) % args.save_interval == 0 and not args.smoke:
            path = os.path.join(args.ckpt_dir, f"nbv_step3_{it+1:05d}.pt")
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "it": it + 1, "args": vars(args)}, path)
            print(f"[train] saved {path}")

    print(f"[train] {'SMOKE PASSED' if args.smoke else 'DONE'} — "
          f"{n_rollouts} 롤아웃, {time.time()-t0:.0f}s")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
