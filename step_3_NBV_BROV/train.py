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
import math
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
parser.add_argument("--wandb_project", type=str, default="step3_NBV_BROV",
                    help="wandb 프로젝트. --no_wandb로 끌 수 있다")
parser.add_argument("--wandb_name",    type=str, default=None,
                    help="run 이름(미지정 시 wandb가 자동 생성)")
parser.add_argument("--wandb_entity",  type=str, default=os.environ.get("WANDB_ENTITY"),
                    help="wandb entity. 미지정 시 로그인 계정 기본값을 쓴다 "
                         "(계정명에 대시가 붙는 경우가 있으니 주의)")
parser.add_argument("--no_wandb", action="store_true", help="wandb 로깅 비활성화")

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

# wandb가 없거나 로그인이 안 돼 있어도 학습 자체는 계속돼야 한다 — 9시간짜리 런이
# 로깅 문제로 죽으면 손해가 크다.
try:
    import wandb
except ImportError:
    wandb = None
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
    if env_cfg.curriculum_enabled and env_cfg.curriculum_adaptive:
        # 적응형은 총 스텝 수를 몰라도 된다 — 성공률만 보고 올린다.
        print(f"[train] 커리큘럼: 적응형 (게이트 성공률 "
              f"{env_cfg.curriculum_success_gate}, 시작 "
              f"{env_cfg.curriculum_coverage_terminal_start} → 상한 "
              f"{env_cfg.curriculum_coverage_terminal_end})")
    elif env_cfg.curriculum_enabled and env_cfg.curriculum_total_steps <= 0:
        env.cfg.curriculum_total_steps = args.total_steps
        print(f"[train] curriculum_total_steps = {args.total_steps} (자동 설정)")

    # ── wandb ────────────────────────────────────────────────────────────────
    use_wandb = (not args.no_wandb) and (not args.smoke) and wandb is not None
    if use_wandb:
        try:
            wandb.init(project=args.wandb_project, name=args.wandb_name,
                       entity=args.wandb_entity, config=vars(args), resume="allow")
            wandb.define_metric("train/global_step")
            wandb.define_metric("*", step_metric="train/global_step")
            print(f"[train] wandb: {wandb.run.url}")
        except Exception as exc:                                      # noqa: BLE001
            print(f"[train] wandb 초기화 실패 — 로깅 없이 계속한다: {exc}")
            use_wandb = False
    elif args.no_wandb:
        print("[train] wandb 비활성화(--no_wandb)")
    elif wandb is None:
        print("[train] wandb 미설치 — 로깅 없이 진행")

    obs, _ = env.reset()
    ep_return = torch.zeros(E, device=device)
    ep_len = torch.zeros(E, dtype=torch.long, device=device)
    done_returns: list[float] = []
    done_covs: list[float] = []
    done_covs_bin: list[float] = []
    done_lens: list[float] = []
    done_term: list[float] = []   # 1=커버리지 달성 종료, 0=시간초과

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
                # cov는 학습이 실제로 최적화하는 지표(quality 모드면 정규화
                # coverage_q)를 쓴다. binary는 2026-08-26 baseline과 같은 축에서
                # 보기 위해 별도로 남긴다.
                done_covs.append(env.terminal_coverage_q[eid].item()
                                 if env.cfg.use_quality_coverage
                                 else env.terminal_coverage[eid].item())
                done_covs_bin.append(env.terminal_coverage[eid].item())
                done_lens.append(float(ep_len[eid].item()))
                # terminated=커버리지 목표 달성, truncated=시간초과 — 커리큘럼이
                # 조여지는지 보려면 이 둘을 반드시 구분해야 한다.
                done_term.append(float(terminated[eid].item()))
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
            f"ev={ev:+.3f} vclip={stats['value_clip_frac']:.2f} "
            f"rstd={flat['returns'].std().item():.1f} "
            f"early_stop={stats['early_stop']} "
            # n_updates=0이면 그 롤아웃은 갱신 0회로 버려진 것이다. 이게
            # 안 보여서 2026-08-29 검증에서 롤아웃 4개가 조용히 낭비됐다.
            f"nupd={stats['n_updates']} "
            f"| success={term_abs['success']/tot*100:.0f}% "
            f"dist={env.last_dist_moved.mean().item():.3f} "
            f"logstd={actor.log_std.mean().item():+.2f} "
            f"({time.time()-t0:.0f}s)"
        )

        if use_wandb:
            global_step = (it + 1) * T * E
            log = {
                "train/global_step":   global_step,
                "train/rollout":       it,
                "train/reward_mean":   rew_mean,
                "train/policy_loss":   stats["policy_loss"],
                "train/value_loss":    stats["value_loss"],
                "train/entropy":       stats["entropy"],
                "train/approx_kl":     stats["approx_kl"],
                # 매 롤아웃 True면 샘플 재사용이 1에폭뿐이라는 뜻 → target_kl 상향 검토
                "train/early_stop":    float(stats["early_stop"]),
                "train/n_updates":     stats["n_updates"],
                "train/explained_var": ev,
                # value clipping이 실제로 제동을 거는 비율. 0에 가까우면
                # 기준점 수정이 무효라는 뜻이다.
                "train/value_clip_frac": stats["value_clip_frac"],
                # 리턴 분포 이동 여부 — ev 하락의 대안 가설(적응형 커리큘럼이
                # 난이도를 올려 리턴 분포가 이동하면 critic이 멀쩡해도 ev가
                # 떨어진다)을 value clipping 가설과 구분하기 위한 계측.
                "train/return_mean": flat["returns"].mean().item(),
                "train/return_std":  flat["returns"].std().item(),
                "train/value_mean":  flat["old_values"].mean().item(),
                "train/log_std":       actor.log_std.mean().item(),
                "train/lr":            opt_a.param_groups[0]["lr"],
                "diag/dist_moved":     env.last_dist_moved.mean().item(),
                # 커리큘럼이 실제로 조여지고 있는지 — cov와 함께 봐야 의미가 있다
                "curriculum/coverage_terminal": float(env._current_coverage_terminal()),
                # 적응형 커리큘럼이 무엇을 보고 올리는지 — 임계값이 정체하면
                # 이 값이 게이트(기본 0.7) 아래에 머물고 있다는 뜻이다.
                "curriculum/success_ema": float(env.curriculum_success_ema),
                "perf/env_steps_per_sec": global_step / max(time.time() - t0, 1e-6),
            }
            for k, v in term_abs.items():
                log[f"reward_share/{k}"] = v / tot          # 보상 항목별 기여 비중
            if env.cfg.use_quality_coverage:
                # GT surface 품질 분포 — binary coverage로는 안 보이는
                # "봤지만 멀어서 흐릿함"을 드러낸다(step_1 diag/gt_* 대응).
                log["diag/gt_never"]   = env._diag_gt_never.mean().item()
                log["diag/gt_partial"] = env._diag_gt_partial.mean().item()
                log["diag/gt_full"]    = env._diag_gt_full.mean().item()
                log["diag/quality_mu"] = env._quality_mu.mean().item()
                log["diag/quality_Q_sat"] = env._quality_Q_sat.mean().item()
            # 정책이 한 지점에 주차하거나 클램프 한계에 붙는 축퇴 행동 감지용
            log["env0/theta_deg"] = math.degrees(env._sph_theta[0].item())
            log["env0/phi_deg"]   = math.degrees(env._sph_phi[0].item())
            log["env0/psi"]       = env._sph_psi[0].item()
            log["env0/vox_unknown_ratio"] = (
                (obs["vox_actor"][0, 0] > 0.5).float().mean().item()
            )
            if done_returns:
                log["episode/mean_return"]   = float(np.mean(done_returns[-20:]))
                log["episode/mean_coverage"] = float(np.mean(done_covs[-20:]))
                log["episode/mean_coverage_binary"] = float(np.mean(done_covs_bin[-20:]))
                log["episode/mean_length"]   = float(np.mean(done_lens[-20:]))
                log["episode/success_rate"]  = float(np.mean(done_term[-20:]))
                # 평균만 보면 "일부 에피소드는 물체를 아예 못 찾는" 이봉 구조를
                # 놓친다 — 9.3시간 런 분석에서 미결로 남았던 질문.
                if wandb is not None:
                    log["episode/coverage_hist"] = wandb.Histogram(done_covs[-200:])
            try:
                wandb.log(log, step=global_step)
            except Exception as exc:                                  # noqa: BLE001
                print(f"[train] wandb.log 실패(무시하고 계속): {exc}")

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
            if use_wandb:
                wandb.finish(exit_code=1)
            env.close()
            return 1

        if (it + 1) % args.save_interval == 0 and not args.smoke:
            path = os.path.join(args.ckpt_dir, f"nbv_step3_{it+1:05d}.pt")
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                        "it": it + 1, "args": vars(args)}, path)
            print(f"[train] saved {path}")

    print(f"[train] {'SMOKE PASSED' if args.smoke else 'DONE'} — "
          f"{n_rollouts} 롤아웃, {time.time()-t0:.0f}s")
    if use_wandb:
        wandb.finish()
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    sys.exit(code)
