from __future__ import annotations
import argparse, os, sys, time, math
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="GenNBV Quality-Aware PPO — Beer-Lambert voxel quality")
parser.add_argument("--num_envs",       type=int,   default=1)
parser.add_argument("--total_steps",    type=int,   default=500_000)
parser.add_argument("--rollout_steps",  type=int,   default=256)
parser.add_argument("--ppo_epochs",     type=int,   default=6)
parser.add_argument("--minibatch_size", type=int,   default=256)
parser.add_argument("--lr",             type=float, default=3e-4)
parser.add_argument("--gamma",          type=float, default=0.99)
parser.add_argument("--gae_lambda",     type=float, default=0.95)
parser.add_argument("--clip_eps",       type=float, default=0.2)
parser.add_argument("--ent_coef",       type=float, default=0.05)
parser.add_argument("--vf_coef",        type=float, default=0.5)
parser.add_argument("--max_grad_norm",  type=float, default=0.5)
parser.add_argument("--target_kl",      type=float, default=0.02)
parser.add_argument("--lr_decay",       action="store_true")
parser.add_argument("--ckpt_dir",       type=str,   default="/workspace/checkpoints")
parser.add_argument("--save_interval",  type=int,   default=10)
parser.add_argument("--resume",         type=str,   default=None)
parser.add_argument("--wandb_project",  type=str,   default="RL_NBV_DR")
parser.add_argument("--wandb_name",     type=str,   default="UW_NBV_DR")
parser.add_argument("--eval_interval",  type=int,   default=10,
                    help="N 롤아웃마다 eval (0=비활성)")
parser.add_argument("--eval_episodes",  type=int,   default=20)

AppLauncher.add_app_launcher_args(parser)

if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args         = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 이후 import ───────────────────────────────────────────────────
import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.envCfg              import OceanEnvCfg
from env.env_GenNBV_quality  import OceanEnvGenNBVQuality
from algorithm.algo_UW_NBV   import (Actor, Critic, RolloutBuffer, PPOConfig,
                                     make_env_action, explained_variance, ppo_update)


# ══════════════════════════════════════════════════════════════════════════════
# Eval
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(env: OceanEnvGenNBVQuality, actor: Actor,
             device, n_episodes: int) -> dict:
    actor.eval()
    results = dict(returns=[], lengths=[], coverages=[], successes=[])

    obs, _   = env.reset()
    vox      = obs["vox_actor"]
    img      = obs["img_semantic"]
    scalar   = obs["extra_info"]

    E         = env.num_envs
    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)
    completed = [0] * E
    max_steps = env.max_episode_length * n_episodes * 2

    with torch.no_grad():
        for _ in range(max_steps):
            if min(completed) >= n_episodes:
                break
            pose_act   = actor.greedy(vox, img, scalar)
            env_action = make_env_action(pose_act, E, device)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            ep_return += reward
            ep_len    += 1

            for eid in (terminated | truncated).nonzero(as_tuple=True)[0].tolist():
                if completed[eid] < n_episodes:
                    results["returns"]  .append(ep_return[eid].item())
                    results["lengths"]  .append(ep_len[eid].item())
                    results["coverages"].append(env._terminal_coverage_q[eid].item())
                    results["successes"].append(float(terminated[eid].item()))
                    completed[eid] += 1
                ep_return[eid] = 0.0
                ep_len[eid]    = 0

            vox    = next_obs["vox_actor"]
            img    = next_obs["img_semantic"]
            scalar = next_obs["extra_info"]

    actor.train()
    return {
        "eval/mean_return":   np.mean(results["returns"])   if results["returns"]   else 0.0,
        "eval/mean_length":   np.mean(results["lengths"])   if results["lengths"]   else 0.0,
        "eval/mean_coverage": np.mean(results["coverages"]) if results["coverages"] else 0.0,
        "eval/success_rate":  np.mean(results["successes"]) if results["successes"] else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 환경 ──────────────────────────────────────────────────────────────────
    env_cfg = OceanEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env_cfg.visual.num_seq_actor  = 2
    env_cfg.visual.num_seq_critic = 2
    env_cfg.visual.h = 64
    env_cfg.visual.w = 64

    env_cfg.use_visit_map = False
    env_cfg.k_explore     = 0.0

    env_cfg.k_c               = 0.0    # binary coverage 보상 비활성
    env_cfg.k_c_q             = 5.0    # quality-weighted coverage 보상
    env_cfg.k_x               = 0.0
    env_cfg.c_step            = 0.02   # 0.1 → 0.02: quality reward와 scale 맞춤
    env_cfg.k_still           = 0.05
    env_cfg.coverage_terminal = 0.60
    env_cfg.coverage_bonus    = 30.0
    env_cfg.jerlov_dr_enabled = True

    env    = OceanEnvGenNBVQuality(cfg=env_cfg, render_mode="rgb_array")
    device = env.device
    E      = env.num_envs
    H, W   = env_cfg.visual.h, env_cfg.visual.w
    M      = 2
    Nx, Ny, Nz = env_cfg.tsdf.vol_dim
    T      = args.rollout_steps

    # ── 네트워크 ──────────────────────────────────────────────────────────────
    actor  = Actor (img_ch=M, scalar_dim=3, n_actions=6, H=H, W=W).to(device)
    critic = Critic(img_ch=M, scalar_dim=4,              H=H, W=W).to(device)

    optimizer_actor  = torch.optim.Adam(actor .parameters(), lr=args.lr)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=args.lr * 2)

    ppo_cfg = PPOConfig(
        ppo_epochs     = args.ppo_epochs,
        minibatch_size = args.minibatch_size,
        clip_eps       = args.clip_eps,
        ent_coef       = args.ent_coef,
        vf_coef        = args.vf_coef,
        max_grad_norm  = args.max_grad_norm,
        gamma          = args.gamma,
        lam            = args.gae_lambda,
        target_kl      = args.target_kl,
    )

    # ── Rollout Buffer ────────────────────────────────────────────────────────
    buf = RolloutBuffer(
        T=T, E=E,
        Nx=Nx, Ny=Ny, Nz=Nz,
        M=M, H=H, W=W,
        scalar_dim=3, scalar_dim_critic=4,
        device=device,
    )

    global_step   = 0
    rollout_idx   = 0
    last_log_step = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        actor           .load_state_dict(ckpt["actor"])
        critic          .load_state_dict(ckpt["critic"])
        optimizer_actor .load_state_dict(ckpt["optimizer_actor"])
        optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
        global_step   = ckpt.get("global_step", 0)
        rollout_idx   = ckpt.get("rollout_idx",  0)
        last_log_step = global_step
        print(f"[resume] {args.resume}  (step={global_step})", flush=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or "genNBV_quality",
            config=vars(args),
            resume="allow",
        )
        wandb.watch(actor,  log="gradients", log_freq=200)
        wandb.watch(critic, log="gradients", log_freq=200)
        wandb.define_metric("train/global_step")
        wandb.define_metric("*", step_metric="train/global_step")

    ckpt_dir = Path(args.ckpt_dir)
    if use_wandb:
        ckpt_dir = ckpt_dir / wandb.run.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 초기 관측 ─────────────────────────────────────────────────────────────
    obs, _   = env.reset()
    vox      = obs["vox_actor"]
    img      = obs["img_semantic"]
    scalar   = obs["extra_info"]
    scalar_c = obs["critic_scalar"]

    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)
    finished  = dict(returns=[], lengths=[], coverages=[], coverages_binary=[], terminated=[],
                     diag_never=[], diag_partial=[], diag_full=[])

    # ══════════════════════════════════════════════════════════════════════════
    # 학습 루프
    # ══════════════════════════════════════════════════════════════════════════
    while global_step < args.total_steps:

        if args.lr_decay:
            frac = max(1.0 - global_step / args.total_steps, 0.0)
            for pg in optimizer_actor .param_groups: pg["lr"] = args.lr       * frac
            for pg in optimizer_critic.param_groups: pg["lr"] = args.lr * 2.0 * frac

        buf.reset()
        actor.eval(); critic.eval()

        rew_cov_list, rew_pen_list, rew_succ_list, rew_stall_list = [], [], [], []

        # ── Rollout 수집 ──────────────────────────────────────────────────────
        for _ in range(T):
            with torch.no_grad():
                pose_act, logprob, _ = actor.sample(vox, img, scalar)
                value                = critic(vox, img, scalar_c)

            env_action = make_env_action(pose_act, E, device)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            rew_cov_list  .append(env._last_rew_coverage  .mean().item())
            rew_pen_list  .append(env._last_rew_penalty   .mean().item())
            rew_succ_list .append(env._last_success_reward.mean().item())
            rew_stall_list.append(env._last_rew_stall     .mean().item())

            done_any = terminated | truncated
            buf.add(vox, img, scalar, scalar_c,
                    pose_act, logprob, reward, terminated.float(), value)

            ep_return += reward
            ep_len    += 1

            for eid in done_any.nonzero(as_tuple=True)[0].tolist():
                finished["returns"]   .append(ep_return[eid].item())
                finished["lengths"]   .append(ep_len[eid].item())
                finished["coverages"]       .append(env._terminal_coverage_q[eid].item())
                finished["coverages_binary"].append(env._terminal_coverage[eid].item())
                finished["terminated"]      .append(float(terminated[eid].item()))
                finished["diag_never"]  .append(env._diag_gt_never[eid].item())
                finished["diag_partial"].append(env._diag_gt_partial[eid].item())
                finished["diag_full"]   .append(env._diag_gt_full[eid].item())
                ep_return[eid] = 0.0
                ep_len[eid]    = 0

            vox      = next_obs["vox_actor"]
            img      = next_obs["img_semantic"]
            scalar   = next_obs["extra_info"]
            scalar_c = next_obs["critic_scalar"]
            global_step += E

        # ── GAE & PPO ─────────────────────────────────────────────────────────
        with torch.no_grad():
            last_val = critic(vox, img, scalar_c)
        buf.compute_gae(last_val, ppo_cfg.gamma, ppo_cfg.lam)

        actor.train(); critic.train()
        stats = ppo_update(actor, critic, optimizer_actor, optimizer_critic, buf, ppo_cfg)
        rollout_idx += 1

        # ── 로그 ──────────────────────────────────────────────────────────────
        ev  = explained_variance(buf.values.reshape(-1), buf.returns.reshape(-1))
        fps = int(global_step / (time.time() - t0 + 1e-8))

        log: dict = {
            "train/mean_step_reward":   buf.rewards.mean().item(),
            "train/policy_loss":        stats["policy_loss"],
            "train/value_loss":         stats["value_loss"],
            "train/entropy":            stats["entropy"],
            "train/approx_kl":          stats["approx_kl"],
            "train/early_stop":         float(stats.get("early_stop", False)),
            "train/explained_variance": ev,
            "train/lr_actor":           optimizer_actor .param_groups[0]["lr"],
            "train/lr_critic":          optimizer_critic.param_groups[0]["lr"],
            "train/fps":                fps,
            "train/global_step":        global_step,
            "reward/coverage_q":        np.mean(rew_cov_list),
            "reward/penalty":           np.mean(rew_pen_list),
            "reward/success":           np.mean(rew_succ_list),
            "reward/stall":             np.mean(rew_stall_list),
            "env0/theta_deg":           math.degrees(env._sph_theta[0].item()),
            "env0/phi_deg":             math.degrees(env._sph_phi[0].item()),
            "env0/psi":                 env._sph_psi[0].item(),
            "env0/coverage_binary":     env.curr_coverage[0].item(),
            "env0/coverage_q":          env.curr_coverage_q[0].item(),
            "env0/vox_unknown_ratio":   (vox[0, 0] > 0.5).float().mean().item(),
            "env0/vox_quality_mean":    vox[0, 2][vox[0, 2] > 0].mean().item()
                                        if (vox[0, 2] > 0).any() else 0.0,
            "env0/quality_mu":          env._quality_mu[0].item(),
        }

        if finished["returns"]:
            log["episode/mean_return"]          = np.mean(finished["returns"])
            log["episode/mean_length"]          = np.mean(finished["lengths"])
            log["episode/mean_coverage_q"]      = np.mean(finished["coverages"])
            log["episode/mean_coverage_binary"] = np.mean(finished["coverages_binary"])
            log["episode/success_rate"]         = np.mean(finished["terminated"])
            log["episode/timeout_rate"]         = 1.0 - np.mean(finished["terminated"])
            # 진단: GT surface voxel quality 분포
            log["diag/gt_never"]   = np.mean(finished["diag_never"])    # quality=0 비율
            log["diag/gt_partial"] = np.mean(finished["diag_partial"])  # 0<q<1 비율
            log["diag/gt_full"]    = np.mean(finished["diag_full"])     # q≥1 비율
            finished = dict(returns=[], lengths=[], coverages=[], coverages_binary=[], terminated=[],
                            diag_never=[], diag_partial=[], diag_full=[])

        if use_wandb:
            wandb.log(log, step=global_step)

        LOG_EVERY = 10_000
        if global_step - last_log_step >= LOG_EVERY:
            last_log_step = global_step
            cov_q_str = (
                f"  cov_q={log['episode/mean_coverage_q']:.3f}"
                f"  cov_bin={log['episode/mean_coverage_binary']:.3f}"
                if "episode/mean_coverage_q" in log else ""
            )

            weight_filled   = (env._weight_vol > 0).float().mean().item()
            curr_cov_now    = env.curr_coverage.mean().item()
            curr_cov_q_now  = env.curr_coverage_q.mean().item()
            surf_total      = env._total_surf_voxels.mean().item()

            print(
                f"[{global_step:9d}]"
                f"  rew={log['train/mean_step_reward']:+.6f}"
                f"  pl={stats['policy_loss']:.4f}"
                f"  vl={stats['value_loss']:.4f}"
                f"  ent={stats['entropy']:.3f}"
                f"  ev={ev:.3f}"
                f"{cov_q_str}  fps={fps}",
                flush=True,
            )
            print(
                f"  [reward]"
                f"  cov_q={np.mean(rew_cov_list):+.4f}"
                f"  penalty={np.mean(rew_pen_list):.4f}"
                f"  success={np.mean(rew_succ_list):+.4f}"
                f"  stall={np.mean(rew_stall_list):.4f}"
                f"  net={buf.rewards.mean().item():+.4f}",
                flush=True,
            )
            print(
                f"  [coverage]"
                f"  binary={curr_cov_now:.4f}"
                f"  quality={curr_cov_q_now:.4f}"
                f"  weight_filled={weight_filled:.6f}"
                f"  surf_voxels={surf_total:.0f}",
                flush=True,
            )
            print(
                f"  [quality]"
                f"  vox_quality_mean={log['env0/vox_quality_mean']:.4f}"
                f"  mu={env._quality_mu[0].item():.4f}",
                flush=True,
            )
            if "diag/gt_never" in log:
                print(
                    f"  [diag/GT]"
                    f"  never={log['diag/gt_never']:.3f}"
                    f"  partial={log['diag/gt_partial']:.3f}"
                    f"  full={log['diag/gt_full']:.3f}",
                    flush=True,
                )

        # ── Eval ──────────────────────────────────────────────────────────────
        if args.eval_interval > 0 and rollout_idx % args.eval_interval == 0:
            eval_metrics = run_eval(env, actor, device, args.eval_episodes)
            if use_wandb:
                wandb.log(eval_metrics, step=global_step)
            print(
                f"  [EVAL]"
                f"  return={eval_metrics['eval/mean_return']:.2f}"
                f"  cov_q={eval_metrics['eval/mean_coverage']:.3f}"
                f"  success={eval_metrics['eval/success_rate']:.2f}"
                f"  len={eval_metrics['eval/mean_length']:.1f}",
                flush=True,
            )
            obs, _   = env.reset()
            vox      = obs["vox_actor"]
            img      = obs["img_semantic"]
            scalar   = obs["extra_info"]
            scalar_c = obs["critic_scalar"]
            ep_return = torch.zeros(E, device=device)
            ep_len    = torch.zeros(E, device=device, dtype=torch.long)

        # ── 체크포인트 ────────────────────────────────────────────────────────
        if rollout_idx % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"genNBV_quality_step_{global_step:010d}.pt"
            torch.save({
                "global_step":     global_step,
                "rollout_idx":     rollout_idx,
                "actor":           actor .state_dict(),
                "critic":          critic.state_dict(),
                "optimizer_actor": optimizer_actor .state_dict(),
                "optimizer_critic":optimizer_critic.state_dict(),
                "args":            vars(args),
                "env_type":        "gennbv_quality",
            }, ckpt_path)
            print(f"[ckpt] → {ckpt_path}", flush=True)

    env.close()
    if use_wandb:
        wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
