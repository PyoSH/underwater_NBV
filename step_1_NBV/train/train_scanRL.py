from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

from isaaclab.app import AppLauncher

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ScanRL DQN — NBV 3D Reconstruction")
parser.add_argument("--num_envs",           type=int,   default=1)
parser.add_argument("--total_steps",        type=int,   default=500_000)
parser.add_argument("--replay_capacity",    type=int,   default=50_000)
parser.add_argument("--batch_size",         type=int,   default=32)
parser.add_argument("--lr",                 type=float, default=1e-4)
parser.add_argument("--gamma",              type=float, default=0.99)
parser.add_argument("--eps_start",          type=float, default=0.8)
parser.add_argument("--eps_end",            type=float, default=0.05)
parser.add_argument("--eps_decay",          type=float, default=0.999)
parser.add_argument("--target_update_freq", type=int,   default=1_000)
parser.add_argument("--min_replay",         type=int,   default=1_000)
parser.add_argument("--ckpt_dir",           type=str,   default="/workspace/checkpoints")
parser.add_argument("--save_interval",      type=int,   default=50_000)
parser.add_argument("--resume",             type=str,   default=None)
parser.add_argument("--wandb_project",      type=str,   default="RL_NBV")
parser.add_argument("--wandb_name",         type=str,   default="scanRL")
parser.add_argument("--eval_interval",      type=int,   default=10_000)
parser.add_argument("--eval_episodes",      type=int,   default=5)

AppLauncher.add_app_launcher_args(parser)

if "--enable_cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

args       = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── AppLauncher 이후 import ───────────────────────────────────────────────────
import math
import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.envCfg            import OceanEnvCfg
from env.env               import OceanEnv
from algorithm.algo_scanRL import (QNetwork, ReplayBuffer, DQNConfig,
                                   make_env_action, dqn_update)


# ══════════════════════════════════════════════════════════════════════════════
# Eval
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(env, q_net: QNetwork, device, n_episodes: int) -> dict:
    q_net.eval()
    results = dict(returns=[], lengths=[], coverages=[], successes=[])

    obs, _  = env.reset()
    obs_img = obs["policy"]

    E         = env.num_envs
    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)
    completed = [0] * E
    max_steps = env.max_episode_length * n_episodes * 2

    with torch.no_grad():
        for _ in range(max_steps):
            if min(completed) >= n_episodes:
                break
            action_idx = q_net(obs_img).argmax(dim=1)
            env_action = make_env_action(action_idx, E, device)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            ep_return += reward
            ep_len    += 1

            for eid in (terminated | truncated).nonzero(as_tuple=True)[0].tolist():
                if completed[eid] < n_episodes:
                    results["returns"]  .append(ep_return[eid].item())
                    results["lengths"]  .append(ep_len[eid].item())
                    results["coverages"].append(env.curr_coverage[eid].item())
                    results["successes"].append(float(terminated[eid].item()))
                    completed[eid] += 1
                ep_return[eid] = 0.0
                ep_len[eid]    = 0

            obs_img = next_obs["policy"]

    q_net.train()
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
    env_cfg.k_c = 1.0
    env_cfg.c_step = 2.0
    env_cfg.coverage_bonus = 100.0
    env_cfg.k_x = 0.02
    env_cfg.k_still = 0.0

    env_cfg.phi_min   = math.radians(20)
    env_cfg.phi_max   = math.radians(70)
    env_cfg.delta_phi = math.radians(25)

    env_cfg.visual.num_seq_actor = 6
    env_cfg.visual.num_seq_critic = 6

    env    = OceanEnv(cfg=env_cfg, render_mode="rgb_array")
    device = env.device
    E      = env.num_envs
    K_img  = env_cfg.visual.num_seq_actor   # 6 frames

    # ── 네트워크 ──────────────────────────────────────────────────────────────
    dqn_cfg = DQNConfig(
        replay_capacity    = args.replay_capacity,
        batch_size         = args.batch_size,
        gamma              = args.gamma,
        lr                 = args.lr,
        eps_start          = args.eps_start,
        eps_end            = args.eps_end,
        eps_decay          = args.eps_decay,
        target_update_freq = args.target_update_freq,
        min_replay         = args.min_replay,
    )

    q_net      = QNetwork(in_ch=K_img, n_actions=6).to(device)
    target_net = QNetwork(in_ch=K_img, n_actions=6).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=dqn_cfg.lr, eps=1e-5)
    replay    = ReplayBuffer(dqn_cfg.replay_capacity, device)

    global_step    = 0
    epsilon        = dqn_cfg.eps_start
    last_log_step  = 0
    last_eval_step = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        q_net     .load_state_dict(ckpt["q_net"])
        target_net.load_state_dict(ckpt["target_net"])
        optimizer .load_state_dict(ckpt["optimizer"])
        global_step    = ckpt.get("global_step", 0)
        epsilon        = ckpt.get("epsilon",      dqn_cfg.eps_start)
        last_log_step  = global_step
        last_eval_step = global_step
        print(f"[resume] {args.resume}  (step={global_step}, eps={epsilon:.4f})", flush=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or "scanrl-dqn",
            config=vars(args),
            resume="allow",
        )
        wandb.watch(q_net, log="gradients", log_freq=500)

    ckpt_dir = Path(args.ckpt_dir)
    if use_wandb:
        ckpt_dir = ckpt_dir / wandb.run.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── 초기 관측 ─────────────────────────────────────────────────────────────
    obs, _    = env.reset()
    obs_img   = obs["policy"]           # (E, K, H, W)
    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)
    finished  = dict(returns=[], lengths=[], coverages=[], terminated=[])

    loss_list, q_list             = [], []
    rew_cov_list, rew_pen_list, rew_succ_list = [], [], []

    # ══════════════════════════════════════════════════════════════════════════
    # 학습 루프
    # ══════════════════════════════════════════════════════════════════════════
    while global_step < args.total_steps:

        # ── Epsilon-greedy 행동 선택 ──────────────────────────────────────────
        with torch.no_grad():
            if np.random.random() < epsilon:
                action_idx = torch.randint(0, 6, (E,), device=device)
            else:
                action_idx = q_net(obs_img).argmax(dim=1)

        env_action = make_env_action(action_idx, E, device)
        next_obs, reward, terminated, truncated, _ = env.step(env_action)
        next_obs_img = next_obs["policy"]

        rew_cov_list .append(env._last_rew_coverage  .mean().item())
        rew_pen_list .append(env._last_rew_penalty   .mean().item())
        rew_succ_list.append(env._last_success_reward.mean().item())

        done_any = terminated | truncated

        # ── Replay buffer push (env 별) ───────────────────────────────────────
        for eid in range(E):
            replay.push(
                obs_img[eid],
                action_idx[eid].item(),
                reward[eid].item(),
                next_obs_img[eid],
                done_any[eid].item(),
            )

        ep_return += reward
        ep_len    += 1

        for eid in done_any.nonzero(as_tuple=True)[0].tolist():
            finished["returns"]   .append(ep_return[eid].item())
            finished["lengths"]   .append(ep_len[eid].item())
            finished["coverages"] .append(env.curr_coverage[eid].item())
            finished["terminated"].append(float(terminated[eid].item()))
            ep_return[eid] = 0.0
            ep_len[eid]    = 0
            # Epsilon decay: paper에서 에피소드마다 decay
            epsilon = max(dqn_cfg.eps_end, epsilon * dqn_cfg.eps_decay)

        obs_img     = next_obs_img
        global_step += E

        # ── DQN Update ────────────────────────────────────────────────────────
        if len(replay) >= dqn_cfg.min_replay:
            stats = dqn_update(q_net, target_net, optimizer, replay, dqn_cfg)
            loss_list.append(stats["loss"])
            q_list   .append(stats["q_mean"])

            if global_step % dqn_cfg.target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

        # ── 로그 ──────────────────────────────────────────────────────────────
        LOG_EVERY = 10_000
        if global_step - last_log_step >= LOG_EVERY:
            last_log_step = global_step
            fps = int(global_step / (time.time() - t0 + 1e-8))

            log: dict = {
                "train/epsilon":        epsilon,
                "train/fps":            fps,
                "train/global_step":    global_step,
                "train/replay_size":    len(replay),
                "reward/coverage":      np.mean(rew_cov_list)  if rew_cov_list  else 0.0,
                "reward/penalty":       np.mean(rew_pen_list)  if rew_pen_list  else 0.0,
                "reward/success":       np.mean(rew_succ_list) if rew_succ_list else 0.0,
                # "train/coverage_mean":  env.curr_coverage.mean().item(),
            }

            if loss_list:
                log["train/loss"]   = np.mean(loss_list)
                log["train/q_mean"] = np.mean(q_list)

            if finished["returns"]:
                log["episode/mean_return"]   = np.mean(finished["returns"])
                log["episode/mean_length"]   = np.mean(finished["lengths"])
                log["episode/mean_coverage"] = np.mean(finished["coverages"])
                log["episode/success_rate"]  = np.mean(finished["terminated"])
                log["episode/timeout_rate"]  = 1.0 - np.mean(finished["terminated"])

            if use_wandb:
                wandb.log(log, step=global_step)

            loss_str = f"  loss={log['train/loss']:.4f}" if "train/loss" in log else "  loss=N/A"
            cov_str  = f"  cov={log['episode/mean_coverage']:.3f}" if "episode/mean_coverage" in log else ""

            print(
                f"[{global_step:9d}]"
                f"  eps={epsilon:.4f}"
                f"{loss_str}"
                f"  q={log.get('train/q_mean', 0):+.3f}"
                f"{cov_str}"
                f"  replay={len(replay)}"
                f"  fps={fps}",
                flush=True,
            )
            print(
                f"  [reward]"
                f"  cov={np.mean(rew_cov_list):+.4f}"
                f"  penalty={np.mean(rew_pen_list):.4f}"
                f"  success={np.mean(rew_succ_list):+.4f}",
                flush=True,
            )

            rew_cov_list.clear(); rew_pen_list.clear(); rew_succ_list.clear()
            loss_list.clear(); q_list.clear()
            finished = dict(returns=[], lengths=[], coverages=[], terminated=[])

        # ── Eval ──────────────────────────────────────────────────────────────
        if args.eval_interval > 0 and global_step - last_eval_step >= args.eval_interval:
            last_eval_step = global_step
            eval_metrics = run_eval(env, q_net, device, args.eval_episodes)
            if use_wandb:
                wandb.log(eval_metrics, step=global_step)
            print(
                f"  [EVAL]"
                f"  return={eval_metrics['eval/mean_return']:.2f}"
                f"  cov={eval_metrics['eval/mean_coverage']:.3f}"
                f"  success={eval_metrics['eval/success_rate']:.2f}"
                f"  len={eval_metrics['eval/mean_length']:.1f}",
                flush=True,
            )
            obs, _    = env.reset()
            obs_img   = obs["policy"]
            ep_return = torch.zeros(E, device=device)
            ep_len    = torch.zeros(E, device=device, dtype=torch.long)

        # ── 체크포인트 ────────────────────────────────────────────────────────
        if global_step % args.save_interval == 0 and global_step > 0:
            ckpt_path = ckpt_dir / f"scanRL_step_{global_step:010d}.pt"
            torch.save({
                "global_step": global_step,
                "q_net":       q_net.state_dict(),
                "target_net":  target_net.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "epsilon":     epsilon,
                "args":        vars(args),
            }, ckpt_path)
            print(f"[ckpt] → {ckpt_path}", flush=True)

    env.close()
    if use_wandb:
        wandb.finish()
    simulation_app.close()


if __name__ == "__main__":
    main()
