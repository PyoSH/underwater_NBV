from __future__ import annotations                                                  
import argparse, os, sys, time
from pathlib import Path
import math

from isaaclab.app import AppLauncher                                                

# ── CLI (AppLauncher 파싱 전 등록) ────────────────────────────────────────────    
parser = argparse.ArgumentParser(description="OceanNBV PPO")
parser.add_argument("--num_envs",       type=int,   default=1)                      
parser.add_argument("--total_steps",    type=int,   default=500_000)             
parser.add_argument("--rollout_steps",  type=int,   default=256)                    
parser.add_argument("--ppo_epochs",     type=int,   default=10)                      
parser.add_argument("--minibatch_size", type=int,   default=64)                    
parser.add_argument("--lr",             type=float, default=1e-4)                   
parser.add_argument("--gamma",          type=float, default=0.99)                   
parser.add_argument("--gae_lambda",     type=float, default=0.95)
parser.add_argument("--clip_eps",       type=float, default=0.1)                    
parser.add_argument("--ent_coef",       type=float, default=0.05)
parser.add_argument("--vf_coef",        type=float, default=0.5)                    
parser.add_argument("--max_grad_norm",  type=float, default=0.5)
parser.add_argument("--lr_decay",       action="store_true")                        
parser.add_argument("--ckpt_dir",       type=str,   default="/workspace/checkpoints")
parser.add_argument("--save_interval",  type=int,   default=200)                    
parser.add_argument("--resume",         type=str,   default=None)
parser.add_argument("--wandb_project",  type=str,   default="RL_NBV")
parser.add_argument("--wandb_name",     type=str,   default=None)
parser.add_argument("--eval_interval",  type=int,   default=10,
                    help="run eval every N rollouts (0 = disabled)")
parser.add_argument("--eval_episodes",  type=int,   default=20)
parser.add_argument("--use_visit_map",  action="store_true")
AppLauncher.add_app_launcher_args(parser)                                           
                
if "--enable_cameras" not in sys.argv:                                              
    sys.argv.append("--enable_cameras")
                                                                                    
args = parser.parse_args()                                                          
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app                                                   
# ── AppLauncher 이후 import ───────────────────────────────────────────────────    
import numpy as np
import torch                                                                        
import wandb    
                                                                                    
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.envCfg    import OceanEnvCfg
from env.env       import OceanEnv
# from algorithm.algorithm2 import (Actor, Critic, RolloutBuffer, PPOConfig,
#                                   make_env_action, explained_variance, ppo_update)
from algorithm.algorithm3 import (Actor, Critic, RolloutBuffer, PPOConfig, make_env_action, explained_variance, ppo_update)                  

def run_eval(env, actor, device, n_episodes: int) -> dict:
    actor.eval()
    results = dict(returns=[], lengths=[], coverages=[], successes=[])

    obs, _ = env.reset()
    obs_img    = obs["policy"]
    obs_scalar = obs["extra_info"]

    E = env.num_envs
    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)
    completed = [0] * E
    max_steps = env.max_episode_length * n_episodes * 2

    with torch.no_grad():
        for _ in range(max_steps):
            if min(completed) >= n_episodes:
                break
            pose_act = actor.greedy(obs_img, obs_scalar)
            env_action = make_env_action(pose_act, E, device)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)

            ep_return += reward
            ep_len    += 1

            for eid in (terminated | truncated).nonzero(as_tuple=True)[0].tolist():
                if completed[eid] < n_episodes:
                    results["returns"]  .append(ep_return[eid].item())
                    results["lengths"]  .append(ep_len[eid].item())
                    results["coverages"].append(env._terminal_coverage[eid].item())
                    results["successes"].append(float(terminated[eid].item()))
                    completed[eid] += 1
                ep_return[eid] = 0.0
                ep_len[eid]    = 0

            obs_img    = next_obs["policy"]
            obs_scalar = next_obs["extra_info"]

    return {
        "eval/mean_return":   np.mean(results["returns"])   if results["returns"]   else 0.0,
        "eval/mean_length":   np.mean(results["lengths"])   if results["lengths"]   else 0.0,
        "eval/mean_coverage": np.mean(results["coverages"]) if results["coverages"] else 0.0,
        "eval/success_rate":  np.mean(results["successes"]) if results["successes"] else 0.0,
    }


def main():                                                                         
    # ── 환경 ─────────────────────────────────────────────────────────────────
    env_cfg = OceanEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.coverage_terminal = 0.82
    env_cfg.c_step = 0.03

    if args.use_visit_map:
        env_cfg.visual.num_seq_actor = 2
        env_cfg.visual.num_seq_critic = 2

        env_cfg.use_visit_map = True
        env_cfg.k_explore     = 0.1
        env_cfg.observation_space = (
            env_cfg.visual.num_seq_actor + 1,
            env_cfg.visual.h, env_cfg.visual.w
        )
        env_cfg.state_space = (
            env_cfg.visual.num_seq_critic + 1,
            env_cfg.visual.h, env_cfg.visual.w
        )
    # env_cfg.sim.dt         = 1.0 / 30.0
    # elevation 3 단계로 제한
    # env_cfg.phi_min   = math.radians(20)
    # env_cfg.phi_max   = math.radians(70)
    # env_cfg.delta_phi = math.radians(25)

    # 거리 단계도 3단계로 제한 (구현 필요)
                                                                                    
    env    = OceanEnv(cfg=env_cfg, render_mode="rgb_array")                         
    device = env.device                                                             
    E      = env.num_envs                                                           
    H, W   = env_cfg.visual.h, env_cfg.visual.w                                     
    K_img  = env_cfg.visual.num_seq_actor  + (1 if args.use_visit_map else 0)
    K_dep  = env_cfg.visual.num_seq_critic + (1 if args.use_visit_map else 0)
    s_dim_critic = 4 if args.use_visit_map else 3
    T      = args.rollout_steps
                                                                                    
    # ── 네트워크 & 옵티마이저 ─────────────────────────────────────────────────    
    actor     = Actor (img_ch=K_img, scalar_dim=3).to(device)                       
    critic    = Critic(depth_ch=K_dep, scalar_dim=s_dim_critic).to(device)                                   
    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr = args.lr * 2)
    ppo_cfg = PPOConfig(                                                            
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,                                         
        clip_eps=args.clip_eps,
        ent_coef=args.ent_coef,                                                     
        vf_coef=args.vf_coef,                                                       
        max_grad_norm=args.max_grad_norm,
    )                                                                               
                
    global_step = 0
    rollout_idx = 0
    last_log_step = 0
                                                                                    
    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume:                                                                 
        ckpt = torch.load(args.resume, map_location=device)
        actor    .load_state_dict(ckpt["actor"])                                    
        critic   .load_state_dict(ckpt["critic"])
        optimizer_actor.load_state_dict(ckpt["optimizer_actor"])
        optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
        global_step = ckpt.get("global_step", 0)
        rollout_idx = ckpt.get("rollout_idx",  0)                                   
        last_log_step = global_step
        print(f"[resume] {args.resume}  (step={global_step})", flush=True)          
                                                                                    
    # ── wandb ─────────────────────────────────────────────────────────────────    
    use_wandb = args.wandb_project is not None                                      
    if use_wandb:                                                                   
        wandb.init(
            project=args.wandb_project,                                             
            name=args.wandb_name,
            config=vars(args),
            resume="allow",                                                         
        )
        wandb.watch(actor,  log="gradients", log_freq=200)                          
        wandb.watch(critic, log="gradients", log_freq=200)                          

    # ── 버퍼 & 에피소드 트래커 ───────────────────────────────────────────────     
    buf = RolloutBuffer(T, E, K_img, K_dep, H, W, scalar_dim=3, scalar_dim_critic = s_dim_critic, device=device)
    obs, _     = env.reset()
    obs_img    = obs["policy"]
    obs_scalar = obs["extra_info"]
    obs_depth  = obs["critic"]
    obs_scalar_c = obs["critic_scalar"]

    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)           
    finished  = dict(returns=[], lengths=[], coverages=[], terminated=[])

    ckpt_dir = Path(args.ckpt_dir)
    if use_wandb:
        ckpt_dir = ckpt_dir / wandb.run.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)                                     
    t0 = time.time()
                                                                                    
    # ══════════════════════════════════════════════════════════════════════════
    # 학습 루프                                                                     
    # ══════════════════════════════════════════════════════════════════════════
    while global_step < args.total_steps:                                           

        if args.lr_decay:                                                           
            frac = max(1.0 - global_step / args.total_steps, 0.0)
            
            for pg in optimizer_actor.param_groups:
                pg["lr"] = args.lr * frac
            for pg in optimizer_critic.param_groups:
                pg["lr"] = args.lr * 2 *frac

        buf.reset()
        actor.eval(); critic.eval()

        rew_cov_list, rew_pen_list, rew_succ_list, rew_exp_list, rew_stall_list = [], [], [], [], []
                                                                                    
        # ── Rollout 수집 ──────────────────────────────────────────────────────    
        for _ in range(T):                                                          
            with torch.no_grad():
                pose_act, logprob, _ = actor.sample(obs_img, obs_scalar)
                value = critic(obs_depth, obs_scalar_c)

            env_action = make_env_action(pose_act, E, device)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            rew_cov_list .append(env._last_rew_coverage  .mean().item())
            rew_pen_list .append(env._last_rew_penalty   .mean().item())
            rew_succ_list.append(env._last_success_reward.mean().item())
            rew_exp_list  .append(env._last_rew_explore   .mean().item())
            rew_stall_list.append(env._last_rew_stall     .mean().item())
            # done = (terminated | truncated).float()
            terminated_f    = terminated.float()
            done_any        = (terminated | truncated)
                
            buf.add(obs_img, obs_scalar, obs_scalar_c, obs_depth,
                    pose_act, logprob, reward, terminated_f, value)
                                                                                    
            ep_return += reward
            ep_len    += 1                                                          
                
            for eid in done_any.nonzero(as_tuple=True)[0].tolist():
                finished["returns"]   .append(ep_return[eid].item())
                finished["lengths"]   .append(ep_len[eid].item())
                finished["coverages"] .append(env._terminal_coverage[eid].item())
                finished["terminated"].append(float(terminated[eid].item()))
                ep_return[eid] = 0.0
                ep_len[eid]    = 0                                                  
                                                                                    
            obs_img    = next_obs["policy"]                                         
            obs_scalar = next_obs["extra_info"]
            obs_depth  = next_obs["critic"]  
            obs_scalar_c = next_obs["critic_scalar"]                                       
            global_step += E

        # ── GAE & PPO ─────────────────────────────────────────────────────────    
        with torch.no_grad():
            last_val = critic(obs_depth, obs_scalar_c)
        buf.compute_gae(last_val, args.gamma, args.gae_lambda)                      

        actor.train(); critic.train()                                
        stats = ppo_update(actor, critic, optimizer_actor, optimizer_critic, buf, ppo_cfg)
        rollout_idx += 1                                                            
                
        # ── 로그 ─────────────────────────────────────────────────────────────     
        ev  = explained_variance(buf.values.reshape(-1), buf.returns.reshape(-1))
        fps = int(global_step / (time.time() - t0 + 1e-8))                          
                                                                                    
        log = {
            "train/mean_step_reward":   buf.rewards.mean().item(),
            "train/policy_loss":        stats["policy_loss"],
            "train/value_loss":         stats["value_loss"],
            "train/entropy":            stats["entropy"],
            "train/approx_kl":          stats["approx_kl"],
            "train/explained_variance": ev,
            "train/early_stop":         float(stats["early_stop"]),
            "train/lr_actor":           optimizer_actor.param_groups[0]["lr"],
            "train/lr_critic":          optimizer_critic.param_groups[0]["lr"],
            "train/fps":                fps,
            "train/global_step":        global_step,
            "reward/coverage":          np.mean(rew_cov_list),
            "reward/penalty":           np.mean(rew_pen_list),
            "reward/success":           np.mean(rew_succ_list),
            "reward/explore":           np.mean(rew_exp_list),
            "reward/stall":             np.mean(rew_stall_list),
            # "train/coverage_mean":      env.curr_coverage.mean().item(),
            "env0/theta_deg":           math.degrees(env._sph_theta[0].item()),
            "env0/phi_deg":             math.degrees(env._sph_phi[0].item()),
            "env0/psi":                 env._sph_psi[0].item(),
            "env0/coverage":            env.curr_coverage[0].item(),
            "env0/surf_voxels":         env._total_surf_voxels[0].item(),
            "env0/weight_filled":       (env._weight_vol[0] > 0).float().mean().item(),
        }                                                                           

        if finished["returns"]:
            log["episode/mean_return"]   = np.mean(finished["returns"])
            log["episode/mean_length"]   = np.mean(finished["lengths"])
            log["episode/mean_coverage"] = np.mean(finished["coverages"])
            log["episode/success_rate"]  = np.mean(finished["terminated"])
            log["episode/timeout_rate"]  = 1.0 - np.mean(finished["terminated"])
            finished = dict(returns=[], lengths=[], coverages=[], terminated=[])                   
                
        if use_wandb:                                                               
            wandb.log(log, step=global_step)
                                  
        # if rollout_idx % 10 == 0:
        LOG_EVERY = 10_000
        if global_step - last_log_step >= LOG_EVERY:
            last_log_step = global_step
            cov = f"  cov={log['episode/mean_coverage']:.3f}" if "episode/mean_coverage" in log else ""
                                                                                            
            # ── 보상 구성요소 분해 ──────────────────────────────────────────────          
            # rew_mean        = buf.rewards.mean().item()
            # penalty_mean    = env.cfg.c_step + env.cfg.k_x * env.cfg.lambda_q
            # positive_mean   = rew_mean + penalty_mean
                                                                                            
            # ── TSDF 상태 ────────────────────────────────────────────────────────
            weight_filled = (env._weight_vol > 0).float().mean().item()       # 관측된 voxel 비율                                                                               
            surf_total    = env._total_surf_voxels.mean().item()               # GT 표면 voxel 수                                                                            
            curr_cov_now  = env.curr_coverage.mean().item()                    # 현재 버리지 (에피소드 끝 아니어도)                                                     
                        
            print(                                                                          
                f"[{global_step:9d}]"
                f"  rew={log['train/mean_step_reward']:+.6f}"
                f"  pl={stats['policy_loss']:.4f}"                                          
                f"  vl={stats['value_loss']:.4f}"
                f"  ent={stats['entropy']:.3f}"                                             
                f"  ev={ev:.3f}"
                f"{cov}  fps={fps}",                                                        
                flush=True,
            )                                                                               
            print(
                f"  [reward]"
                f"  cov={np.mean(rew_cov_list):+.4f}"
                f"  penalty={np.mean(rew_pen_list):.4f}"
                f"  success={np.mean(rew_succ_list):+.4f}"
                f"  explore={np.mean(rew_exp_list):+.4f}"
                f"  stall={np.mean(rew_stall_list):.4f}"
                f"  net={buf.rewards.mean().item():+.4f}",
                flush=True,
            )               
            print(
                f"  [tsdf]"
                f"  curr_cov={curr_cov_now:.4f}"                                            
                f"  weight_filled={weight_filled:.6f}"
                f"  surf_voxels={surf_total:.0f}",                                          
                flush=True,                                                                 
            )

                                                                                    
        # ── Eval ─────────────────────────────────────────────────────────────
        if args.eval_interval > 0 and rollout_idx % args.eval_interval == 0:
            eval_metrics = run_eval(env, actor, device, args.eval_episodes)
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
            obs, _ = env.reset()
            obs_img    = obs["policy"]
            obs_scalar = obs["extra_info"] 
            obs_depth  = obs["critic"]
            obs_scalar_c = obs["critic_scalar"]
            ep_return  = torch.zeros(E, device=device)
            ep_len     = torch.zeros(E, device=device, dtype=torch.long)

        # ── 체크포인트 ────────────────────────────────────────────────────────
        if rollout_idx % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"rl_NBV_step_{global_step:010d}.pt"                    
            torch.save({
                "global_step": global_step,                                         
                "rollout_idx": rollout_idx,                                         
                "actor":       actor.state_dict(),
                "critic":      critic.state_dict(),                                 
                "optimizer_actor":   optimizer_actor.state_dict(),
                "optimizer_critic":   optimizer_critic.state_dict(),
                "args":        vars(args),                                          
            }, ckpt_path)
            print(f"[ckpt] → {ckpt_path}", flush=True)                              

    env.close()
    
    if use_wandb:
        wandb.finish()

    simulation_app.close()

                                                                                    
if __name__ == "__main__":
    main()