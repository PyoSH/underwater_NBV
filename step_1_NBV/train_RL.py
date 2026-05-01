from __future__ import annotations                                                  
import argparse, os, sys, time
from pathlib import Path

from isaaclab.app import AppLauncher                                                

# ── CLI (AppLauncher 파싱 전 등록) ────────────────────────────────────────────    
parser = argparse.ArgumentParser(description="OceanNBV PPO")
parser.add_argument("--num_envs",       type=int,   default=4)                      
parser.add_argument("--total_steps",    type=int,   default=10_000_000)             
parser.add_argument("--rollout_steps",  type=int,   default=256)                    
parser.add_argument("--ppo_epochs",     type=int,   default=4)                      
parser.add_argument("--minibatch_size", type=int,   default=512)                    
parser.add_argument("--lr",             type=float, default=3e-4)                   
parser.add_argument("--gamma",          type=float, default=0.99)                   
parser.add_argument("--gae_lambda",     type=float, default=0.95)
parser.add_argument("--clip_eps",       type=float, default=0.2)                    
parser.add_argument("--ent_coef",       type=float, default=0.01)
parser.add_argument("--vf_coef",        type=float, default=0.5)                    
parser.add_argument("--max_grad_norm",  type=float, default=0.5)
parser.add_argument("--lr_decay",       action="store_true")                        
parser.add_argument("--ckpt_dir",       type=str,   default="./checkpoints")
parser.add_argument("--save_interval",  type=int,   default=200)                    
parser.add_argument("--resume",         type=str,   default=None)
parser.add_argument("--wandb_project",  type=str,   default=None)                   
parser.add_argument("--wandb_name",     type=str,   default=None)
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
                                                                                    
sys.path.insert(0, os.path.dirname(__file__))
from envCfg   import OceanEnvCfg                                                    
from env      import OceanEnv                                                       
from algorithm import (Actor, Critic, RolloutBuffer, PPOConfig,
                        make_env_action, explained_variance, ppo_update)            
                                                                                    

def main():                                                                         
    # ── 환경 ─────────────────────────────────────────────────────────────────
    env_cfg = OceanEnvCfg()
    env_cfg.scene.num_envs = args.num_envs                                          
    env_cfg.sim.dt         = 1.0 / 30.0
                                                                                    
    env    = OceanEnv(cfg=env_cfg, render_mode="rgb_array")                         
    device = env.device                                                             
    E      = env.num_envs                                                           
    H, W   = env_cfg.visual.h, env_cfg.visual.w                                     
    K_img  = env_cfg.visual.num_seq_actor
    K_dep  = env_cfg.visual.num_seq_critic                                          
    T      = args.rollout_steps
                                                                                    
    # ── 네트워크 & 옵티마이저 ─────────────────────────────────────────────────    
    actor     = Actor (img_ch=K_img, scalar_dim=5).to(device)                       
    critic    = Critic(depth_ch=K_dep).to(device)                                   
    optimizer = torch.optim.Adam(                                                   
        list(actor.parameters()) + list(critic.parameters()),
        lr=args.lr, eps=1e-5,                                                       
    )           
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
                                                                                    
    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume:                                                                 
        ckpt = torch.load(args.resume, map_location=device)
        actor    .load_state_dict(ckpt["actor"])                                    
        critic   .load_state_dict(ckpt["critic"])
        optimizer.load_state_dict(ckpt["optimizer"])                                
        global_step = ckpt.get("global_step", 0)
        rollout_idx = ckpt.get("rollout_idx",  0)                                   
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
    buf = RolloutBuffer(T, E, K_img, K_dep, H, W, scalar_dim=5, device=device)
                                                                                    
    obs, _     = env.reset()
    obs_img    = obs["policy"]                                                      
    obs_scalar = obs["extra_info"]                                                  
    obs_depth  = obs["critic"]
                                                                                    
    ep_return = torch.zeros(E, device=device)
    ep_len    = torch.zeros(E, device=device, dtype=torch.long)                     
    finished  = dict(returns=[], lengths=[], coverages=[])                          
                                                                                    
    ckpt_dir = Path(args.ckpt_dir)                                                  
    ckpt_dir.mkdir(parents=True, exist_ok=True)                                     
    t0 = time.time()
                                                                                    
    # ══════════════════════════════════════════════════════════════════════════
    # 학습 루프                                                                     
    # ══════════════════════════════════════════════════════════════════════════
    while global_step < args.total_steps:                                           

        if args.lr_decay:                                                           
            frac = max(1.0 - global_step / args.total_steps, 0.0)
            for pg in optimizer.param_groups:                                       
                pg["lr"] = args.lr * frac                                           

        buf.ptr = 0                                                                 
        actor.eval(); critic.eval()
                                                                                    
        # ── Rollout 수집 ──────────────────────────────────────────────────────    
        for _ in range(T):                                                          
            with torch.no_grad():                                                   
                pose_act, light_act, logprob, _ = actor.sample(obs_img, obs_scalar)
                value = critic(obs_depth)                                           

            env_action = make_env_action(pose_act, light_act, E, device)            
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = (terminated | truncated).float()                                 
                
            buf.add(obs_img, obs_scalar, obs_depth,                                 
                    pose_act, light_act, logprob, reward, done, value)
                                                                                    
            ep_return += reward
            ep_len    += 1                                                          
                
            for eid in done.nonzero(as_tuple=True)[0].tolist():                     
                finished["returns"]  .append(ep_return[eid].item())
                finished["lengths"]  .append(ep_len[eid].item())                    
                finished["coverages"].append(env.curr_coverage[eid].item())         
                ep_return[eid] = 0.0                                                
                ep_len[eid]    = 0                                                  
                                                                                    
            obs_img    = next_obs["policy"]                                         
            obs_scalar = next_obs["extra_info"]
            obs_depth  = next_obs["critic"]                                         
            global_step += E

        # ── GAE & PPO ─────────────────────────────────────────────────────────    
        with torch.no_grad():
            last_val = critic(obs_depth)                                            
        buf.compute_gae(last_val, args.gamma, args.gae_lambda)                      

        actor.train(); critic.train()                                               
        stats = ppo_update(actor, critic, optimizer, buf, ppo_cfg)
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
            "train/learning_rate":      optimizer.param_groups[0]["lr"],            
            "train/fps":                fps,
            "train/global_step":        global_step,                                
        }                                                                           

        if finished["returns"]:                                                     
            log["episode/mean_return"]   = np.mean(finished["returns"])
            log["episode/mean_length"]   = np.mean(finished["lengths"])             
            log["episode/mean_coverage"] = np.mean(finished["coverages"])
            finished = dict(returns=[], lengths=[], coverages=[])                   
                
        if use_wandb:                                                               
            wandb.log(log, step=global_step)
                                                                                    
        if rollout_idx % 10 == 0:
            cov = f"  cov={log['episode/mean_coverage']:.3f}" if "episode/mean_coverage" in log else ""                                              
            print(
                f"[{global_step:9d}]"                                               
                f"  rew={log['train/mean_step_reward']:+.3f}"
                f"  pl={stats['policy_loss']:.4f}"                                  
                f"  vl={stats['value_loss']:.4f}"                                   
                f"  ent={stats['entropy']:.3f}"                                     
                f"  ev={ev:.3f}"                                                    
                f"{cov}  fps={fps}",
                flush=True,                                                         
            )   
                                                                                    
        # ── 체크포인트 ────────────────────────────────────────────────────────
        if rollout_idx % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"step_{global_step:010d}.pt"                    
            torch.save({
                "global_step": global_step,                                         
                "rollout_idx": rollout_idx,                                         
                "actor":       actor.state_dict(),
                "critic":      critic.state_dict(),                                 
                "optimizer":   optimizer.state_dict(),
                "args":        vars(args),                                          
            }, ckpt_path)
            if use_wandb:                                                           
                wandb.save(str(ckpt_path))
            print(f"[ckpt] → {ckpt_path}", flush=True)                              

    env.close()                                                                     
    simulation_app.close()
    if use_wandb:
        wandb.finish()                                                              

                                                                                    
if __name__ == "__main__":
    main()