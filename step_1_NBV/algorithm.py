from __future__ import annotations
from dataclasses import dataclass                                                   
                
import torch                                                                        
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
                                                                                    
# ══════════════════════════════════════════════════════════════════════════════
# PPO 하이퍼파라미터                                                                
# ══════════════════════════════════════════════════════════════════════════════
                                                                                    
@dataclass
class PPOConfig:                                                                    
    ppo_epochs:     int   = 4
    minibatch_size: int   = 512
    clip_eps:       float = 0.2                                                     
    ent_coef:       float = 0.01
    vf_coef:        float = 0.5                                                     
    max_grad_norm:  float = 0.5
                                                                                    

# ══════════════════════════════════════════════════════════════════════════════    
# Network       
# ══════════════════════════════════════════════════════════════════════════════
                                                                                    
# 84×84: Conv(8,s4)→20, Conv(4,s2)→9, Conv(3,s1)→7  →  64*7*7 = 3136                
_CNN_OUT = 64 * 7 * 7                                                               
                                                                                    
                
def _build_cnn(in_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, kernel_size=8, stride=4), nn.ReLU(),                   
        nn.Conv2d(32,    64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(64,    64, kernel_size=3, stride=1), nn.ReLU(),                   
        nn.Flatten(),                                                               
    )                                                                               
                                                                                    
                
class Actor(nn.Module):
    """
    이미지(K,H,W) + scalar(5) → pose(6) / light(3) 독립 Categorical.
    _apply_action() 이 pose·light 각각 argmax 를 취하는 구조에 대응.                
    """                                                                             
    def __init__(self, img_ch: int = 6, scalar_dim: int = 5,                        
                n_pose: int = 6, n_light: int = 3):                                
        super().__init__()                                                          
        self.cnn  = _build_cnn(img_ch)                                              
        self.mlp  = nn.Sequential(                                                  
            nn.Linear(_CNN_OUT + scalar_dim, 512), nn.ReLU(),                       
            nn.Linear(512, 256),                   nn.ReLU(),
        )                                                                           
        self.pose_head  = nn.Linear(256, n_pose)
        self.light_head = nn.Linear(256, n_light)                                   
                
    def _dists(self, img: torch.Tensor, scalar: torch.Tensor):                      
        feat = self.cnn(img)
        feat = self.mlp(torch.cat([feat, scalar], dim=-1))                          
        return (Categorical(logits=self.pose_head(feat)),
                Categorical(logits=self.light_head(feat)))                          

    def sample(self, img: torch.Tensor, scalar: torch.Tensor):                      
        """(pose_act, light_act, joint_logprob, joint_entropy)"""
        pd, ld = self._dists(img, scalar)                                           
        pa, la = pd.sample(), ld.sample()                                           
        return pa, la, pd.log_prob(pa) + ld.log_prob(la), pd.entropy() + ld.entropy()                                                                        
                
    def evaluate(self, img: torch.Tensor, scalar: torch.Tensor,                     
                pose_act: torch.Tensor, light_act: torch.Tensor):
        """(joint_logprob, joint_entropy)"""                                        
        pd, ld = self._dists(img, scalar)
        return (pd.log_prob(pose_act) + ld.log_prob(light_act),                     
                pd.entropy()          + ld.entropy())
                                                                                    
                
class Critic(nn.Module):                                                            
    """GT depth 시퀀스 전용 privileged critic."""
    def __init__(self, depth_ch: int = 6):                                          
        super().__init__()
        self.cnn = _build_cnn(depth_ch)                                             
        self.mlp = nn.Sequential(                                                   
            nn.Linear(_CNN_OUT, 512), nn.ReLU(),
            nn.Linear(512,      256), nn.ReLU(),                                    
            nn.Linear(256,        1),                                               
        )
                                                                                    
    def forward(self, depth: torch.Tensor) -> torch.Tensor:                         
        return self.mlp(self.cnn(depth)).squeeze(-1)  # (B,)
                                                                                    
                
# ══════════════════════════════════════════════════════════════════════════════    
# Rollout Buffer
# ══════════════════════════════════════════════════════════════════════════════
                                                                                    
class RolloutBuffer:
    def __init__(self, T: int, E: int, K_img: int, K_dep: int,                      
                H: int, W: int, scalar_dim: int, device):                          
        self.T, self.E, self.ptr = T, E, 0                                          
        kw = dict(device=device)                                                    
        self.obs_img    = torch.zeros(T, E, K_img, H, W, **kw)                      
        self.obs_scalar = torch.zeros(T, E, scalar_dim,  **kw)                      
        self.obs_depth  = torch.zeros(T, E, K_dep, H, W, **kw)                      
        self.pose_acts  = torch.zeros(T, E, dtype=torch.long, **kw)                 
        self.light_acts = torch.zeros(T, E, dtype=torch.long, **kw)                 
        self.logprobs   = torch.zeros(T, E, **kw)                                   
        self.rewards    = torch.zeros(T, E, **kw)                                   
        self.dones      = torch.zeros(T, E, **kw)                                   
        self.values     = torch.zeros(T, E, **kw)
        self.returns:    torch.Tensor                                               
        self.advantages: torch.Tensor                                               

    def add(self, obs_img, obs_scalar, obs_depth,                                   
            pose_act, light_act, logprob, reward, done, value):
        t = self.ptr                                                                
        self.obs_img[t]    = obs_img
        self.obs_scalar[t] = obs_scalar                                             
        self.obs_depth[t]  = obs_depth
        self.pose_acts[t]  = pose_act                                               
        self.light_acts[t] = light_act                                              
        self.logprobs[t]   = logprob
        self.rewards[t]    = reward                                                 
        self.dones[t]      = done
        self.values[t]     = value                                                  
        self.ptr += 1
                                                                                    
    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):      
        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.E, device=self.rewards.device)                       
        for t in reversed(range(self.T)):                                           
            nv    = last_value if t == self.T - 1 else self.values[t + 1]
            mask  = 1.0 - self.dones[t]                                             
            delta = self.rewards[t] + gamma * nv * mask - self.values[t]            
            gae   = delta + gamma * lam * mask * gae                                
            adv[t] = gae                                                            
        self.returns    = adv + self.values
        self.advantages = adv                                                       
                                                                                    
    def flat(self) -> dict[str, torch.Tensor]:
        TE = self.T * self.E                                                        
        return {                                                                    
            "obs_img":    self.obs_img   .reshape(TE, *self.obs_img.shape[2:]),
            "obs_scalar": self.obs_scalar.reshape(TE, *self.obs_scalar.shape[2:]),  
            "obs_depth":  self.obs_depth .reshape(TE, *self.obs_depth.shape[2:]),
            "pose_acts":  self.pose_acts .reshape(TE),                              
            "light_acts": self.light_acts.reshape(TE),
            "logprobs":   self.logprobs  .reshape(TE),                              
            "returns":    self.returns   .reshape(TE),                              
            "advantages": self.advantages.reshape(TE),
            "old_values": self.values    .reshape(TE),                              
        }                                                                           

                                                                                    
# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════    

def make_env_action(pose_idx: torch.Tensor, light_idx: torch.Tensor,                
                    E: int, device) -> torch.Tensor:
    """                                                                             
    (E,) index → (E, 9) one-hot.
    슬롯 0-5: pose one-hot,  슬롯 6-8: light one-hot.                               
    """                                                                             
    act = torch.zeros(E, 9, device=device)                                          
    act.scatter_(1, pose_idx.unsqueeze(1),        1.0)                              
    act.scatter_(1, (light_idx + 6).unsqueeze(1), 1.0)
    return act                                                                      
                
                                                                                    
def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    var_ret = returns.var()
    if var_ret < 1e-8:
        return float("nan")                                                         
    return (1.0 - (returns - values).var() / var_ret).item()
                                                                                    
                
# ══════════════════════════════════════════════════════════════════════════════    
# PPO 업데이트  
# ══════════════════════════════════════════════════════════════════════════════    

def ppo_update(actor: Actor, critic: Critic,                                        
                optimizer: torch.optim.Optimizer,
                buf: RolloutBuffer,                                                  
                cfg: PPOConfig) -> dict:
    data = buf.flat()                                                               
    adv  = data["advantages"]
    adv  = (adv - adv.mean()) / (adv.std() + 1e-8)                                  

    TE  = adv.shape[0]                                                              
    acc = dict(policy_loss=0., value_loss=0., entropy=0., approx_kl=0., n=0)
                                                                                    
    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(TE, device=adv.device)                                
        for s in range(0, TE, cfg.minibatch_size):                                  
            mb = perm[s : s + cfg.minibatch_size]
            if mb.numel() == 0:                                                     
                continue
                                                                                    
            new_logp, entropy = actor.evaluate(
                data["obs_img"][mb], data["obs_scalar"][mb],                        
                data["pose_acts"][mb], data["light_acts"][mb],                      
            )
            ratio  = (new_logp - data["logprobs"][mb]).exp()                        
            mb_adv = adv[mb]                                                        

            pg = torch.max(                                                         
                -mb_adv * ratio,
                -mb_adv * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),          
            ).mean()                                                                

            v  = critic(data["obs_depth"][mb])                                      
            vl = F.mse_loss(v, data["returns"][mb])
                                                                                    
            loss = pg + cfg.vf_coef * vl - cfg.ent_coef * entropy.mean()            
            optimizer.zero_grad()                                                   
            loss.backward()                                                         
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                cfg.max_grad_norm,                                                  
            )
            optimizer.step()                                                        
                
            B = mb.numel()
            acc["policy_loss"] += pg.item()             * B
            acc["value_loss"]  += vl.item()             * B                         
            acc["entropy"]     += entropy.mean().item() * B
            acc["approx_kl"]   += (data["logprobs"][mb] - new_logp).mean().item() * B               
            acc["n"]           += B                                                 
                
    n = acc.pop("n")                                                                
    return {k: v / n for k, v in acc.items()}
