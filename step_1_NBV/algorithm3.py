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
    ppo_epochs:     int   = 6
    minibatch_size: int   = 256
    clip_eps:       float = 0.2
    ent_coef:       float = 0.05
    vf_coef:        float = 0.5
    max_grad_norm:  float = 0.5
    gamma:          float = 0.99
    lam:            float = 0.95
    target_kl:      float = 0.02


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
    입력:
        img    : 이미지 시퀀스 (B, K_img, H, W)
                 use_visit_map=False → (B, num_seq_actor,   H, W)
                 use_visit_map=True  → (B, num_seq_actor+1, H, W)
        scalar : (B, scalar_dim)  — (θ, φ, ψ), 항상 dim=3
    """
    def __init__(self, img_ch: int = 6, scalar_dim: int = 3, n_pose: int = 6):
        super().__init__()
        self.cnn = _build_cnn(img_ch)
        self.mlp = nn.Sequential(
            nn.Linear(_CNN_OUT + scalar_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),                   nn.ReLU(),
        )
        self.pose_head = nn.Linear(256, n_pose)

    def _dist(self, img: torch.Tensor, scalar: torch.Tensor) -> Categorical:
        feat = self.cnn(img)
        feat = self.mlp(torch.cat([feat, scalar], dim=-1))
        return Categorical(logits=self.pose_head(feat))

    def greedy(self, img: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        return self._dist(img, scalar).logits.argmax(dim=-1)

    def sample(self, img: torch.Tensor, scalar: torch.Tensor):
        dist = self._dist(img, scalar)
        act  = dist.sample()
        return act, dist.log_prob(act), dist.entropy()

    def evaluate(self, img: torch.Tensor, scalar: torch.Tensor,
                 pose_act: torch.Tensor):
        dist = self._dist(img, scalar)
        return dist.log_prob(pose_act), dist.entropy()


class Critic(nn.Module):
    """
    Asymmetric Critic — GT depth 시퀀스 + 스칼라.

    입력:
        depth  : GT depth 시퀀스 (B, K_dep, H, W)
                 use_visit_map=False → (B, num_seq_critic,   H, W)
                 use_visit_map=True  → (B, num_seq_critic+1, H, W)
        scalar : (B, scalar_dim_critic)
                 use_visit_map=False → dim=3  (θ, φ, ψ)
                 use_visit_map=True  → dim=4  (θ, φ, ψ, curr_coverage)
    """
    def __init__(self, depth_ch: int = 6, scalar_dim: int = 3):
        super().__init__()
        self.cnn = _build_cnn(depth_ch)
        self.mlp = nn.Sequential(
            nn.Linear(_CNN_OUT + scalar_dim, 512), nn.ReLU(),
            nn.Linear(512,                  256), nn.ReLU(),
            nn.Linear(256,                    1),
        )

    def forward(self, depth: torch.Tensor,
                scalar: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(depth)
        return self.mlp(torch.cat([feat, scalar], dim=-1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    algorithm2.py 대비 변경:
        - scalar_dim_critic 파라미터 추가
        - obs_scalar_critic 버퍼 추가 (Actor scalar와 독립)
        - add() / flat() 인터페이스 갱신

    Actor scalar (dim=3)와 Critic scalar (dim=3 or 4)를 분리 저장하여
    use_visit_map 시 Critic이 curr_coverage를 추가로 받는 구조를 지원.
    """
    def __init__(self, T: int, E: int, K_img: int, K_dep: int,
                 H: int, W: int,
                 scalar_dim: int, scalar_dim_critic: int,
                 device):
        self.T, self.E, self.ptr = T, E, 0
        kw = dict(device=device)
        self.obs_img           = torch.zeros(T, E, K_img, H, W,          **kw)
        self.obs_scalar        = torch.zeros(T, E, scalar_dim,            **kw)
        self.obs_scalar_critic = torch.zeros(T, E, scalar_dim_critic,     **kw)
        self.obs_depth         = torch.zeros(T, E, K_dep, H, W,          **kw)
        self.pose_acts         = torch.zeros(T, E, dtype=torch.long,      **kw)
        self.logprobs          = torch.zeros(T, E,                        **kw)
        self.rewards           = torch.zeros(T, E,                        **kw)
        self.dones             = torch.zeros(T, E,                        **kw)
        self.values            = torch.zeros(T, E,                        **kw)
        self.returns:    torch.Tensor
        self.advantages: torch.Tensor

    def add(self, obs_img, obs_scalar, obs_scalar_critic, obs_depth,
            pose_act, logprob, reward, done, value):
        t = self.ptr
        self.obs_img[t]           = obs_img
        self.obs_scalar[t]        = obs_scalar
        self.obs_scalar_critic[t] = obs_scalar_critic
        self.obs_depth[t]         = obs_depth
        self.pose_acts[t]         = pose_act
        self.logprobs[t]          = logprob
        self.rewards[t]           = reward
        self.dones[t]             = done
        self.values[t]            = value
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
            "obs_img":           self.obs_img          .reshape(TE, *self.obs_img.shape[2:]),
            "obs_scalar":        self.obs_scalar        .reshape(TE, *self.obs_scalar.shape[2:]),
            "obs_scalar_critic": self.obs_scalar_critic .reshape(TE, *self.obs_scalar_critic.shape[2:]),
            "obs_depth":         self.obs_depth         .reshape(TE, *self.obs_depth.shape[2:]),
            "pose_acts":         self.pose_acts         .reshape(TE),
            "logprobs":          self.logprobs          .reshape(TE),
            "returns":           self.returns           .reshape(TE),
            "advantages":        self.advantages        .reshape(TE),
            "old_values":        self.values            .reshape(TE),
        }

    def reset(self):
        self.ptr = 0


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════

def make_env_action(pose_idx: torch.Tensor, E: int, device) -> torch.Tensor:
    """(E,) pose index → (E, 6) one-hot."""
    act = torch.zeros(E, 6, device=device)
    act.scatter_(1, pose_idx.unsqueeze(1), 1.0)
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
               optimizer_actor:  torch.optim.Optimizer,
               optimizer_critic: torch.optim.Optimizer,
               buf: RolloutBuffer,
               cfg: PPOConfig) -> dict:
    data = buf.flat()
    adv  = data["advantages"]
    adv  = (adv - adv.mean()) / (adv.std() + 1e-8)

    TE         = adv.shape[0]
    acc        = dict(policy_loss=0., value_loss=0., entropy=0., approx_kl=0., n=0)
    early_stop = False

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(TE, device=adv.device)
        for s in range(0, TE, cfg.minibatch_size):
            mb = perm[s : s + cfg.minibatch_size]
            if mb.numel() == 0:
                continue

            new_logp, entropy = actor.evaluate(
                data["obs_img"][mb], data["obs_scalar"][mb],
                data["pose_acts"][mb],
            )

            approx_kl_mb = (data["logprobs"][mb] - new_logp).mean().item()

            ratio  = (new_logp - data["logprobs"][mb]).exp()
            mb_adv = adv[mb]

            pg = torch.max(
                -mb_adv * ratio,
                -mb_adv * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),
            ).mean()

            actor_loss = pg - cfg.ent_coef * entropy.mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            optimizer_actor.step()

            # Critic은 obs_scalar_critic 사용 (Actor scalar와 dim이 다를 수 있음)
            v = critic(data["obs_depth"][mb], data["obs_scalar_critic"][mb])
            v_clipped = data["returns"][mb] + (v - data["old_values"][mb]).clamp(
                -cfg.clip_eps, cfg.clip_eps
            )
            vl = torch.max(
                F.mse_loss(v,         data["returns"][mb]),
                F.mse_loss(v_clipped, data["returns"][mb]),
            )

            critic_loss = cfg.vf_coef * vl
            optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            optimizer_critic.step()

            B = mb.numel()
            acc["policy_loss"] += pg.item()             * B
            acc["value_loss"]  += vl.item()             * B
            acc["entropy"]     += entropy.mean().item() * B
            acc["approx_kl"]   += approx_kl_mb          * B
            acc["n"]           += B

        if early_stop:
            break

    n = max(acc.pop("n"), 1)
    return {k: v / n for k, v in acc.items()} | {"early_stop": early_stop}


# ══════════════════════════════════════════════════════════════════════════════
# train_RL.py 연동 참고
# ══════════════════════════════════════════════════════════════════════════════
#
# [초기화 — use_visit_map=False (baseline)]
#   K_img          = 6          # num_seq_actor
#   K_dep          = 6          # num_seq_critic
#   s_dim_actor    = 3          # (θ, φ, ψ)
#   s_dim_critic   = 3
#
# [초기화 — use_visit_map=True]
#   K_img          = num_seq_actor + 1   # RGB seq + visit_map
#   K_dep          = num_seq_critic + 1  # depth seq + visit_map
#   s_dim_actor    = 3
#   s_dim_critic   = 4          # (θ, φ, ψ, curr_coverage)
#
#   actor  = Actor(img_ch=K_img, scalar_dim=s_dim_actor).to(device)
#   critic = Critic(depth_ch=K_dep, scalar_dim=s_dim_critic).to(device)
#   buf    = RolloutBuffer(T, E, K_img, K_dep, H, W,
#                          scalar_dim=s_dim_actor,
#                          scalar_dim_critic=s_dim_critic, device=device)
#
# [롤아웃 수집]
#   obs = env.reset()
#   for step in range(T):
#       img          = obs["policy"]         # (E, K_img, H, W)
#       scalar       = obs["extra_info"]     # (E, 3)
#       depth        = obs["critic"]         # (E, K_dep, H, W)
#       scalar_crit  = obs["critic_scalar"]  # (E, 3 or 4)
#
#       with torch.no_grad():
#           pose_act, logprob, _ = actor.sample(img, scalar)
#           value                = critic(depth, scalar_crit)
#
#       env_act = make_env_action(pose_act, E, device)
#       obs, reward, done, _ = env.step(env_act)
#       buf.add(img, scalar, scalar_crit, depth,
#               pose_act, logprob, reward, done, value)
#
# [GAE 및 업데이트]
#   with torch.no_grad():
#       last_val = critic(obs["critic"], obs["critic_scalar"])
#   buf.compute_gae(last_val, cfg_ppo.gamma, cfg_ppo.lam)
#   stats = ppo_update(actor, critic, optimizer_actor, optimizer_critic, buf, cfg_ppo)
#   buf.reset()
