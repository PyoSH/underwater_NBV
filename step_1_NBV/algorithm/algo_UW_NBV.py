"""
algo_UW_NBV.py — UW-NBV PPO (algo_GenNBV 기반 개선)
=====================================================

algo_GenNBV.py 대비 변경 사항:

1. PPOConfig.target_kl 추가 (기본 0.02)
   - 에포크 단위 평균 KL이 target_kl 초과 시 조기 종료
   - 성공 배치에서 과도한 policy 업데이트 방지 (UW_NBV_3 붕괴 원인)

2. PPOConfig.ent_coef 기본값 0.05 → 0.03
   - coverage 신호 대비 entropy 보너스 비중 축소
   - 수렴 단계에서 불필요한 탐색 억제

3. ppo_update 반환값에 "early_stop" 추가
   - 조기 종료 여부를 wandb / 로그에서 모니터링 가능

권장 학습 설정 (train_GenNBV_quality.py):
  --rollout_steps 512   (minibatch 수 8개 확보)
  --ent_coef      0.03
  coverage_bonus  10.0  (envCfg / train 스크립트에서 설정)
"""

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
    ent_coef:       float = 0.03    # 0.05 → 0.03: coverage 신호 대비 entropy 비중 축소
    vf_coef:        float = 0.5
    max_grad_norm:  float = 0.5
    gamma:          float = 0.99
    lam:            float = 0.95
    target_kl:      float = 0.02    # 에포크 평균 KL 초과 시 조기 종료 (추가)


# ══════════════════════════════════════════════════════════════════════════════
# 인코더 상수
# ══════════════════════════════════════════════════════════════════════════════

_GEO_EMBED  = 256
_SEM_EMBED  = 256
_POSE_EMBED = 64
_STATE_DIM  = 256


# ══════════════════════════════════════════════════════════════════════════════
# F^G: Geometric Encoder
# ══════════════════════════════════════════════════════════════════════════════

class GeometricEncoder(nn.Module):
    """
    Input : (B, 3, Nx, Ny, Nz)
              ch0 unknown : weight == 0
              ch1 free    : weight > 0 & tsdf > 0
              ch2 quality : clip(quality_vol / Q_sat, 0, 1)
    Output: (B, _GEO_EMBED)
    """
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64,    64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(4),
            nn.Flatten(),
        )
        self.proj = nn.Linear(4096, _GEO_EMBED)

    def forward(self, vox: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(vox))


# ══════════════════════════════════════════════════════════════════════════════
# F^S: Semantic Encoder
# ══════════════════════════════════════════════════════════════════════════════

class SemanticEncoder(nn.Module):
    """
    Input : (B, M, H, W) — grayscale 시퀀스 (M=2)
    Output: (B, _SEM_EMBED)
    """
    def __init__(self, in_ch: int = 2, H: int = 84, W: int = 84):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,    64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, in_ch, H, W)).shape[1]
        self.proj = nn.Linear(n_flat, _SEM_EMBED)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(img))


# ══════════════════════════════════════════════════════════════════════════════
# State Embedding
# ══════════════════════════════════════════════════════════════════════════════

class StateEmbedding(nn.Module):
    def __init__(self, img_ch: int = 2, scalar_dim: int = 3,
                 H: int = 84, W: int = 84):
        super().__init__()
        self.geo    = GeometricEncoder()
        self.sem    = SemanticEncoder(in_ch=img_ch, H=H, W=W)
        self.pose   = nn.Linear(scalar_dim, _POSE_EMBED)
        self.fusion = nn.Linear(_GEO_EMBED + _SEM_EMBED + _POSE_EMBED, _STATE_DIM)

    def forward(self, vox: torch.Tensor, img: torch.Tensor,
                scalar: torch.Tensor) -> torch.Tensor:
        s_G = self.geo(vox)
        s_S = self.sem(img)
        s_A = F.relu(self.pose(scalar))
        return F.relu(self.fusion(torch.cat([s_G, s_S, s_A], dim=-1)))


# ══════════════════════════════════════════════════════════════════════════════
# Actor
# ══════════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    def __init__(self, img_ch: int = 2, scalar_dim: int = 3,
                 n_actions: int = 6, H: int = 84, W: int = 84):
        super().__init__()
        self.embed = StateEmbedding(img_ch, scalar_dim, H, W)
        self.mlp   = nn.Sequential(
            nn.Linear(_STATE_DIM, 256), nn.ReLU(),
            nn.Linear(256,        256), nn.ReLU(),
            nn.Linear(256,        n_actions),
        )

    def _dist(self, vox, img, scalar) -> Categorical:
        return Categorical(logits=self.mlp(self.embed(vox, img, scalar)))

    def greedy(self, vox, img, scalar) -> torch.Tensor:
        return self._dist(vox, img, scalar).logits.argmax(dim=-1)

    def sample(self, vox, img, scalar):
        dist = self._dist(vox, img, scalar)
        act  = dist.sample()
        return act, dist.log_prob(act), dist.entropy()

    def evaluate(self, vox, img, scalar, actions):
        dist = self._dist(vox, img, scalar)
        return dist.log_prob(actions), dist.entropy()


# ══════════════════════════════════════════════════════════════════════════════
# Critic
# ══════════════════════════════════════════════════════════════════════════════

class Critic(nn.Module):
    """scalar_dim=4: (θ, φ, ψ, curr_coverage_q)"""
    def __init__(self, img_ch: int = 2, scalar_dim: int = 4,
                 H: int = 84, W: int = 84):
        super().__init__()
        self.embed = StateEmbedding(img_ch, scalar_dim, H, W)
        self.mlp   = nn.Sequential(
            nn.Linear(_STATE_DIM, 256), nn.ReLU(),
            nn.Linear(256,        256), nn.ReLU(),
            nn.Linear(256,          1),
        )

    def forward(self, vox, img, scalar) -> torch.Tensor:
        return self.mlp(self.embed(vox, img, scalar)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """vox (uint8): 메모리 절약, flat() 시 float 변환"""
    def __init__(self, T: int, E: int,
                 Nx: int, Ny: int, Nz: int,
                 M: int, H: int, W: int,
                 scalar_dim: int, scalar_dim_critic: int,
                 device):
        self.T, self.E, self.ptr = T, E, 0
        kw = dict(device=device)

        self.vox               = torch.zeros(T, E, 3, Nx, Ny, Nz, dtype=torch.uint8, **kw)
        self.img               = torch.zeros(T, E, M, H, W,                          **kw)
        self.obs_scalar        = torch.zeros(T, E, scalar_dim,                        **kw)
        self.obs_scalar_critic = torch.zeros(T, E, scalar_dim_critic,                 **kw)
        self.pose_acts         = torch.zeros(T, E, dtype=torch.long,                  **kw)
        self.logprobs          = torch.zeros(T, E,                                    **kw)
        self.rewards           = torch.zeros(T, E,                                    **kw)
        self.dones             = torch.zeros(T, E,                                    **kw)
        self.values            = torch.zeros(T, E,                                    **kw)

    def add(self, vox, img, obs_scalar, obs_scalar_critic,
            pose_act, logprob, reward, done, value):
        t = self.ptr
        self.vox              [t] = vox.to(torch.uint8)
        self.img              [t] = img
        self.obs_scalar       [t] = obs_scalar
        self.obs_scalar_critic[t] = obs_scalar_critic
        self.pose_acts        [t] = pose_act
        self.logprobs         [t] = logprob
        self.rewards          [t] = reward
        self.dones            [t] = done
        self.values           [t] = value
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
        def _r(x): return x.reshape(TE, *x.shape[2:])
        return {
            "vox":               _r(self.vox).float(),
            "img":               _r(self.img),
            "obs_scalar":        _r(self.obs_scalar),
            "obs_scalar_critic": _r(self.obs_scalar_critic),
            "pose_acts":         _r(self.pose_acts),
            "logprobs":          _r(self.logprobs),
            "returns":           _r(self.returns),
            "advantages":        _r(self.advantages),
            "old_values":        _r(self.values),
        }

    def reset(self):
        self.ptr = 0


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════

def make_env_action(pose_idx: torch.Tensor, E: int, device) -> torch.Tensor:
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

    for epoch in range(cfg.ppo_epochs):
        perm        = torch.randperm(TE, device=adv.device)
        epoch_kl    = 0.0
        epoch_n     = 0

        for s in range(0, TE, cfg.minibatch_size):
            mb = perm[s : s + cfg.minibatch_size]
            if mb.numel() == 0:
                continue

            new_logp, entropy = actor.evaluate(
                data["vox"][mb], data["img"][mb],
                data["obs_scalar"][mb],
                data["pose_acts"][mb],
            )

            # abs(log-ratio): policy가 더 확신하는 방향도 감지 (signed KL은 음수 가능)
            approx_kl_mb = (data["logprobs"][mb] - new_logp).abs().mean().item()
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

            v = critic(data["vox"][mb], data["img"][mb],
                       data["obs_scalar_critic"][mb])
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
            epoch_kl            += approx_kl_mb          * B
            epoch_n             += B

        # 에포크 단위 평균 KL 체크 — target_kl 초과 시 조기 종료
        if epoch_n > 0 and (epoch_kl / epoch_n) > cfg.target_kl:
            early_stop = True
            break

    n = max(acc.pop("n"), 1)
    return {k: v / n for k, v in acc.items()} | {"early_stop": early_stop}
