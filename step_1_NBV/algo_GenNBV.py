"""
algo_GenNBV.py — GenNBV (CVPR 2024) 방식 PPO
==============================================

[논문 Sec 3.2 / Appendix A.2 기반 Multi-source State Embedding]

  F^G  : Probabilistic 3D occupancy grid → 2-layer 3D CNN + Linear
  F^S  : Grayscale RGB sequence (2 frames) → 2-layer 2D CNN + Linear
  s^A  : Pose scalar proxy (θ,φ,ψ) → Linear
  s_t  : Linear(concat(s^G, s^S, s^A)) → 256-dim    [논문 Eq.(7)]
  π    : 3-layer MLP(s_t) → Categorical (6 이산 행동)

[논문 대비 적응]
  Occupancy  : log-odds ray casting → TSDF 기반 3-state 근사
               (env.py의 _weight_vol + _tsdf_vol 재활용)
  Action     : 5D 연속 Normal → 6 이산 Categorical (기존 코드 호환)
  s^A        : 행동 이력 a_{1:t} → 현재 pose scalar (θ,φ,ψ) proxy
  Grid size  : 논문 20³ → 현재 env 40³ 유지
  Critic     : Symmetric (논문 동일), curr_coverage만 scalar에 추가

[논문 ablation 근거 — Table 3]
  Probabilistic 3D Grid alone : 84.56% FCR
  Semantic 2D Map alone       : 87.90% FCR
  3D Grid + Semantic          : 96.67% FCR  ← 조합이 핵심
  All 3 sources               : 98.26% FCR

[메모리 (T=256, E=8, 40³)]
  vox (uint8) : 256×8×3×40³ ≈ 393 MB
  img (float) : 256×8×2×84² ≈  92 MB
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
    ent_coef:       float = 0.05
    vf_coef:        float = 0.5
    max_grad_norm:  float = 0.5
    gamma:          float = 0.99
    lam:            float = 0.95


# ══════════════════════════════════════════════════════════════════════════════
# 인코더 상수
# ══════════════════════════════════════════════════════════════════════════════

_GEO_EMBED  = 256   # f^G 출력 dim
_SEM_EMBED  = 256   # f^S 출력 dim
_POSE_EMBED = 64    # pose scalar 임베딩 dim
_STATE_DIM  = 256   # 최종 s_t dim


# ══════════════════════════════════════════════════════════════════════════════
# F^G: Geometric Encoder  (논문: 2-layer 3D CNN + Linear(Flatten))
# ══════════════════════════════════════════════════════════════════════════════

class GeometricEncoder(nn.Module):
    """
    Input : (B, 3, Nx, Ny, Nz)  — 3-state TSDF voxel
              ch0 unknown : _weight_vol == 0
              ch1 free    : _weight_vol > 0 & _tsdf_vol > 0
              ch2 occupied: _weight_vol > 0 & _tsdf_vol <= 0
    Output: (B, _GEO_EMBED)
    """
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=3, padding=1), nn.ReLU(),   # spatial 유지
            nn.Conv3d(64,    64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # /2
            nn.AdaptiveAvgPool3d(4),   # (B, 64, 4, 4, 4)
            nn.Flatten(),              # (B, 4096)
        )
        self.proj = nn.Linear(4096, _GEO_EMBED)

    def forward(self, vox: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(vox))


# ══════════════════════════════════════════════════════════════════════════════
# F^S: Semantic Encoder  (논문: 2-layer 2D CNN + Linear(Flatten), M=2 프레임)
# ══════════════════════════════════════════════════════════════════════════════

class SemanticEncoder(nn.Module):
    """
    Input : (B, M, H, W)  — grayscale 시퀀스 (M=2, H=W=84)
    Output: (B, _SEM_EMBED)

    논문: 6 프레임 사용하나 ablation에서 2 프레임으로 충분함을 확인.
    H, W 변경 시 n_flat이 자동 계산됨.
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
# State Embedding  s_t = Linear(s^G ; s^S ; s^A)  [논문 Eq.(7)]
# ══════════════════════════════════════════════════════════════════════════════

class StateEmbedding(nn.Module):
    """
    논문 s_t = Linear(s_t^G ; s_t^S ; s_t^A) → 256-dim
    s^A: 논문에서 행동 이력 Linear(a_{1:t}),
         여기서는 현재 pose scalar (θ,φ,ψ) 사용 (proxy)
    """
    def __init__(self, img_ch: int = 2, scalar_dim: int = 3,
                 H: int = 84, W: int = 84):
        super().__init__()
        self.geo     = GeometricEncoder()
        self.sem     = SemanticEncoder(in_ch=img_ch, H=H, W=W)
        self.pose    = nn.Linear(scalar_dim, _POSE_EMBED)
        self.fusion  = nn.Linear(_GEO_EMBED + _SEM_EMBED + _POSE_EMBED, _STATE_DIM)

    def forward(self, vox: torch.Tensor, img: torch.Tensor,
                scalar: torch.Tensor) -> torch.Tensor:
        s_G = self.geo(vox)
        s_S = self.sem(img)
        s_A = F.relu(self.pose(scalar))
        return F.relu(self.fusion(torch.cat([s_G, s_S, s_A], dim=-1)))


# ══════════════════════════════════════════════════════════════════════════════
# Actor  (논문: 3-layer MLP → continuous Normal; 여기서는 Categorical)
# ══════════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """
    입력:
        vox    : (B, 3, Nx, Ny, Nz)   — 3-state voxel
        img    : (B, 2, H, W)          — grayscale 2프레임
        scalar : (B, 3)                — (θ, φ, ψ) normalized
    출력:
        action logits (B, n_actions)
    """
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
# Critic  (Symmetric — 논문 동일 설계, curr_coverage만 scalar에 추가)
# ══════════════════════════════════════════════════════════════════════════════

class Critic(nn.Module):
    """
    Actor와 동일한 관측 (vox, img) 사용.
    scalar_dim=4: (θ, φ, ψ, curr_coverage)
    """
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
    """
    vox (uint8) : (T, E, 3, Nx, Ny, Nz) — 메모리 절약, flat() 시 float 변환
    img (float) : (T, E, M, H, W)
    """
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

    TE  = adv.shape[0]
    acc = dict(policy_loss=0., value_loss=0., entropy=0., approx_kl=0., n=0)

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(TE, device=adv.device)
        for s in range(0, TE, cfg.minibatch_size):
            mb = perm[s : s + cfg.minibatch_size]
            if mb.numel() == 0:
                continue

            new_logp, entropy = actor.evaluate(
                data["vox"][mb], data["img"][mb],
                data["obs_scalar"][mb],
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

    n = max(acc.pop("n"), 1)
    return {k: v / n for k, v in acc.items()}


# ══════════════════════════════════════════════════════════════════════════════
# train_GenNBV.py 연동 참고
# ══════════════════════════════════════════════════════════════════════════════
#
# [초기화]
#   Nx, Ny, Nz = env.cfg.tsdf.vol_dim         # (40, 40, 40)
#   H, W  = env.cfg.visual.h, env.cfg.visual.w # (84, 84)
#   M     = 2                                   # semantic frames
#
#   actor  = Actor (img_ch=M, scalar_dim=3, H=H, W=W).to(device)
#   critic = Critic(img_ch=M, scalar_dim=4, H=H, W=W).to(device)
#   buf    = RolloutBuffer(T=256, E=E,
#                          Nx=Nx, Ny=Ny, Nz=Nz, M=M, H=H, W=W,
#                          scalar_dim=3, scalar_dim_critic=4,
#                          device=device)
#
# [env.py 필요 추가 — _get_observations() 내부]
#
#   def _get_vox_actor(self) -> torch.Tensor:  # (E, 3, Nx, Ny, Nz)
#       observed = self._weight_vol > 0
#       return torch.stack([
#           (~observed).float(),
#           (observed & (self._tsdf_vol  > 0)).float(),
#           (observed & (self._tsdf_vol <= 0)).float(),
#       ], dim=1)
#
#   # _get_observations() return dict에 추가:
#   "vox_actor"    : self._get_vox_actor(),              # (E, 3, Nx, Ny, Nz)
#   "img_semantic" : self._image_buffer[:, -2:, :, :],   # (E, 2, H, W) 마지막 2프레임
#   # "extra_info", "critic_scalar" 기존 유지
#
# [롤아웃 수집]
#   vox      = obs["vox_actor"]      # (E, 3, Nx, Ny, Nz)
#   img      = obs["img_semantic"]   # (E, 2, H, W)
#   scalar   = obs["extra_info"]     # (E, 3) — (θ,φ,ψ)
#   scalar_c = obs["critic_scalar"]  # (E, 4) — (θ,φ,ψ,coverage)
#
#   pose_act, logprob, _ = actor.sample(vox, img, scalar)
#   value                = critic(vox, img, scalar_c)
#   buf.add(vox, img, scalar, scalar_c,
#           pose_act, logprob, reward, done, value)
#
# [GAE 및 업데이트]
#   with torch.no_grad():
#       last_val = critic(obs["vox_actor"], obs["img_semantic"],
#                         obs["critic_scalar"])
#   buf.compute_gae(last_val, cfg_ppo.gamma, cfg_ppo.lam)
#   stats = ppo_update(actor, critic, opt_a, opt_c, buf, cfg_ppo)
#   buf.reset()
