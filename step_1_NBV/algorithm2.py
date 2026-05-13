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
    ppo_epochs:     int   = 6      # 4 → 10: NBV 희소 보상 환경에서 샘플 효율 향상
    minibatch_size: int   = 256
    clip_eps:       float = 0.2
    ent_coef:       float = 0.05    # 0.05 → 0.01: 단일 Categorical 분포 엔트로피 스케일 조정
    vf_coef:        float = 0.5
    max_grad_norm:  float = 0.5
    gamma:          float = 0.99    # 장거리 Δcoverage 보상 반영
    lam:            float = 0.95    # GAE λ
    target_kl:      float = 0.02

# ══════════════════════════════════════════════════════════════════════════════
# Network
# ══════════════════════════════════════════════════════════════════════════════

# 84×84: Conv(8,s4)→20, Conv(4,s2)→9, Conv(3,s1)→7  →  64*7*7 = 3136
_CNN_OUT = 64 * 4 * 4


def _build_cnn(in_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(32,    64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(64,    64, kernel_size=3, stride=1), nn.ReLU(),
        nn.Flatten(),
    )


class Actor(nn.Module):
    """
    NBV 전용 Actor.

    입력:
        img    : RGB 흑백 이미지 시퀀스  (B, K_img, H, W)   — DR 적용
        scalar : 구면좌표 정규화 벡터    (B, 3)              — (θ, φ, ψ)

    출력:
        단일 Categorical 분포 → pose action 6개
            (+Δθ, -Δθ, +Δφ, -Δφ, +Δψ, -Δψ)

    변경 사항 (기존 대비):
        - light_head 제거 (조명 행동 없음)
        - scalar_dim 5 → 3 (contrast, light_level 제거)
        - sample / evaluate 단일 분포로 단순화
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
        """결정론적 행동 (평가용): logits argmax."""
        return self._dist(img, scalar).logits.argmax(dim=-1)

    def sample(self, img: torch.Tensor, scalar: torch.Tensor):
        """
        탐색용 샘플링.
        반환: (pose_act, logprob, entropy)
        """
        dist    = self._dist(img, scalar)
        act     = dist.sample()
        return act, dist.log_prob(act), dist.entropy()

    def evaluate(self, img: torch.Tensor, scalar: torch.Tensor,
                 pose_act: torch.Tensor):
        """
        PPO 업데이트용 재평가.
        반환: (logprob, entropy)
        """
        dist = self._dist(img, scalar)
        return dist.log_prob(pose_act), dist.entropy()


class Critic(nn.Module):
    """
    Asymmetric Critic — GT depth 시퀀스 + 구면좌표 스칼라.

    입력:
        depth  : GT depth map 시퀀스  (B, K_dep, H, W)  — DR 없는 특권 정보
        scalar : 구면좌표 정규화 벡터 (B, 4)             — (θ, φ, ψ, curr_coverage) # to-be

    변경 사항 (기존 대비):
        - scalar 입력 추가: 동일 geometry라도 위치(거리·각도)에 따라
          기대 커버리지가 달라지므로 가치 추정 정확도 향상
        - scalar_dim 3 고정 (조명값 미포함 — 조명 고정 운용 전제)
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
        """반환: (B,) 상태 가치 V(s)."""
        feat = self.cnn(depth)
        return self.mlp(torch.cat([feat, scalar], dim=-1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    변경 사항 (기존 대비):
        - light_acts 버퍼 제거
        - scalar_dim 3으로 고정
        - Critic forward 시그니처 변경에 맞춰 flat() 키 유지
    """
    def __init__(self, T: int, E: int, K_img: int, K_dep: int,
                 H: int, W: int, scalar_dim: int, device):
        self.T, self.E, self.ptr = T, E, 0
        kw = dict(device=device)
        self.obs_img    = torch.zeros(T, E, K_img, H, W, **kw)
        self.obs_scalar = torch.zeros(T, E, scalar_dim,  **kw)
        self.obs_depth  = torch.zeros(T, E, K_dep, H, W, **kw)
        self.pose_acts  = torch.zeros(T, E, dtype=torch.long, **kw)
        # light_acts 제거
        self.logprobs   = torch.zeros(T, E, **kw)
        self.rewards    = torch.zeros(T, E, **kw)
        self.dones      = torch.zeros(T, E, **kw)
        self.values     = torch.zeros(T, E, **kw)
        self.returns:    torch.Tensor
        self.advantages: torch.Tensor

    def add(self, obs_img, obs_scalar, obs_depth,
            pose_act, logprob, reward, done, value):
        """
        light_act 인자 제거.
        호출부: buf.add(img, scalar, depth, pose_act, logprob, reward, done, value)
        """
        t = self.ptr
        self.obs_img[t]    = obs_img
        self.obs_scalar[t] = obs_scalar
        self.obs_depth[t]  = obs_depth
        self.pose_acts[t]  = pose_act
        self.logprobs[t]   = logprob
        self.rewards[t]    = reward
        self.dones[t]      = done
        self.values[t]     = value
        self.ptr += 1

    def compute_gae(self, last_value: torch.Tensor, gamma: float, lam: float):
        """
        GAE (Generalized Advantage Estimation).

        last_value: Critic이 롤아웃 마지막 상태에서 추정한 V(s_T).
        Δcoverage 보상이 희소하므로 gamma=0.99, lam=0.95 권고.
        """
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
            # light_acts 키 제거
            "logprobs":   self.logprobs  .reshape(TE),
            "returns":    self.returns   .reshape(TE),
            "advantages": self.advantages.reshape(TE),
            "old_values": self.values    .reshape(TE),
        }

    def reset(self):
        self.ptr = 0


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════

def make_env_action(pose_idx: torch.Tensor, E: int, device) -> torch.Tensor:
    """
    (E,) pose index → (E, 6) one-hot.

    변경 사항 (기존 대비):
        - light_idx 인자 제거
        - 출력 shape (E, 9) → (E, 6)
        - _apply_action()의 actions[:, 0:6].argmax() 구조에 대응

    슬롯:
        0: +Δθ  1: -Δθ
        2: +Δφ  3: -Δφ
        4: +Δψ  5: -Δψ
    """
    act = torch.zeros(E, 6, device=device)
    act.scatter_(1, pose_idx.unsqueeze(1), 1.0)
    return act


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """Critic 성능 지표. 1.0에 가까울수록 가치 추정 정확."""
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

            #   approx_kl_mb = (data["logprobs"][mb] - new_logp).mean().item()
              approx_kl_mb = (data["logprobs"][mb] - new_logp).abs().mean().item()
              if approx_kl_mb > cfg.target_kl:
                  early_stop = True
                  break

              ratio  = (new_logp - data["logprobs"][mb]).exp()
              mb_adv = adv[mb]

              pg = torch.max(
                  -mb_adv * ratio,
                  -mb_adv * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps),
              ).mean()

              v = critic(data["obs_depth"][mb], data["obs_scalar"][mb])  # ← 수정
              v_clipped = data["returns"][mb] + (v - data["old_values"][mb]).clamp(
                  -cfg.clip_eps, cfg.clip_eps
              )
              vl = torch.max(
                  F.mse_loss(v,         data["returns"][mb]),
                  F.mse_loss(v_clipped, data["returns"][mb]),
              )

              loss = pg + cfg.vf_coef * vl - cfg.ent_coef * entropy.mean()
              optimizer.zero_grad()
              loss.backward()
            #   nn.utils.clip_grad_norm_(
            #       list(actor.parameters()) + list(critic.parameters()),
            #       cfg.max_grad_norm,
            #   )
              nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
              nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
              optimizer.step()

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
# train.py 연동 참고
# ══════════════════════════════════════════════════════════════════════════════
#
# [초기화]
# cfg_ppo = PPOConfig()
# actor   = Actor(img_ch=6, scalar_dim=3, n_pose=6).to(device)
# critic  = Critic(depth_ch=6, scalar_dim=3).to(device)
# optim   = torch.optim.Adam(
#               list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
# buf     = RolloutBuffer(T=2048, E=num_envs, K_img=6, K_dep=6,
#                         H=84, W=84, scalar_dim=3, device=device)
#
# [롤아웃 수집]
# obs       = env.reset()
# for step in range(T):
#     img    = obs["policy"]      # (E, 6, 84, 84)
#     scalar = obs["extra_info"]  # (E, 3) — θ, φ, ψ만
#     depth  = obs["critic"]      # (E, 6, 84, 84)
#
#     with torch.no_grad():
#         pose_act, logprob, _ = actor.sample(img, scalar)
#         value                = critic(depth, scalar)
#
#     env_act = make_env_action(pose_act, E=num_envs, device=device)
#     obs, reward, done, _ = env.step(env_act)
#
#     buf.add(img, scalar, depth, pose_act, logprob, reward, done, value)
#
# [GAE 및 업데이트]
# with torch.no_grad():
#     last_val = critic(obs["critic"], obs["extra_info"])
# buf.compute_gae(last_val, cfg_ppo.gamma, cfg_ppo.lam)
# stats = ppo_update(actor, critic, optim, buf, cfg_ppo)
# buf.reset()