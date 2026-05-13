from __future__ import annotations
from dataclasses import dataclass
import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 하이퍼파라미터
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DQNConfig:
    replay_capacity:    int   = 50_000
    batch_size:         int   = 32       # paper: 32 (single house)
    gamma:              float = 0.99
    lr:                 float = 1e-4
    eps_start:          float = 0.8      # paper: 0.8
    eps_end:            float = 0.05
    eps_decay:          float = 0.999    # paper: 0.999 per episode
    target_update_freq: int   = 1_000    # steps
    min_replay:         int   = 1_000    # learning 시작 최소 replay 크기
    max_grad_norm:      float = 10.0


# ══════════════════════════════════════════════════════════════════════════════
# CNN 출력 크기 (64×64 입력 기준)
#   Conv(32, k=8, s=4): floor((64-8)/4)+1 = 15
#   Conv(64, k=4, s=2): floor((15-4)/2)+1 = 6
#   Conv(64, k=3, s=1): floor((6-3)/1)+1  = 4
#   Flatten: 64 * 4 * 4 = 1024
# ══════════════════════════════════════════════════════════════════════════════
_CNN_OUT = 64 * 4 * 4


def _build_cnn(in_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(32,    64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(64,    64, kernel_size=3, stride=1), nn.ReLU(),
        nn.Flatten(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Network
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    ScanRL DQN Q-Network.

    입력: grayscale frame stack  (B, K, H, W)  — paper: K=6, H=W=84
    출력: Q-value per action     (B, n_actions)
    """
    def __init__(self, in_ch: int = 6, n_actions: int = 6):
        super().__init__()
        self.cnn = _build_cnn(in_ch)
        self.fc  = nn.Sequential(
            nn.Linear(_CNN_OUT, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(x))


# ══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-size circular replay buffer (off-policy)."""

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._buf: deque = deque(maxlen=capacity)

    def push(self,
             obs:      torch.Tensor,
             action:   int,
             reward:   float,
             next_obs: torch.Tensor,
             done:     bool) -> None:
        self._buf.append((obs.cpu(), action, reward, next_obs.cpu(), float(done)))

    def sample(self, batch_size: int):
        batch                         = random.sample(self._buf, batch_size)
        obs, acts, rews, next_obs, dones = zip(*batch)
        return (
            torch.stack(obs)    .to(self.device),
            torch.tensor(acts,  dtype=torch.long,  device=self.device),
            torch.tensor(rews,  dtype=torch.float, device=self.device),
            torch.stack(next_obs).to(self.device),
            torch.tensor(dones, dtype=torch.float, device=self.device),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
# DQN Update (Double DQN)
# ══════════════════════════════════════════════════════════════════════════════

def dqn_update(
    q_net:      QNetwork,
    target_net: QNetwork,
    optimizer:  torch.optim.Optimizer,
    replay:     ReplayBuffer,
    cfg:        DQNConfig,
) -> dict:
    """
    Double DQN update step.
    q_net으로 action 선택, target_net으로 Q-value 평가.
    """
    obs, acts, rews, next_obs, dones = replay.sample(cfg.batch_size)

    with torch.no_grad():
        next_acts = q_net(next_obs).argmax(dim=1)
        next_q    = target_net(next_obs).gather(1, next_acts.unsqueeze(1)).squeeze(1)
        target_q  = rews + cfg.gamma * next_q * (1.0 - dones)

    current_q = q_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(q_net.parameters(), cfg.max_grad_norm)
    optimizer.step()

    return {
        "loss":         loss.item(),
        "q_mean":       current_q.mean().item(),
        "q_max":        current_q.max().item(),
        "target_mean":  target_q.mean().item(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 유틸
# ══════════════════════════════════════════════════════════════════════════════

def make_env_action(action_idx: torch.Tensor, E: int, device) -> torch.Tensor:
    """(E,) action index → (E, 6) one-hot, env._apply_action()과 호환."""
    act = torch.zeros(E, 6, device=device)
    act.scatter_(1, action_idx.unsqueeze(1), 1.0)
    return act
