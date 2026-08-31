"""
algo_nbv_continuous.py — step_3 NBV PPO (연속 액션)
======================================================
`step_1_NBV/algorithm/algo_UW_NBV.py`를 기반으로 **액션 헤드만 연속화**한 것.

왜 step_1 알고리즘을 그대로 가져왔는가
--------------------------------------
step_3의 성패는 "step_1 대비 물리 제약이 들어가서 얼마나 어려워졌나"로 해석해야
하는데, 알고리즘이 다르면 그 비교가 성립하지 않는다. 인코더 3종(3D CNN/2D CNN/
pose MLP), PPO 루프, `RolloutBuffer`, `target_kl` 조기종료를 전부 동일하게 두고
**액션 공간만 다른** 상태를 만드는 것이 실험 설계상 깔끔하다.
(`target_kl=0.02`는 step_1에서 UW_NBV_3 붕괴를 막은 실적이 있는 설정이다.)

step_1 대비 변경점 (이 4가지가 전부)
------------------------------------
1. `Categorical(logits)` → **tanh-squashed `Normal(mu, std)`**
   - `log_std`가 학습 가능한 파라미터(상태 무관, PPO 표준 관례)
2. `RolloutBuffer.pose_acts`(long) → `actions`(float, action_dim)
   - **저장하는 값은 tanh 적용 *전*의 `u`**(아래 설명)
3. `make_env_action()` 삭제 — 이산 인덱스→one-hot 변환이 불필요해짐
4. log_prob/entropy가 다차원이므로 `.sum(-1)`

왜 tanh 스쿼시인가
------------------
step_2(Sim2Swim)는 unsquashed Gaussian + 환경측 클리핑을 썼는데, raw actor 출력이
`[-1,1]`을 벗어나는 비율이 99%까지 올라가 별도 페널티 항(`deploy_penalty_raw_
overflow_l2`)을 만들어 억눌러야 했다([[project_step2_brov_retrain_spec]]).
tanh는 경계를 **구조적으로** 보장해 그 실패 모드 자체를 없앤다.

왜 buffer에 `u`(pre-tanh)를 저장하는가
--------------------------------------
PPO ratio는 "같은 액션"에 대한 new/old log-prob 비율이어야 한다. squash된 `a`만
저장하면 갱신 시 `atanh(a)`로 역산해야 하는데 `a`가 ±1 근처면 수치적으로 발산한다.
`u`를 저장하면 역산 없이 `new_dist.log_prob(u)`를 바로 쓸 수 있고 `a=tanh(u)`도
결정론적으로 복원되므로 정확하고 안전하다.

entropy 주의
------------
tanh-squashed 분포의 엔트로피는 해석해가 없다. 여기서는 **squash 이전 Normal의
엔트로피를 대리값으로** 쓴다(PPO 구현에서 널리 쓰이는 관례). 절대값 자체보다
`ent_coef`와의 상대 스케일이 중요하고, 붕괴 감지용 모니터링 지표로는 충분하다.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ══════════════════════════════════════════════════════════════════════════════
# PPO 하이퍼파라미터 (step_1 algo_UW_NBV.py와 동일 — 비교 가능성 유지)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PPOConfig:
    ppo_epochs:     int   = 6
    minibatch_size: int   = 256
    clip_eps:       float = 0.2
    ent_coef:       float = 0.03
    vf_coef:        float = 0.5
    max_grad_norm:  float = 0.5
    gamma:          float = 0.99
    lam:            float = 0.95
    target_kl:      float = 0.02    # 에포크 평균 KL 초과 시 조기 종료


_GEO_EMBED  = 256
_SEM_EMBED  = 256
_POSE_EMBED = 64
_STATE_DIM  = 256
_LOG_STD_MIN, _LOG_STD_MAX = -5.0, 2.0
_TANH_EPS = 1e-6

# 정책 평균 mu의 소프트 상한 (2026-08-28 추가).
#
# 왜 필요한가: std는 위에서 클램프되는데 `mu = self.mlp(...)`는 아무 제약이
# 없었다. 2026-08-27 런에서 mu가 **±256까지 발산**해(관측 approx_kl 87,477에서
# 역산) tanh가 완전히 포화됐고, 평가에서 액션의 **89%가 |a|>0.99**로 확인됐다.
# 그 상태에서는 ratio=exp(-87,477)=0으로 언더플로되고, clamp가 범위 밖이라
# 기울기가 정확히 0이 되어 actor가 학습 신호를 전혀 못 받는다.
#
# 3.0인 이유: tanh(3)=0.995라 액션 공간 [-1,1]의 끝까지 표현할 수 있으면서,
# 이보다 큰 값은 표현력을 늘리지 못하고 log-prob만 폭발시킨다. 하드 클램프
# 대신 소프트(mu = M·tanh(raw/M))를 쓰는 이유는 하드 클램프가 경계에서
# 기울기를 0으로 만들어 한 번 붙으면 빠져나올 수 없기 때문이다.
_MU_MAX = 3.0

# log-ratio를 exp 하기 전 클램프할 범위. exp(20)≈4.9e8, exp(-20)≈2e-9로
# 충분히 넓으면서 inf/0 언더플로를 막는다. 위 _MU_MAX가 근본 원인을 없애지만,
# 학습 초기 큰 갱신에서도 수치가 죽지 않도록 두는 안전장치다.
_LOG_RATIO_CLAMP = 20.0

# 미니배치 단위 KL 가드의 임계값 = max(target_kl × _MB_KL_SLACK, _MB_KL_FLOOR).
#
# 이 가드의 역할은 **발산 감지 하나뿐**이다. 정상적인 trust-region 제한은
# 에포크 평균 검사(target_kl)가 담당한다. 둘을 혼동해 임계값을 좁게 잡으면
# 안 된다 — 2026-08-29 검증에서 배수 3.0(트립 0.36)으로 뒀다가, abs() 기반
# KL이 정상 상태에서도 0.22~0.31에 이르는 탓에 미니배치 단위로는 routinely
# 초과했고, 첫 미니배치에서 걸린 롤아웃 4개(005/009/010/011)가 갱신 0회로
# 통째로 버려졌다.
#
# 발산은 1,756 이상으로 튀므로 임계값이 1.2든 0.36이든 똑같이 즉시 잡힌다.
# 넉넉하게 잡을수록 정상 변동을 안 건드리므로 손해가 없다.
_MB_KL_SLACK = 10.0
_MB_KL_FLOOR = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 인코더 3종 — step_1에서 무수정 이식
# ══════════════════════════════════════════════════════════════════════════════

class GeometricEncoder(nn.Module):
    """(B, 3, Nx, Ny, Nz) → (B, _GEO_EMBED).

    ch0 unknown / ch1 free / ch2 occupied — `envs/env.py::_get_vox_actor()` 참조.
    (step_1 주석은 ch2를 quality라 했지만 step_3는 binary occupancy를 쓴다.)
    """
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv3d(64,    64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool3d(4),
            nn.Flatten(),
        )
        self.proj = nn.Linear(4096, _GEO_EMBED)   # 64ch × 4×4×4

    def forward(self, vox: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(vox))


class SemanticEncoder(nn.Module):
    """(B, M, H, W) grayscale 시퀀스 → (B, _SEM_EMBED)."""
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
# Actor — 연속 액션 (tanh-squashed Gaussian)
# ══════════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """행동: (Δθ, Δφ, Δψ) ∈ [-1,1]^3 (env가 max_rate_*로 스케일)."""

    def __init__(self, img_ch: int = 2, scalar_dim: int = 3,
                 action_dim: int = 3, H: int = 84, W: int = 84,
                 log_std_init: float = -0.5):
        super().__init__()
        self.embed = StateEmbedding(img_ch, scalar_dim, H, W)
        self.mlp   = nn.Sequential(
            nn.Linear(_STATE_DIM, 256), nn.ReLU(),
            nn.Linear(256,        256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        # 상태 무관 학습 파라미터 (PPO 표준 관례)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def _dist(self, vox, img, scalar) -> Normal:
        raw = self.mlp(self.embed(vox, img, scalar))
        # 소프트 유계화 — `_MU_MAX` 주석의 발산 사고 참조. 하드 클램프가 아니라
        # tanh를 쓰는 이유는 경계에서도 기울기가 살아 있어야 복귀가 가능해서다.
        mu  = _MU_MAX * torch.tanh(raw / _MU_MAX)
        std = self.log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX).exp().expand_as(mu)
        return Normal(mu, std)

    @staticmethod
    def _squash(dist: Normal, u: torch.Tensor):
        """u(pre-tanh) → (a, log_prob(a)). tanh 야코비안 보정 포함."""
        a = torch.tanh(u)
        logp = dist.log_prob(u).sum(-1) - torch.log(1.0 - a.pow(2) + _TANH_EPS).sum(-1)
        return a, logp

    def greedy(self, vox, img, scalar) -> torch.Tensor:
        """평가용 결정론적 행동."""
        return torch.tanh(self._dist(vox, img, scalar).mean)

    def sample(self, vox, img, scalar):
        """Returns (a, u, log_prob, entropy).

        `a`는 환경에 넣을 행동, `u`는 buffer에 저장할 pre-tanh 값(모듈 docstring 참조).
        """
        dist = self._dist(vox, img, scalar)
        u = dist.sample()
        a, logp = self._squash(dist, u)
        return a, u, logp, dist.entropy().sum(-1)

    def evaluate(self, vox, img, scalar, u):
        """저장된 pre-tanh `u`에 대한 새 정책의 log_prob/entropy."""
        dist = self._dist(vox, img, scalar)
        _, logp = self._squash(dist, u)
        return logp, dist.entropy().sum(-1)


class Critic(nn.Module):
    """scalar_dim=4: (θ, φ, ψ, curr_coverage) — step_1과 동일하게 critic만 coverage 수신."""
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
# Rollout Buffer — step_1 대비 actions만 long → float(action_dim)
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """vox는 uint8로 저장(메모리 절약), `flat()`에서 float 변환."""

    def __init__(self, T: int, E: int,
                 Nx: int, Ny: int, Nz: int,
                 M: int, H: int, W: int,
                 scalar_dim: int, scalar_dim_critic: int,
                 action_dim: int, device):
        self.T, self.E, self.ptr = T, E, 0
        kw = dict(device=device)

        self.vox               = torch.zeros(T, E, 3, Nx, Ny, Nz, dtype=torch.uint8, **kw)
        self.img               = torch.zeros(T, E, M, H, W,            **kw)
        self.obs_scalar        = torch.zeros(T, E, scalar_dim,          **kw)
        self.obs_scalar_critic = torch.zeros(T, E, scalar_dim_critic,   **kw)
        self.actions           = torch.zeros(T, E, action_dim,          **kw)   # pre-tanh u
        self.logprobs          = torch.zeros(T, E,                      **kw)
        self.rewards           = torch.zeros(T, E,                      **kw)
        self.dones             = torch.zeros(T, E,                      **kw)
        self.values            = torch.zeros(T, E,                      **kw)

    def add(self, vox, img, obs_scalar, obs_scalar_critic,
            action_u, logprob, reward, done, value):
        t = self.ptr
        self.vox              [t] = vox.to(torch.uint8)
        self.img              [t] = img
        self.obs_scalar       [t] = obs_scalar
        self.obs_scalar_critic[t] = obs_scalar_critic
        self.actions          [t] = action_u
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
            "actions":           _r(self.actions),
            "logprobs":          _r(self.logprobs),
            "returns":           _r(self.returns),
            "advantages":        _r(self.advantages),
            "old_values":        _r(self.values),
        }

    def reset(self):
        self.ptr = 0


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    var_ret = returns.var()
    if var_ret < 1e-8:
        return float("nan")
    return (1.0 - (returns - values).var() / var_ret).item()


# ══════════════════════════════════════════════════════════════════════════════
# PPO 업데이트 — step_1 무수정(액션 키 이름만 pose_acts → actions)
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
    acc        = dict(policy_loss=0., value_loss=0., entropy=0., n=0)
    kl_sum, kl_n = 0.0, 0      # 측정 — 갱신 여부와 무관하게 누적
    n_updates  = 0             # 실제로 적용된 gradient step 수
    # value clipping이 실제로 제동을 걸었는지 — 기준점 버그를 고친 뒤에도
    # 정말 작동하는지 **측정**해야 한다. 고쳤으니 되겠지로 넘기지 않는다.
    n_clip_sel = 0
    early_stop = False
    stop_reason = ""

    for _epoch in range(cfg.ppo_epochs):
        perm     = torch.randperm(TE, device=adv.device)
        epoch_kl = 0.0
        epoch_n  = 0

        for s in range(0, TE, cfg.minibatch_size):
            mb = perm[s : s + cfg.minibatch_size]
            if mb.numel() == 0:
                continue

            new_logp, entropy = actor.evaluate(
                data["vox"][mb], data["img"][mb],
                data["obs_scalar"][mb],
                data["actions"][mb],
            )

            # abs(log-ratio): policy가 더 확신하는 방향도 감지 (signed KL은 음수 가능)
            log_ratio = new_logp - data["logprobs"][mb]
            approx_kl_mb = log_ratio.abs().mean().item()

            # KL은 **측정값**이므로 갱신 여부와 무관하게 항상 기록한다.
            # 갱신 결과 통계(acc)와 함께 묶어두면, 첫 미니배치에서 중단됐을 때
            # n=0이 되어 보고값이 전부 0.0000으로 나오고 — 실제로 그렇게 됐다 —
            # 롤아웃이 통째로 버려진 사실이 로그에서 보이지 않는다.
            kl_sum += approx_kl_mb * mb.numel()
            kl_n   += mb.numel()

            # 미니배치 단위 조기 종료 (2026-08-28 추가).
            #
            # 기존에는 **에포크 끝**에서만 검사했다. 롤아웃 2048 / minibatch 32면
            # 미니배치가 64개라, 발산이 에포크 안에서 일어나면 64회 갱신을 다
            # 하고 나서야 알아챈다. 2026-08-27 런이 정확히 그렇게 무너졌다
            # (KL 0.09 → 1,756 → 319,209). 초과 즉시 멈추면 이 창이 닫힌다.
            #
            # 갱신 **이전**에 검사한다 — 이 시점의 KL은 지금까지의 누적 이탈량이고,
            # 이미 과도하면 여기에 갱신을 더 얹을 이유가 없다.
            #
            # 여유 배수 `_MB_KL_SLACK`: 단일 미니배치 KL은 에포크 평균보다
            # 자연히 시끄러우므로 target_kl을 그대로 적용하면 정상 변동에도 걸린다.
            if approx_kl_mb > max(cfg.target_kl * _MB_KL_SLACK, _MB_KL_FLOOR):
                early_stop = True
                stop_reason = f"mb_kl={approx_kl_mb:.3f}@e{_epoch}"
                break

            # exp 전에 클램프 — `_LOG_RATIO_CLAMP` 주석 참조. 클램프가 없으면
            # log_ratio가 커질 때 ratio가 0/inf로 죽고, 그러면 아래 clamp()가
            # 범위 밖이라 기울기까지 0이 되어 학습이 멈춘다.
            ratio  = log_ratio.clamp(-_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP).exp()
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
            # PPO value clipping — 기준점은 `returns`가 아니라 **`old_values`**다.
            #
            # step_1 `algo_UW_NBV.py`에서 `returns + clamp(...)`로 이식돼 있었고
            # step_3도 그대로 물려받았는데, 그러면 두 번째 항이
            # `mean(clamp(·)²) ≤ clip_eps² = 0.04`로 고정되어 실제 MSE(40~65)가
            # 항상 max에 선택된다 = **클리핑이 완전히 무력화**된다.
            #
            # 2026-08-31 검증에서 이 결함이 드러났다. KL 가드를 완화해 롤아웃당
            # 갱신이 64→384회로 늘자 critic이 제동 없이 과갱신되어 explained
            # variance가 +0.46 → **-2.49**로 붕괴했고, advantage가 무의미해져
            # coverage가 0.44 근처에서 정체했다. 갱신이 적던 이전 런에서는
            # 같은 결함이 있어도 critic이 크게 움직이지 않아 드러나지 않았다.
            v_clipped = data["old_values"][mb] + (v - data["old_values"][mb]).clamp(
                -cfg.clip_eps, cfg.clip_eps
            )
            vl_plain   = F.mse_loss(v,         data["returns"][mb])
            vl_clipped = F.mse_loss(v_clipped, data["returns"][mb])
            vl = torch.max(vl_plain, vl_clipped)
            if vl_clipped.item() > vl_plain.item():
                n_clip_sel += 1

            critic_loss = cfg.vf_coef * vl
            optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            optimizer_critic.step()

            B = mb.numel()
            acc["policy_loss"] += pg.item()             * B
            acc["value_loss"]  += vl.item()             * B
            acc["entropy"]     += entropy.mean().item() * B
            acc["n"]           += B
            n_updates          += 1
            epoch_kl           += approx_kl_mb          * B
            epoch_n            += B
        if early_stop:
            break

        if epoch_n > 0 and (epoch_kl / epoch_n) > cfg.target_kl:
            early_stop = True
            stop_reason = stop_reason or f"epoch_kl={epoch_kl/epoch_n:.3f}@e{_epoch}"
            break

    n = max(acc.pop("n"), 1)
    return {k: v / n for k, v in acc.items()} | {
        "approx_kl":  kl_sum / max(kl_n, 1),
        "early_stop": early_stop,
        # n_updates=0이면 그 롤아웃은 통째로 버려진 것 — 반드시 보이게 한다
        "n_updates":  n_updates,
        "stop_reason": stop_reason,
        # 0에 가까우면 클리핑이 여전히 무력하다는 뜻 = 기준점 수정이 무효
        "value_clip_frac": n_clip_sel / max(n_updates, 1),
    }
