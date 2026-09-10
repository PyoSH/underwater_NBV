"""NBV 베이스라인 정책 — Isaac 평가와 Gazebo 배포 루프가 **같은 코드**를 쓰게 한다.

`eval_core.Policy` 에서 뽑아냈다(2026-09-10). 배포 루프가 따로 구현하면 두 곳의
행동이 미세하게 갈라져 "같은 결정 수 random 대비" 라는 판정 기준이 무의미해진다.
학습 정책은 `algorithm.algo_nbv_continuous.Actor` 가 필요하지만, 베이스라인은
torch 만 있으면 되므로 ROS 컨테이너에서도 돈다.

행동은 전부 정규화 액션 (delta_theta, delta_phi, delta_psi) in [-1, 1] 이다.
"""
from __future__ import annotations

import torch

KINDS = ("hold", "approach", "sweep", "orbit", "random")


class BaselinePolicy:
    """상태를 가진 베이스라인(sweep 은 결정 카운터를 쓴다)."""

    def __init__(self, kind: str, *, seed: int = 0, device=None):
        if kind not in KINDS:
            raise ValueError(f"kind 는 {KINDS} 중 하나여야 한다 (받은 값 {kind!r})")
        self.kind = kind
        self._device = device or torch.device("cpu")
        self._gen = torch.Generator(device=self._device).manual_seed(seed)
        self._sweep_t = 0

    def reset(self) -> None:
        self._sweep_t = 0

    def act(self, n_env: int, a_dim: int = 3) -> torch.Tensor:
        if self.kind == "hold":
            return torch.zeros(n_env, a_dim, device=self._device)

        if self.kind == "approach":
            # psi 를 하한까지 밀어붙이고 고착 — (A) 정규화가 새는지 보는 회귀 지표.
            a = torch.zeros(n_env, a_dim, device=self._device)
            a[:, 2] = -1.0
            return a

        if self.kind == "sweep":
            # 결정론적 격자 훑기: 12결정에 방위 한 바퀴, 바퀴마다 phi 30 도 이동,
            # 4바퀴 뒤 psi +0.5 m. 관측 가능 표면 측정용(계획 §11.7-4).
            t = self._sweep_t
            self._sweep_t += 1
            a = torch.zeros(n_env, a_dim, device=self._device)
            a[:, 0] = 1.0
            ring, phase = t // 12, t % 12
            if phase == 11:
                a[:, 1] = -1.0 if ring < 4 else 1.0
            if t == 48:
                a[:, 2] = 1.0
            return a

        if self.kind == "orbit":
            a = torch.zeros(n_env, a_dim, device=self._device)
            a[:, 0] = 1.0
            return a

        return torch.rand((n_env, a_dim), generator=self._gen,
                          device=self._device) * 2.0 - 1.0


def step_spherical(theta, phi, psi, action, *, max_rate_theta, max_rate_phi,
                   max_rate_psi, phi_min, phi_max, psi_min, psi_max):
    """액션 -> 다음 (theta, phi, psi). `env.py::_pre_physics_step` 와 동일.

    theta 는 감고(wrap), phi/psi 는 clamp 한다. 배포의 geofence 사영은 **이 다음에**
    적용한다 — 정책이 고른 방향(theta, phi)을 보존하고 psi 만 밀기 위해서다.
    """
    a = action.clamp(-1.0, 1.0)
    two_pi = 2.0 * torch.pi
    theta = (theta + a[:, 0] * max_rate_theta) % two_pi
    phi = (phi + a[:, 1] * max_rate_phi).clamp(phi_min, phi_max)
    psi = (psi + a[:, 2] * max_rate_psi).clamp(psi_min, psi_max)
    return theta, phi, psi
