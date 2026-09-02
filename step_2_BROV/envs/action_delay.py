"""행동 지연 · 행동 이력 · 관측 신선도 jitter — 순수 torch 모듈.

`DELAY_TRAINING_PLAN.md`(2026-09-02)의 설계 A/B 를 구현한다. 세 부품 모두
Isaac 의존성이 없어서 컨테이너 밖 pytest 로 검증할 수 있고, `BROVVelEnv` 는
이 클래스들을 조립만 한다.

배경 — 왜 필요한가
------------------
실기 수조 세션(2026-09-02)이 배포 진동의 근본원인을 dead time τ = 80 ms +
정책 포화(relay) 로 확정했다. 학습 환경에는 이 지연이 구조적으로 없어서
(τ=0) 정책이 실기 안정 문턱을 넘는 이득을 학습한다. 지연을 학습 환경에
주입하면 MDP 가 깨지므로(같은 관측에서 최적 행동이 "비행 중" 과거 행동에
따라 달라짐 → POMDP), 증강 등가 정리(Katsikopoulos & Engelbrecht 2003)에
따라 최근 실행 행동 d_max 개를 관측에 붙여 MDP 를 복원한다.

세 부품
--------
- :class:`ActionDelayBuffer` — 물리 스텝(10 ms) 해상도 링버퍼. 정책 스텝
  (40 ms) 해상도로는 60 ms 를 표현할 수 없다는 것이 물리 해상도를 쓰는
  이유다. 지연값은 에피소드마다 uniform 랜덤(기본 40~80 ms).
- :class:`ActionHistoryBuffer` — 관측에 붙일 최근 **실행 행동**(탐색 노이즈
  포함, clip 이후, 지연버퍼에 들어간 바로 그 값) N 개.
- :class:`ObservationStalenessModel` — 확률 p 로 직전 스텝 관측을 그대로 다시
  공급(실기 attitude_age 의 15.1% 가 40~50 ms = 1 틱 묵음인 것의 재현).

난수 스트림 격리
-----------------
세 부품 모두 **전용 `torch.Generator`** 를 쓴다. 전역 RNG 스트림을 건드리지
않아야 기능이 꺼진 프로파일(`paper_ref_v1` 등)의 학습 결과가 1 bit 도 변하지
않는다 — 애초에 기능이 꺼져 있으면 인스턴스 자체를 만들지 않지만, 켜진
프로파일에서도 다른 도메인 랜덤화의 난수 소비 순서를 바꾸지 않는 편이 A/B
비교를 깨끗하게 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def _as_env_ids(
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    num_envs: int,
    device: torch.device,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.long, device=device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    """`device` 전용 generator. CUDA 에서도 전역 스트림과 분리된다."""
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


@dataclass(frozen=True)
class ActionDelayConfig:
    """행동 지연 랜덤화 범위 (물리 스텝 해상도)."""

    delay_ms_range: tuple[float, float] = (40.0, 80.0)
    physics_dt_s: float = 0.01

    def __post_init__(self) -> None:
        low, high = self.delay_ms_range
        if not (0.0 <= low <= high):
            raise ValueError("delay_ms_range must be a valid non-negative range")
        if self.physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")

    @property
    def delay_step_range(self) -> tuple[int, int]:
        """[스텝] 물리 스텝 단위 지연 범위 (양끝 포함)."""
        dt_ms = self.physics_dt_s * 1000.0
        low = int(round(self.delay_ms_range[0] / dt_ms))
        high = int(round(self.delay_ms_range[1] / dt_ms))
        return max(0, low), max(0, high)


class ActionDelayBuffer:
    """물리 스텝 해상도 행동 지연 링버퍼.

    :meth:`step` 을 **물리 스텝마다 한 번** 호출한다(정책 스텝마다가 아니다).
    반환값이 이번 물리 스텝에 플랜트에 실제로 인가되는 행동이다.

    에피소드 시작 직후 버퍼가 비어 있는 구간에서는 0(중립)을 돌려준다 —
    "에피소드 시작에 유령 행동이 없어야 한다"는 설계 요구를 그대로 만족한다.
    지연값은 :meth:`reset` 에서 에피소드마다 새로 뽑고 에피소드 내내 고정이다
    (실기 A2 의 r=0.809 가 "대체로 고정 + 약간의 jitter" 를 시사 — 스텝 내
    jitter 는 :class:`ObservationStalenessModel` 이 대신한다).
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        config: ActionDelayConfig | None = None,
    ) -> None:
        if num_envs <= 0 or action_dim <= 0:
            raise ValueError("num_envs and action_dim must be positive")
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config or ActionDelayConfig()
        self._generator = _make_generator(self.device, seed)

        self._min_steps, self._max_steps = self.config.delay_step_range
        # 링버퍼 길이는 최대 지연 + 1 — 방금 쓴 칸을 덮어쓴 뒤에도 d 스텝 전
        # 값이 살아 있어야 한다.
        self._length = self._max_steps + 1
        self._history = torch.zeros(
            self.num_envs, self._length, self.action_dim, device=self.device, dtype=dtype
        )
        self._write_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._delay_steps = torch.full(
            (self.num_envs,), self._max_steps, dtype=torch.long, device=self.device
        )
        self._row = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def delay_steps(self) -> torch.Tensor:
        """(num_envs,) 현재 에피소드의 지연 [물리 스텝]."""
        return self._delay_steps

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        if ids.numel() == 0:
            return
        # 정수 스텝 균등 샘플. 연속 ms 를 뽑아 반올림하면 양끝(4, 8)만 절반
        # 무게가 되므로, 표현 가능한 값 위에서 정확히 균등한 이 방식을 쓴다.
        self._delay_steps[ids] = torch.randint(
            self._min_steps,
            self._max_steps + 1,
            (ids.numel(),),
            generator=self._generator,
            device=self.device,
            dtype=torch.long,
        )
        self._history[ids] = 0.0
        self._write_idx[ids] = 0

    def step(self, action: torch.Tensor) -> torch.Tensor:
        """행동 하나를 넣고 d 물리 스텝 전 행동을 돌려준다."""
        if action.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"action must be ({self.num_envs}, {self.action_dim}), got {tuple(action.shape)}"
            )
        write = self._write_idx % self._length
        self._history[self._row, write] = action.to(dtype=self.dtype)
        read = (self._write_idx - self._delay_steps) % self._length
        # 아직 d 스텝을 채우지 않은 env 는 0(중립) — reset 에서 버퍼를 0 으로
        # 비웠으므로 별도 분기 없이 그대로 만족된다.
        applied = self._history[self._row, read]
        self._write_idx = self._write_idx + 1
        return applied


class ActionHistoryBuffer:
    """관측에 붙일 최근 실행 행동 N 개 (index 0 = 가장 최근).

    :meth:`push` 를 **정책 스텝마다 한 번**, 실행 행동이 확정된 직후에
    호출한다. :meth:`as_observation` 은 `[a_{t-1}, a_{t-2}, …]` 를 이어붙인
    (num_envs, N*action_dim) 텐서를 돌려준다.
    """

    def __init__(
        self,
        num_envs: int,
        length: int,
        action_dim: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if num_envs <= 0 or length <= 0 or action_dim <= 0:
            raise ValueError("num_envs, length and action_dim must be positive")
        self.num_envs = int(num_envs)
        self.length = int(length)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self._history = torch.zeros(
            self.num_envs, self.length, self.action_dim, device=self.device, dtype=dtype
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        if ids.numel() == 0:
            return
        self._history[ids] = 0.0

    def push(self, action: torch.Tensor) -> None:
        if action.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"action must be ({self.num_envs}, {self.action_dim}), got {tuple(action.shape)}"
            )
        if self.length > 1:
            self._history[:, 1:] = self._history[:, :-1].clone()
        self._history[:, 0] = action.to(dtype=self.dtype)

    def as_observation(self) -> torch.Tensor:
        return self._history.reshape(self.num_envs, self.length * self.action_dim)


class ObservationStalenessModel:
    """확률 p 로 직전 스텝 관측을 다시 공급하는 소규모 POMDP 섭동.

    실기 attitude_age 분포(15.1% 가 40~50 ms = 1 틱 묵음)의 재현이 목적이다.
    에피소드 첫 틱은 직전 관측이 없으므로 항상 신선하다. 묵음 틱에서는
    적분(z_v/z_q)도 멈춰야 하므로 :meth:`step` 이 돌려주는 `fresh_mask` 를
    호출측이 integrate 마스크에 AND 로 걸어 쓴다 — 배포측
    (`brov_base/observation.py`) 의 stale/duplicate 표본 처리와 같은 규칙이다.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        probability: float,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ) -> None:
        if num_envs <= 0 or obs_dim <= 0:
            raise ValueError("num_envs and obs_dim must be positive")
        if not (0.0 <= probability <= 1.0):
            raise ValueError("probability must be in [0, 1]")
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.probability = float(probability)
        self.device = torch.device(device)
        self.dtype = dtype
        self._generator = _make_generator(self.device, seed)
        self._held = torch.zeros(self.num_envs, self.obs_dim, device=self.device, dtype=dtype)
        self._has_previous = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        if ids.numel() == 0:
            return
        self._held[ids] = 0.0
        self._has_previous[ids] = False

    def draw_stale_mask(self) -> torch.Tensor:
        """(num_envs,) 이번 틱에 묵은 관측을 공급할 env 마스크."""
        if self.probability <= 0.0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        draw = torch.rand(
            self.num_envs, generator=self._generator, device=self.device, dtype=self.dtype
        )
        return (draw < self.probability) & self._has_previous

    def apply(self, observation: torch.Tensor, stale_mask: torch.Tensor) -> torch.Tensor:
        """마스크에 따라 관측을 교체하고 hold 상태를 갱신한다."""
        if observation.shape != (self.num_envs, self.obs_dim):
            raise ValueError(
                f"observation must be ({self.num_envs}, {self.obs_dim}), "
                f"got {tuple(observation.shape)}"
            )
        published = torch.where(stale_mask.unsqueeze(-1), self._held, observation)
        # 연속 묵음이면 같은 값을 계속 내보낸다(중복 발행) — held 를 published
        # 로 갱신하는 것이 그 규칙이다.
        self._held = published.clone()
        self._has_previous = torch.ones_like(self._has_previous)
        return published


__all__ = [
    "ActionDelayConfig",
    "ActionDelayBuffer",
    "ActionHistoryBuffer",
    "ObservationStalenessModel",
]
