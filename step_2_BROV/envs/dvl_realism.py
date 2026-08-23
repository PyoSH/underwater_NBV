"""Pure-torch DVL sensor realism model for training-time domain randomization.

Simulates the Water Linked A50-class DVL characteristics already measured and
used in the Gazebo Stage-2 harness (``stage2_sitl_dvl_injector.py`` defaults:
delay~0.10s, noise_std~0.003 m/s, rate 5-15 Hz depending on altitude) so the
training-time observation ``v_e_b`` experiences the same hold/noise/delay
statistics deployment does, instead of a perfect, instantaneous PhysX
ground-truth reading every policy tick.

Per Sim2Swim's own 16-D observation split, this only touches the
velocity-derived components (``v_e_b``, and therefore ``z_v``); ``q_e``/``ω_b``
stay on the (comparatively fast, low-latency) IMU/AHRS path and are
unaffected by this module.

Design notes (see project_step2_brov_retrain_spec, memory):

- Delay is applied to the *measured velocity itself*, not to ``v_e_b``. The
  desired velocity ``v_d^b`` is generated onboard in real time with no delay
  (it comes from guidance, not a sensor); only the DVL measurement lags. This
  matches ``brov_base/observation.py``'s real deployment structure exactly.
- Hold/rate gates *when a fresh measurement exists*; between DVL ticks the
  measurement (and therefore ``v_e_b``) is piecewise constant. Combined with
  ``build_velocity_observation``'s ``integrate_velocity`` mask, the caller can
  make ``z_v`` advance only on fresh samples -- the real system integrates
  "only for new telemetry samples," not every policy tick.
- All three parameters (rate, noise, delay) are randomized once per episode
  (sampled at ``reset``), not fixed constants, so the policy generalizes
  across the plausible range rather than overfitting one exact number.
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


@dataclass(frozen=True)
class DVLRealismConfig:
    """Randomization ranges for one simulated Water Linked A50-class DVL."""

    rate_hz_range: tuple[float, float] = (5.0, 15.0)
    noise_std_range_mps: tuple[float, float] = (0.0, 0.006)
    delay_s_range: tuple[float, float] = (0.0, 0.15)
    policy_dt_s: float = 0.04

    def __post_init__(self) -> None:
        low_r, high_r = self.rate_hz_range
        low_n, high_n = self.noise_std_range_mps
        low_d, high_d = self.delay_s_range
        if not (0.0 < low_r <= high_r):
            raise ValueError("rate_hz_range must be a valid positive range")
        if not (0.0 <= low_n <= high_n):
            raise ValueError("noise_std_range_mps must be a valid non-negative range")
        if not (0.0 <= low_d <= high_d):
            raise ValueError("delay_s_range must be a valid non-negative range")
        if self.policy_dt_s <= 0.0:
            raise ValueError("policy_dt_s must be positive")


class DVLRealismModel:
    """Per-environment sample-and-hold + Gaussian noise + fixed delay on v^b.

    Call :meth:`reset` for environments whose RL episode reset (this samples
    fresh per-episode rate/noise/delay parameters and clears history), then
    call :meth:`step` once per policy tick with the current true body
    velocity. The returned ``fresh_sample_mask`` tells the caller which
    environments actually received a new "DVL reading" this tick, for gating
    ``z_v``'s integration.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        config: DVLRealismConfig | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config or DVLRealismConfig()
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

        max_delay_s = self.config.delay_s_range[1]
        self._max_delay_steps = max(
            1, int(round(max_delay_s / self.config.policy_dt_s)) + 1
        )
        self._history = torch.zeros(
            self.num_envs, self._max_delay_steps, 3, device=self.device, dtype=dtype
        )
        self._write_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._rate_hz = torch.full(
            (self.num_envs,), self.config.rate_hz_range[0], device=self.device, dtype=dtype
        )
        self._noise_std = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._next_update_s = torch.zeros(self.num_envs, device=self.device, dtype=dtype)
        self._held_measurement = torch.zeros(self.num_envs, 3, device=self.device, dtype=dtype)
        self._is_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        count = ids.numel()
        if count == 0:
            return

        low_r, high_r = self.config.rate_hz_range
        rate = low_r + (high_r - low_r) * torch.rand(count, generator=self._generator)
        low_n, high_n = self.config.noise_std_range_mps
        noise = low_n + (high_n - low_n) * torch.rand(count, generator=self._generator)
        low_d, high_d = self.config.delay_s_range
        delay_s = low_d + (high_d - low_d) * torch.rand(count, generator=self._generator)
        delay_steps = (
            (delay_s / self.config.policy_dt_s)
            .round()
            .long()
            .clamp(0, self._max_delay_steps - 1)
        )

        self._rate_hz[ids] = rate.to(device=self.device, dtype=self.dtype)
        self._noise_std[ids] = noise.to(device=self.device, dtype=self.dtype)
        self._delay_steps[ids] = delay_steps.to(self.device)
        self._next_update_s[ids] = 0.0
        self._history[ids] = 0.0
        self._write_idx[ids] = 0
        self._held_measurement[ids] = 0.0
        self._is_initialized[ids] = True

    def step(
        self, true_velocity_body: torch.Tensor, episode_time_s: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one policy tick.

        Parameters
        ----------
        true_velocity_body : (num_envs, 3) ground-truth body-frame velocity.
        episode_time_s      : (num_envs,) elapsed time since this env's last
            reset -- callers already track this for the desired-state
            scheduler (``episode_length_buf * policy_dt``); reuse it here so
            the DVL clock and the command-schedule clock agree.

        Returns
        -------
        measured_velocity : (num_envs, 3) delayed, held, noised velocity.
        fresh_sample_mask  : (num_envs,) True where a new DVL reading landed
            this tick -- gate ``z_v`` integration on this.
        """

        idx = self._write_idx % self._max_delay_steps
        row = torch.arange(self.num_envs, device=self.device)
        self._history[row, idx] = true_velocity_body
        read_idx = (self._write_idx - self._delay_steps) % self._max_delay_steps
        delayed = self._history[row, read_idx]
        self._write_idx = self._write_idx + 1

        fresh_mask = episode_time_s >= self._next_update_s
        noise = torch.randn(
            self.num_envs, 3, generator=None, device=self.device, dtype=self.dtype
        ) * self._noise_std.unsqueeze(-1)
        candidate = delayed + noise
        self._held_measurement = torch.where(
            fresh_mask.unsqueeze(-1), candidate, self._held_measurement
        )
        period = 1.0 / self._rate_hz.clamp_min(1.0e-3)
        self._next_update_s = torch.where(
            fresh_mask, self._next_update_s + period, self._next_update_s
        )
        return self._held_measurement, fresh_mask


__all__ = ["DVLRealismConfig", "DVLRealismModel"]
