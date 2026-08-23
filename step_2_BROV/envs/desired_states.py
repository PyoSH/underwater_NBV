"""Batched desired-state generators for BROV velocity-policy training.

This module deliberately depends only on :mod:`torch`.  It can therefore be
unit-tested without starting Isaac Sim and can be shared by training and
validation code without importing an environment.

Two contracts are provided:

``PaperReferenceBatch``
    Reproduces the desired states described in Sim2Swim, arXiv:2512.08656v2,
    section 4.5.  Each environment receives one body-frame velocity sampled
    uniformly on S² with norm exactly 0.5 m/s.  The desired attitude starts at
    a uniformly sampled orientation and evolves with the analytic
    Frenet--Serret frame of Eq. (9).

``DeployV2Scheduler``
    A bounded five-second command curriculum for deployment-oriented smoke
    training.  Every environment has exactly one command transition in
    [2, 3) seconds.  The post-transition command bins (hold, 0.1, and 0.5 m/s)
    are balanced across the reset batch.  Exact non-zero reversals can be
    enabled without changing that balance.

``DeployV3Scheduler``
    A mission-scale, multi-leg curriculum.  Each environment gets a sequence
    of legs spanning the full (tens-of-seconds) episode, each with an
    independently sampled body-velocity command (direction on S^2, speed from
    the hold/0.1/0.5 m/s bins) and a coin-flip attitude retarget.  This lets
    ``z_v``/``z_q`` experience realistic multi-leg accumulation -- and, when
    the caller resets them on ``transition_mask``, realistic per-leg
    reset-then-accumulate dynamics -- instead of only ever seeing the first
    few seconds of a single 5 s window.  See
    ``project_step2_brov_retrain_spec`` (project memory) for why this
    generalization was chosen over a literal replay of one field mission.

Quaternion convention is ``[w, x, y, z]`` throughout.  All public outputs are
batched tensors resident on the configured device.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch


def _as_env_ids(
    env_ids: Sequence[int] | torch.Tensor | None,
    *,
    num_envs: int,
    device: torch.device,
    validate: bool = True,
) -> torch.Tensor:
    """Normalize an optional environment selection to a 1-D long tensor."""

    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.long, device=device)
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=device).reshape(-1)
    if validate:
        # Validation is used only by reset/configuration paths.  sample() sets
        # validate=False to avoid device-to-host synchronization in the hot
        # policy loop.
        if ids.numel() and bool(((ids < 0) | (ids >= num_envs)).any()):
            raise IndexError(f"environment id outside [0, {num_envs})")
        if ids.unique().numel() != ids.numel():
            raise ValueError("env_ids must not contain duplicates")
    return ids


def _normalize(vector: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector / norm.clamp_min(eps)


def _quat_normalize(quaternion: torch.Tensor) -> torch.Tensor:
    return _normalize(quaternion)


def _quat_unique(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the scalar-nonnegative representative of each unit quaternion."""

    return torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _heading_from_direction(direction_w: torch.Tensor) -> torch.Tensor:
    """Pure-torch mirror of ``guidance/los_guidance.py::_heading_from_direction``.

    Reproduces the roll=0 specialization of
    ``isaaclab.utils.math.quat_from_euler_xyz`` in closed form so this module
    keeps its "no isaaclab import" property. Must stay numerically identical
    to the real guidance module's function -- see
    ``tests/test_desired_states_los_heading_parity.py``.
    """

    d = _normalize(direction_w)
    yaw = torch.atan2(d[..., 1], d[..., 0])
    pitch = torch.asin(d[..., 2].clamp(-1.0, 1.0))
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    return torch.stack((cy * cp, -sy * sp, cy * sp, sy * cp), dim=-1)


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Hamilton product for broadcast-compatible wxyz quaternions."""

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert proper rotation matrices to scalar-nonnegative wxyz quaternions.

    The desired-state path uses this only for the relative Frenet--Serret
    rotation over one five-second episode.  The four-candidate implementation
    remains well-conditioned if a future trajectory approaches a half turn.
    """

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {tuple(matrix.shape)}")

    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    # Each row is a quaternion candidate multiplied by twice the magnitude of
    # its dominant component.  Selecting the largest denominator avoids the
    # numerical singularity of a trace-only conversion.
    q_abs = torch.sqrt(
        torch.clamp_min(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    denominator = (2.0 * q_abs).clamp_min(torch.finfo(matrix.dtype).eps)
    candidates = candidates / denominator.unsqueeze(-1)
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(*best.shape, 1, 4)
    quaternion = candidates.gather(dim=-2, index=gather_index).squeeze(-2)
    return _quat_unique(_quat_normalize(quaternion))


def _sample_uniform_sphere(
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample ``count`` directions uniformly on S² using a local CPU RNG."""

    if count == 0:
        return torch.empty((0, 3), device=device, dtype=dtype)
    # Sampling on CPU makes a fixed seed reproducible across CPU/CUDA devices;
    # reset-time transfer is negligible compared with simulation rollout.
    direction = torch.randn((count, 3), generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    return direction.to(device=device, dtype=dtype)


def _sample_uniform_quaternion(
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample uniformly from SO(3) through normalized Gaussian S³ samples."""

    if count == 0:
        return torch.empty((0, 4), device=device, dtype=dtype)
    quaternion = torch.randn((count, 4), generator=generator, dtype=torch.float64)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    quaternion = _quat_unique(quaternion)
    return quaternion.to(device=device, dtype=dtype)


def _broadcast_time(
    time_s: float | torch.Tensor,
    *,
    count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    time = torch.as_tensor(time_s, device=device, dtype=dtype)
    if time.ndim == 0:
        return time.expand(count)
    time = time.reshape(-1)
    if time.numel() != count:
        raise ValueError(f"time_s has {time.numel()} values, expected {count}")
    return time


def _frenet_frame(
    time_s: torch.Tensor,
    *,
    coefficients: tuple[float, float, float],
    omega_rad_s: float,
) -> torch.Tensor:
    """Analytic Frenet--Serret frame of the published Eq. (9) curve."""

    a, b, c = coefficients
    phase = omega_rad_s * time_s
    velocity = torch.stack(
        (
            torch.full_like(time_s, a),
            b * torch.sin(phase),
            c * torch.cos(phase),
        ),
        dim=-1,
    )
    acceleration = torch.stack(
        (
            torch.zeros_like(time_s),
            b * omega_rad_s * torch.cos(phase),
            -c * omega_rad_s * torch.sin(phase),
        ),
        dim=-1,
    )
    tangent = _normalize(velocity)
    normal_component = acceleration - (acceleration * tangent).sum(-1, keepdim=True) * tangent
    normal = _normalize(normal_component)
    binormal = _normalize(torch.cross(tangent, normal, dim=-1))
    # Recompute N to suppress the small loss of orthogonality from float32.
    normal = torch.cross(binormal, tangent, dim=-1)
    return torch.stack((tangent, normal, binormal), dim=-1)


def _frenet_relative_quaternion(
    time_s: torch.Tensor,
    *,
    coefficients: tuple[float, float, float],
    omega_rad_s: float,
) -> torch.Tensor:
    frame_zero = _frenet_frame(
        torch.zeros_like(time_s),
        coefficients=coefficients,
        omega_rad_s=omega_rad_s,
    )
    frame_now = _frenet_frame(
        time_s,
        coefficients=coefficients,
        omega_rad_s=omega_rad_s,
    )
    return _matrix_to_quaternion(frame_zero.transpose(-1, -2) @ frame_now)


@dataclass(frozen=True)
class PaperReferenceConfig:
    """Published Sim2Swim desired-state constants."""

    speed_mps: float = 0.5
    trajectory_coefficients: tuple[float, float, float] = (0.5, 0.5, 0.3)
    trajectory_omega_rad_s: float = 0.2
    episode_length_s: float = 5.0

    def __post_init__(self) -> None:
        if self.speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive")
        if self.trajectory_omega_rad_s <= 0.0:
            raise ValueError("trajectory_omega_rad_s must be positive")
        if self.episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be positive")
        if len(self.trajectory_coefficients) != 3:
            raise ValueError("trajectory_coefficients must contain [a, b, c]")


class PaperReferenceBatch:
    """Per-environment paper-reference state with deterministic local sampling.

    Call :meth:`reset` for the environments whose RL episodes reset, then call
    :meth:`sample` with their episode times.  A scalar time samples every
    selected environment at the same instant; a vector supplies independent
    episode times.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        config: PaperReferenceConfig | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config or PaperReferenceConfig()
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

        self.velocity_body = torch.zeros((self.num_envs, 3), device=self.device, dtype=dtype)
        self.initial_quaternion = torch.zeros((self.num_envs, 4), device=self.device, dtype=dtype)
        self.initial_quaternion[:, 0] = 1.0
        self._is_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        initial_quaternion: torch.Tensor | None = None,
    ) -> None:
        """Sample new per-episode velocity and initial desired attitude."""

        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        count = ids.numel()
        direction = _sample_uniform_sphere(
            count,
            generator=self._generator,
            device=self.device,
            dtype=self.dtype,
        )
        # Normalize after conversion too, so the configured norm is preserved
        # to floating-point precision even when training in float32.
        self.velocity_body[ids] = _normalize(direction) * self.config.speed_mps

        if initial_quaternion is None:
            quaternion = _sample_uniform_quaternion(
                count,
                generator=self._generator,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            quaternion = torch.as_tensor(initial_quaternion, device=self.device, dtype=self.dtype)
            if quaternion.shape == (4,) and count == 1:
                quaternion = quaternion.unsqueeze(0)
            if quaternion.shape != (count, 4):
                raise ValueError(
                    f"initial_quaternion shape {tuple(quaternion.shape)}, expected {(count, 4)}"
                )
            if bool((torch.linalg.vector_norm(quaternion, dim=-1) <= 1.0e-12).any()):
                raise ValueError("initial_quaternion must be non-zero")
            quaternion = _quat_unique(_quat_normalize(quaternion))
        self.initial_quaternion[ids] = quaternion
        self._is_initialized[ids] = True

    def frenet_frame(self, time_s: float | torch.Tensor) -> torch.Tensor:
        """Return analytic Eq. (9) Frenet--Serret frames as rotation matrices.

        Matrix columns are tangent, principal normal, and binormal.  With the
        published non-zero coefficients the curve has non-zero speed and
        curvature throughout the five-second episode.
        """

        time = torch.as_tensor(time_s, device=self.device, dtype=self.dtype)
        return _frenet_frame(
            time,
            coefficients=self.config.trajectory_coefficients,
            omega_rad_s=self.config.trajectory_omega_rad_s,
        )

    def sample(
        self,
        time_s: float | torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(desired_body_velocity, desired_quaternion_wxyz)``."""

        ids = _as_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self.device,
            validate=False,
        )
        time = _broadcast_time(
            time_s,
            count=ids.numel(),
            device=self.device,
            dtype=self.dtype,
        )
        relative_quaternion = _frenet_relative_quaternion(
            time,
            coefficients=self.config.trajectory_coefficients,
            omega_rad_s=self.config.trajectory_omega_rad_s,
        )
        desired_quaternion = _quat_normalize(
            _quat_mul(self.initial_quaternion[ids], relative_quaternion)
        )
        # The relative rotation remains under a half turn over the published
        # five-second episode, so its scalar-positive representation keeps
        # q_d(t) in the initial quaternion's hemisphere without discontinuity.
        return self.velocity_body[ids].clone(), desired_quaternion


@dataclass(frozen=True)
class DeployV2Config:
    """Five-second, single-transition deployment curriculum."""

    episode_length_s: float = 5.0
    transition_time_range_s: tuple[float, float] = (2.0, 3.0)
    speed_bins_mps: tuple[float, float, float] = (0.0, 0.1, 0.5)
    exact_reversal: bool = True
    policy_dt_s: float = 0.04
    trajectory_coefficients: tuple[float, float, float] = (0.5, 0.5, 0.3)
    trajectory_omega_rad_s: float = 0.2

    def __post_init__(self) -> None:
        low, high = self.transition_time_range_s
        if self.episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be positive")
        if not (0.0 < low < high < self.episode_length_s):
            raise ValueError("transition range must lie strictly inside the episode")
        if tuple(self.speed_bins_mps) != (0.0, 0.1, 0.5):
            raise ValueError("deploy_v2 speed bins are fixed to hold/0.1/0.5 m/s")
        if self.policy_dt_s <= 0.0:
            raise ValueError("policy_dt_s must be positive")
        if self.trajectory_omega_rad_s <= 0.0:
            raise ValueError("trajectory_omega_rad_s must be positive")
        if len(self.trajectory_coefficients) != 3:
            raise ValueError("trajectory_coefficients must contain [a, b, c]")


class DeployVelocityMode(IntEnum):
    """Meaning of the post-transition velocity command."""

    HOLD = 0
    LOW_0P1 = 1
    CRUISE_0P5 = 2


class DeployAttitudeMode(IntEnum):
    """Attitude reference family assigned to one deployment episode."""

    FRENET_CONTINUOUS = 0
    RUNTIME_HOLD = 1
    RUNTIME_YAW_180_STEP = 2


class DeployTransitionMode(IntEnum):
    """Velocity transition family used by one deployment episode."""

    STOP = 0
    RESTART = 1
    REVERSAL = 2


@dataclass(frozen=True)
class DeployV2Sample:
    """One batched scheduler sample.

    ``transition_mask`` is true only on the policy tick which crosses the
    per-environment transition time.  ``after_transition`` remains true after
    that edge.  Mode tensors contain the integer values of
    :class:`DeployVelocityMode` and :class:`DeployAttitudeMode` for cheap
    reward conditioning and machine-readable logging.
    """

    velocity_body: torch.Tensor
    desired_quaternion: torch.Tensor
    transition_mask: torch.Tensor
    after_transition: torch.Tensor
    velocity_mode: torch.Tensor
    attitude_mode: torch.Tensor
    transition_mode: torch.Tensor
    reversal_mask: torch.Tensor


class DeployV2Scheduler:
    """Deterministic batched 5 s command scheduler with one transition.

    The three *post-transition* speed bins are assigned round-robin and then
    shuffled, so their counts differ by at most one for arbitrary batch sizes.
    The hold bin always supplies stop transitions.  When ``exact_reversal`` is
    enabled (the deploy-v2 default), each non-zero speed bin is split as evenly
    as possible between restart ``0 -> v`` and exact reversal ``-v -> v``.
    Reversal episodes receive an exact body-yaw 180-degree attitude step at the
    same instant.  Remaining episodes are balanced between continuous paper
    Frenet--Serret attitude and a runtime-style constant attitude.  Thus one
    reset batch contains stop, restart, reversal, continuous attitude, and
    runtime attitude-step coverage without extending the five-second episode.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        config: DeployV2Config | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config or DeployV2Config()
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

        self.command_before = torch.zeros((num_envs, 3), device=self.device, dtype=dtype)
        self.command_after = torch.zeros((num_envs, 3), device=self.device, dtype=dtype)
        self.transition_time_s = torch.zeros(num_envs, device=self.device, dtype=dtype)
        self.post_speed_bin = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.attitude_mode = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.transition_mode = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.is_reversal = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.initial_quaternion = torch.zeros((num_envs, 4), device=self.device, dtype=dtype)
        self.initial_quaternion[:, 0] = 1.0
        self.step_quaternion = torch.zeros((num_envs, 4), device=self.device, dtype=dtype)
        self.step_quaternion[:, 0] = 1.0
        self._is_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Create one deterministic single-transition schedule per environment."""

        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        count = ids.numel()
        if count == 0:
            return

        # Round-robin category construction makes the reset batch balanced;
        # permutation removes a category-to-environment-index correlation.
        categories_cpu = torch.arange(count, dtype=torch.long) % 3
        permutation = torch.randperm(count, generator=self._generator)
        categories_cpu = categories_cpu[permutation]
        categories = categories_cpu.to(self.device)

        reversal_cpu = torch.zeros(count, dtype=torch.bool)
        if self.config.exact_reversal:
            for category in (int(DeployVelocityMode.LOW_0P1), int(DeployVelocityMode.CRUISE_0P5)):
                members = torch.nonzero(categories_cpu == category, as_tuple=False).flatten()
                member_order = members[
                    torch.randperm(members.numel(), generator=self._generator)
                ]
                reversal_cpu[member_order[::2]] = True

        transition_mode_cpu = torch.full(
            (count,), int(DeployTransitionMode.RESTART), dtype=torch.long
        )
        transition_mode_cpu[categories_cpu == int(DeployVelocityMode.HOLD)] = int(
            DeployTransitionMode.STOP
        )
        transition_mode_cpu[reversal_cpu] = int(DeployTransitionMode.REVERSAL)

        if self.config.exact_reversal:
            attitude_categories_cpu = torch.full(
                (count,), int(DeployAttitudeMode.RUNTIME_YAW_180_STEP), dtype=torch.long
            )
            remaining = torch.nonzero(~reversal_cpu, as_tuple=False).flatten()
            remaining_modes = torch.arange(remaining.numel(), dtype=torch.long) % 2
            remaining_modes = remaining_modes[
                torch.randperm(remaining.numel(), generator=self._generator)
            ]
            attitude_categories_cpu[remaining] = remaining_modes
        else:
            attitude_categories_cpu = torch.arange(count, dtype=torch.long) % 3
            attitude_categories_cpu = attitude_categories_cpu[
                torch.randperm(count, generator=self._generator)
            ]
        attitude_categories = attitude_categories_cpu.to(self.device)
        reversal = reversal_cpu.to(self.device)
        transition_mode = transition_mode_cpu.to(self.device)

        direction = _sample_uniform_sphere(
            count,
            generator=self._generator,
            device=self.device,
            dtype=self.dtype,
        )
        speed_values = torch.tensor(
            self.config.speed_bins_mps,
            device=self.device,
            dtype=self.dtype,
        )
        post_speed = speed_values[categories]
        after = direction * post_speed.unsqueeze(-1)

        # Hold-bin episodes always stop from a real command.  Alternate their
        # incoming speed between 0.1 and 0.5 without consuming global RNG.
        hold = categories == 0
        hold_rank = torch.cumsum(hold.to(torch.long), dim=0) - 1
        hold_speed = torch.where(
            (hold_rank % 2) == 0,
            torch.full_like(post_speed, 0.1),
            torch.full_like(post_speed, 0.5),
        )
        before = torch.zeros_like(after)
        before[hold] = direction[hold] * hold_speed[hold].unsqueeze(-1)

        before[reversal] = -after[reversal]
        # Other non-zero bins implement hold -> commanded speed.

        initial_quaternion = _sample_uniform_quaternion(
            count,
            generator=self._generator,
            device=self.device,
            dtype=self.dtype,
        )
        yaw_180 = torch.zeros((count, 4), device=self.device, dtype=self.dtype)
        yaw_180[:, 3] = 1.0
        step_quaternion = _quat_normalize(_quat_mul(initial_quaternion, yaw_180))

        low, high = self.config.transition_time_range_s
        transition_cpu = low + (high - low) * torch.rand(
            count,
            generator=self._generator,
            dtype=torch.float64,
        )

        self.command_before[ids] = before
        self.command_after[ids] = after
        self.transition_time_s[ids] = transition_cpu.to(self.device, self.dtype)
        self.post_speed_bin[ids] = categories
        self.attitude_mode[ids] = attitude_categories
        self.transition_mode[ids] = transition_mode
        self.is_reversal[ids] = reversal
        self.initial_quaternion[ids] = initial_quaternion
        self.step_quaternion[ids] = step_quaternion
        self._is_initialized[ids] = True

    def sample(
        self,
        time_s: float | torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        previous_time_s: float | torch.Tensor | None = None,
    ) -> DeployV2Sample:
        """Return velocity, attitude, transition edge, and mode metadata.

        If ``previous_time_s`` is omitted, a regular policy tick is inferred as
        ``time_s - config.policy_dt_s``.  Supplying it explicitly is preferred
        when a caller has irregular timing or samples the scheduler sparsely.
        """

        ids = _as_env_ids(
            env_ids,
            num_envs=self.num_envs,
            device=self.device,
            validate=False,
        )
        time = _broadcast_time(
            time_s,
            count=ids.numel(),
            device=self.device,
            dtype=self.dtype,
        )
        if previous_time_s is None:
            previous_time = (time - self.config.policy_dt_s).clamp_min(0.0)
        else:
            previous_time = _broadcast_time(
                previous_time_s,
                count=ids.numel(),
                device=self.device,
                dtype=self.dtype,
            )
        transition_time = self.transition_time_s[ids]
        after_transition = time >= transition_time
        transition_mask = (previous_time < transition_time) & after_transition
        velocity = torch.where(
            after_transition.unsqueeze(-1),
            self.command_after[ids],
            self.command_before[ids],
        )

        mode = self.attitude_mode[ids]
        initial_quaternion = self.initial_quaternion[ids]
        step_quaternion = torch.where(
            after_transition.unsqueeze(-1),
            self.step_quaternion[ids],
            initial_quaternion,
        )
        relative_frenet = _frenet_relative_quaternion(
            time,
            coefficients=self.config.trajectory_coefficients,
            omega_rad_s=self.config.trajectory_omega_rad_s,
        )
        frenet_quaternion = _quat_normalize(_quat_mul(initial_quaternion, relative_frenet))
        desired_quaternion = torch.where(
            (mode == int(DeployAttitudeMode.FRENET_CONTINUOUS)).unsqueeze(-1),
            frenet_quaternion,
            torch.where(
                (mode == int(DeployAttitudeMode.RUNTIME_YAW_180_STEP)).unsqueeze(-1),
                step_quaternion,
                initial_quaternion,
            ),
        )

        return DeployV2Sample(
            velocity_body=velocity,
            desired_quaternion=desired_quaternion,
            transition_mask=transition_mask,
            after_transition=after_transition,
            velocity_mode=self.post_speed_bin[ids].clone(),
            attitude_mode=mode.clone(),
            transition_mode=self.transition_mode[ids].clone(),
            reversal_mask=self.is_reversal[ids].clone(),
        )


@dataclass(frozen=True)
class DeployV3Config:
    """Mission-scale, multi-leg deployment curriculum.

    Generalizes :class:`DeployV2Config` from one 5 s episode with a single
    transition to a much longer episode built from a sequence of legs, each
    with its own randomly sampled duration and command.  ``max_legs`` is a
    static upper bound on how many legs one episode can hold; it must be
    generous enough that the cumulative leg durations always exceed
    ``episode_length_s`` (validated below using the shortest possible leg).
    """

    episode_length_s: float = 30.0
    leg_duration_range_s: tuple[float, float] = (3.0, 8.0)
    speed_bins_mps: tuple[float, float, float] = (0.0, 0.1, 0.5)
    new_attitude_probability: float = 0.5
    policy_dt_s: float = 0.04
    trajectory_coefficients: tuple[float, float, float] = (0.5, 0.5, 0.3)
    trajectory_omega_rad_s: float = 0.2
    max_legs: int = 48

    def __post_init__(self) -> None:
        low, high = self.leg_duration_range_s
        if self.episode_length_s <= 0.0:
            raise ValueError("episode_length_s must be positive")
        if not (0.0 < low <= high):
            raise ValueError("leg_duration_range_s must be a valid positive range")
        if tuple(self.speed_bins_mps) != (0.0, 0.1, 0.5):
            raise ValueError("deploy_v3 speed bins are fixed to hold/0.1/0.5 m/s")
        if not (0.0 <= self.new_attitude_probability <= 1.0):
            raise ValueError("new_attitude_probability must be in [0, 1]")
        if self.policy_dt_s <= 0.0:
            raise ValueError("policy_dt_s must be positive")
        if self.trajectory_omega_rad_s <= 0.0:
            raise ValueError("trajectory_omega_rad_s must be positive")
        if len(self.trajectory_coefficients) != 3:
            raise ValueError("trajectory_coefficients must contain [a, b, c]")
        if self.max_legs < 2:
            raise ValueError("max_legs must allow at least one transition")
        # Worst case (most legs) happens when every leg takes the shortest
        # duration in the sampled range.
        required_legs = int(self.episode_length_s / low) + 2
        if self.max_legs < required_legs:
            raise ValueError(
                f"max_legs={self.max_legs} is too small for episode_length_s="
                f"{self.episode_length_s} with minimum leg duration {low}s "
                f"(need >= {required_legs})"
            )


@dataclass(frozen=True)
class DeployV3Sample:
    """One batched :meth:`DeployV3Scheduler.sample` result.

    Field names deliberately match :class:`DeployV2Sample` so
    ``BROVVelEnv._current_v_d_b`` needs no branching on scheduler type.
    ``transition_mode`` is repurposed to carry the active leg index (a
    diagnostically useful long tensor); nothing in the environment branches
    on its value, only ``DeployV2Scheduler`` gives it enum meaning.
    """

    velocity_body: torch.Tensor
    desired_quaternion: torch.Tensor
    transition_mask: torch.Tensor
    transition_mode: torch.Tensor
    reversal_mask: torch.Tensor
    active_leg: torch.Tensor


class DeployV3Scheduler:
    """Deterministic batched multi-leg command scheduler spanning one episode.

    Precomputes, per environment, a fixed-size (``max_legs``) table of leg
    start times, body-velocity commands, and attitude targets at
    :meth:`reset`.  :meth:`sample` looks up each environment's currently
    active leg (fully vectorized -- no per-environment Python loop) and
    reports a ``transition_mask`` wherever the active leg index just changed,
    so a caller can reset an implicit integral state (``z_v``/``z_q``) at
    exactly those ticks -- the training-side mirror of a real deployment
    resetting on every waypoint transition.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        config: DeployV3Config | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if not dtype.is_floating_point:
            raise TypeError("dtype must be floating point")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype
        self.config = config or DeployV3Config()
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

        k = self.config.max_legs
        self.leg_starts = torch.zeros((num_envs, k), device=self.device, dtype=dtype)
        self.leg_velocity = torch.zeros((num_envs, k, 3), device=self.device, dtype=dtype)
        self.leg_quaternion = torch.zeros((num_envs, k, 4), device=self.device, dtype=dtype)
        self.leg_quaternion[..., 0] = 1.0
        self._is_initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Sample a fresh multi-leg schedule for the given environments."""

        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        count = ids.numel()
        if count == 0:
            return
        k = self.config.max_legs
        low, high = self.config.leg_duration_range_s

        durations = low + (high - low) * torch.rand(
            (count, k), generator=self._generator, dtype=torch.float64,
        )
        starts_cpu = torch.zeros((count, k), dtype=torch.float64)
        starts_cpu[:, 1:] = torch.cumsum(durations[:, :-1], dim=1)
        starts = starts_cpu.to(device=self.device, dtype=self.dtype)

        direction = _sample_uniform_sphere(
            count * k, generator=self._generator, device=self.device, dtype=self.dtype,
        ).reshape(count, k, 3)
        speed_values = torch.tensor(
            self.config.speed_bins_mps, device=self.device, dtype=self.dtype
        )
        bin_idx = torch.randint(0, 3, (count, k), generator=self._generator).to(self.device)
        speed = speed_values[bin_idx]
        velocity = direction * speed.unsqueeze(-1)

        # Leg 0 gets a fresh random attitude target.  Each later leg either
        # keeps the previous leg's target or gets an independent fresh one --
        # a coin-flip retarget generalizing v2's single reversal-triggered
        # yaw step to an arbitrary number of legs.
        initial_quaternion = _sample_uniform_quaternion(
            count, generator=self._generator, device=self.device, dtype=self.dtype,
        )
        fresh_targets = _sample_uniform_quaternion(
            count * (k - 1), generator=self._generator, device=self.device, dtype=self.dtype,
        ).reshape(count, k - 1, 4)
        keep_roll = torch.rand((count, k - 1), generator=self._generator).to(self.device)
        keep_mask = keep_roll >= self.config.new_attitude_probability

        quaternion = torch.empty((count, k, 4), device=self.device, dtype=self.dtype)
        quaternion[:, 0] = initial_quaternion
        previous = initial_quaternion
        for leg in range(1, k):
            candidate = torch.where(
                keep_mask[:, leg - 1 : leg], previous, fresh_targets[:, leg - 1]
            )
            quaternion[:, leg] = candidate
            previous = candidate

        self.leg_starts[ids] = starts
        self.leg_velocity[ids] = velocity
        self.leg_quaternion[ids] = quaternion
        self._is_initialized[ids] = True

    def _active_leg(self, starts: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        leg = (starts <= time.unsqueeze(-1)).sum(dim=-1) - 1
        return leg.clamp(0, self.config.max_legs - 1)

    def sample(
        self,
        time_s: float | torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        previous_time_s: float | torch.Tensor | None = None,
    ) -> DeployV3Sample:
        """Return the active leg's command plus a leg-change transition mask."""

        ids = _as_env_ids(
            env_ids, num_envs=self.num_envs, device=self.device, validate=False,
        )
        time = _broadcast_time(
            time_s, count=ids.numel(), device=self.device, dtype=self.dtype,
        )
        if previous_time_s is None:
            previous_time = (time - self.config.policy_dt_s).clamp_min(0.0)
        else:
            previous_time = _broadcast_time(
                previous_time_s, count=ids.numel(), device=self.device, dtype=self.dtype,
            )

        starts = self.leg_starts[ids]
        active_leg = self._active_leg(starts, time)
        previous_leg = self._active_leg(starts, previous_time)
        transition_mask = active_leg != previous_leg

        def _gather(table: torch.Tensor, leg: torch.Tensor, width: int) -> torch.Tensor:
            index = leg.view(-1, 1, 1).expand(-1, 1, width)
            return torch.gather(table[ids], 1, index).squeeze(1)

        velocity = _gather(self.leg_velocity, active_leg, 3)
        quaternion = _gather(self.leg_quaternion, active_leg, 4)
        previous_velocity = _gather(self.leg_velocity, previous_leg, 3)

        previous_speed = previous_velocity.norm(dim=-1)
        current_speed = velocity.norm(dim=-1)
        denominator = (previous_speed * current_speed).clamp_min(1.0e-9)
        cos_angle = (previous_velocity * velocity).sum(dim=-1) / denominator
        reversal_mask = (
            transition_mask
            & (cos_angle < 0.0)
            & (previous_speed > 1.0e-6)
            & (current_speed > 1.0e-6)
        )

        return DeployV3Sample(
            velocity_body=velocity,
            desired_quaternion=quaternion,
            transition_mask=transition_mask,
            transition_mode=active_leg.clone(),
            reversal_mask=reversal_mask,
            active_leg=active_leg,
        )


@dataclass(frozen=True)
class DeployV6Config(DeployV3Config):
    """DeployV3Config plus a LOS-coupled attitude-retarget probability.

    See ``DeployV6Scheduler`` -- a fraction of leg attitude retargets are set
    to the heading of the new leg's own velocity direction (mirroring
    ``guidance/los_guidance.py``'s ``heading_mode="align"``) instead of an
    independent random draw.
    """

    los_coupled_retarget_probability: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not (0.0 <= self.los_coupled_retarget_probability <= 1.0):
            raise ValueError("los_coupled_retarget_probability must be in [0, 1]")


@dataclass(frozen=True)
class DeployV6Sample(DeployV3Sample):
    """DeployV3Sample plus the fields needed for live LOS-style coupling.

    ``world_direction`` is the active leg's raw sampled direction*speed
    vector. For a LOS-coupled leg this must be rotated into the vehicle's
    *current* body frame by the caller (``BROVVelEnv._current_v_d_b``) --
    unlike every other velocity command in this module, it is not already a
    body-frame quantity. ``los_coupled_mask`` says which environments are
    currently on such a leg.
    """

    world_direction: torch.Tensor
    los_coupled_mask: torch.Tensor


class DeployV6Scheduler(DeployV3Scheduler):
    """DeployV3Scheduler with a fraction of retargets coupled to velocity.

    Every field/table from :class:`DeployV3Scheduler` is reused unchanged;
    this only adds a per-(env, leg) ``leg_los_coupled`` table and overrides
    ``reset()`` to populate it, and ``sample()`` to expose it. The active
    leg's stored velocity for a coupled leg is a world-frame direction, not
    a body-frame command -- see :class:`DeployV6Sample`.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        config: DeployV6Config | None = None,
    ) -> None:
        super().__init__(
            num_envs,
            device=device,
            dtype=dtype,
            seed=seed,
            config=config or DeployV6Config(),
        )
        k = self.config.max_legs
        self.leg_los_coupled = torch.zeros(
            (num_envs, k), dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Sample a fresh multi-leg schedule, coupling some attitude retargets."""

        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        count = ids.numel()
        if count == 0:
            return
        k = self.config.max_legs
        low, high = self.config.leg_duration_range_s

        durations = low + (high - low) * torch.rand(
            (count, k), generator=self._generator, dtype=torch.float64,
        )
        starts_cpu = torch.zeros((count, k), dtype=torch.float64)
        starts_cpu[:, 1:] = torch.cumsum(durations[:, :-1], dim=1)
        starts = starts_cpu.to(device=self.device, dtype=self.dtype)

        direction = _sample_uniform_sphere(
            count * k, generator=self._generator, device=self.device, dtype=self.dtype,
        ).reshape(count, k, 3)
        speed_values = torch.tensor(
            self.config.speed_bins_mps, device=self.device, dtype=self.dtype
        )
        bin_idx = torch.randint(0, 3, (count, k), generator=self._generator).to(self.device)
        speed = speed_values[bin_idx]
        velocity = direction * speed.unsqueeze(-1)

        initial_quaternion = _sample_uniform_quaternion(
            count, generator=self._generator, device=self.device, dtype=self.dtype,
        )
        fresh_targets = _sample_uniform_quaternion(
            count * (k - 1), generator=self._generator, device=self.device, dtype=self.dtype,
        ).reshape(count, k - 1, 4)
        keep_roll = torch.rand((count, k - 1), generator=self._generator).to(self.device)
        keep_mask = keep_roll >= self.config.new_attitude_probability

        # Among legs that DO get a new target (not keep_mask), a fraction
        # additionally couple that target to the leg's own velocity
        # direction instead of drawing an independent fresh_targets sample.
        coupled_roll = torch.rand((count, k - 1), generator=self._generator).to(self.device)
        los_coupled_new = (~keep_mask) & (
            coupled_roll < self.config.los_coupled_retarget_probability
        )
        los_coupled_targets = _heading_from_direction(direction[:, 1:])

        quaternion = torch.empty((count, k, 4), device=self.device, dtype=self.dtype)
        quaternion[:, 0] = initial_quaternion
        los_coupled = torch.zeros((count, k), dtype=torch.bool, device=self.device)
        previous = initial_quaternion
        for leg in range(1, k):
            candidate = torch.where(
                keep_mask[:, leg - 1 : leg], previous, fresh_targets[:, leg - 1]
            )
            candidate = torch.where(
                los_coupled_new[:, leg - 1 : leg],
                los_coupled_targets[:, leg - 1],
                candidate,
            )
            quaternion[:, leg] = candidate
            los_coupled[:, leg] = los_coupled_new[:, leg - 1]
            previous = candidate

        self.leg_starts[ids] = starts
        self.leg_velocity[ids] = velocity
        self.leg_quaternion[ids] = quaternion
        self.leg_los_coupled[ids] = los_coupled
        self._is_initialized[ids] = True

    def sample(
        self,
        time_s: float | torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        previous_time_s: float | torch.Tensor | None = None,
    ) -> DeployV6Sample:
        """Return the active leg's command, transition edge, and coupling flag."""

        base = super().sample(time_s, env_ids, previous_time_s=previous_time_s)
        ids = _as_env_ids(
            env_ids, num_envs=self.num_envs, device=self.device, validate=False,
        )
        index = base.active_leg.view(-1, 1)
        los_coupled_mask = torch.gather(self.leg_los_coupled[ids], 1, index).squeeze(1)
        world_direction = torch.gather(
            self.leg_velocity[ids], 1, base.active_leg.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)

        return DeployV6Sample(
            velocity_body=base.velocity_body,
            desired_quaternion=base.desired_quaternion,
            transition_mask=base.transition_mask,
            transition_mode=base.transition_mode,
            reversal_mask=base.reversal_mask,
            active_leg=base.active_leg,
            world_direction=world_direction,
            los_coupled_mask=los_coupled_mask,
        )


__all__ = [
    "DeployAttitudeMode",
    "DeployTransitionMode",
    "DeployV2Config",
    "DeployV2Sample",
    "DeployV2Scheduler",
    "DeployV3Config",
    "DeployV3Sample",
    "DeployV3Scheduler",
    "DeployV6Config",
    "DeployV6Sample",
    "DeployV6Scheduler",
    "DeployVelocityMode",
    "PaperReferenceBatch",
    "PaperReferenceConfig",
]
