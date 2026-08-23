"""Shared, pure-Torch 16-D velocity-policy observation contract.

The training environment receives a fresh state at a fixed 25 Hz, while the
ROS runtime may freeze on duplicate/stale samples and use a bounded source
``dt``.  Keeping the state transition in a small tensor-only function makes the
two implementations comparable with golden traces.
"""

from __future__ import annotations

import torch


def canonicalize_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Return the scalar-positive representation of a unit quaternion.

    ``q`` and ``-q`` encode the same rotation but are very different neural
    network inputs.  The runtime uses the same ``w >= 0`` convention.
    """

    if quaternion_wxyz.shape[-1] != 4:
        raise ValueError("quaternion_wxyz must have a final dimension of 4")
    return torch.where(
        quaternion_wxyz[..., :1] < 0.0,
        -quaternion_wxyz,
        quaternion_wxyz,
    )


def _as_batched_dt(dt: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dt_tensor = torch.as_tensor(dt, dtype=reference.dtype, device=reference.device)
    if dt_tensor.ndim == 0:
        return dt_tensor
    if dt_tensor.shape == reference.shape[:-1]:
        return dt_tensor.unsqueeze(-1)
    if dt_tensor.shape == (*reference.shape[:-1], 1):
        return dt_tensor
    raise ValueError("dt must be scalar or match the observation batch shape")


def _as_integrate_mask(
    integrate: bool | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor | bool:
    """Accept a per-env scalar mask (broadcast to every component) or a
    full per-component mask (``reference.shape``, e.g. one bool per z_v/z_q
    axis) -- the latter is what deploy_v6's per-axis integrator anti-windup
    (see vel_env.py::_get_observations) needs: halting z_q's pitch
    component specifically while roll/yaw keep integrating."""

    if isinstance(integrate, bool):
        return integrate
    mask = torch.as_tensor(integrate, dtype=torch.bool, device=reference.device)
    if mask.shape == reference.shape:
        return mask
    if mask.shape == reference.shape[:-1]:
        return mask.unsqueeze(-1)
    if mask.shape == (*reference.shape[:-1], 1):
        return mask
    raise ValueError("integrate mask must match the observation batch or full shape")


def build_velocity_observation(
    *,
    quaternion_error_wxyz: torch.Tensor,
    velocity_error_body: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    integral_velocity: torch.Tensor,
    integral_attitude: torch.Tensor,
    dt: float | torch.Tensor,
    integrate: bool | torch.Tensor = True,
    integrate_velocity: bool | torch.Tensor | None = None,
    integrate_attitude: bool | torch.Tensor | None = None,
    integral_velocity_limit: float = 5.0,
    integral_attitude_limit: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``[q_e, v_e, omega, z_v, z_q]`` and advance its integrals.

    ``integrate=False`` is the runtime stale/duplicate-sample behavior.  Isaac
    training always supplies ``True`` with ``dt=0.04``.  Returned integral
    tensors are new values; callers own their persistent buffers.

    ``integrate_velocity``/``integrate_attitude`` independently override
    ``integrate`` for ``z_v``/``z_q`` respectively (default: fall back to
    ``integrate``, so every existing call site is unaffected).  A DVL-realism
    model that only produces a fresh velocity reading at, say, 10 Hz should
    pass its own ``fresh_sample_mask`` as ``integrate_velocity`` while leaving
    ``integrate_attitude`` on the regular per-tick IMU-rate cadence. Either
    override also accepts a full ``(..., 3)`` per-axis mask (not just a
    per-env scalar) -- deploy_v6's integrator anti-windup uses this to halt
    only the saturated axis's integral component (e.g. z_q's pitch entry)
    while its other two components keep integrating normally.
    """

    vectors = (
        velocity_error_body,
        angular_velocity_body,
        integral_velocity,
        integral_attitude,
    )
    if quaternion_error_wxyz.shape[-1] != 4 or any(v.shape[-1] != 3 for v in vectors):
        raise ValueError("expected quaternion (...,4) and vector (...,3) inputs")
    batch_shape = quaternion_error_wxyz.shape[:-1]
    if any(v.shape[:-1] != batch_shape for v in vectors):
        raise ValueError("all observation inputs must share the same batch shape")
    if integral_velocity_limit <= 0.0 or integral_attitude_limit <= 0.0:
        raise ValueError("integral limits must be positive")

    q_e = canonicalize_quaternion(quaternion_error_wxyz)
    dt_tensor = _as_batched_dt(dt, velocity_error_body)
    vel_mask = _as_integrate_mask(
        integrate if integrate_velocity is None else integrate_velocity,
        velocity_error_body,
    )
    att_mask = _as_integrate_mask(
        integrate if integrate_attitude is None else integrate_attitude,
        velocity_error_body,
    )

    z_v_candidate = (integral_velocity + velocity_error_body * dt_tensor).clamp(
        -integral_velocity_limit, integral_velocity_limit
    )
    z_q_candidate = (integral_attitude + q_e[..., 1:] * dt_tensor).clamp(
        -integral_attitude_limit, integral_attitude_limit
    )
    z_v_next = (
        z_v_candidate if isinstance(vel_mask, bool) and vel_mask
        else integral_velocity if isinstance(vel_mask, bool)
        else torch.where(vel_mask, z_v_candidate, integral_velocity)
    )
    z_q_next = (
        z_q_candidate if isinstance(att_mask, bool) and att_mask
        else integral_attitude if isinstance(att_mask, bool)
        else torch.where(att_mask, z_q_candidate, integral_attitude)
    )

    observation = torch.cat(
        [q_e, velocity_error_body, angular_velocity_body, z_v_next, z_q_next],
        dim=-1,
    )
    return observation, z_v_next, z_q_next

