from __future__ import annotations

import torch

from envs.observation_contract import (
    build_velocity_observation,
    canonicalize_quaternion,
)


def _build(q: torch.Tensor, *, integrate=True, z=0.0, dt=0.04):
    batch = q.shape[0]
    vec = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32).repeat(batch, 1)
    omega = torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float32).repeat(batch, 1)
    integral = torch.full((batch, 3), float(z))
    return build_velocity_observation(
        quaternion_error_wxyz=q,
        velocity_error_body=vec,
        angular_velocity_body=omega,
        integral_velocity=integral,
        integral_attitude=integral.clone(),
        dt=dt,
        integrate=integrate,
    )


def test_antipodal_quaternions_produce_identical_observation():
    q = torch.tensor([[0.5, -0.5, 0.5, -0.5]], dtype=torch.float32)
    obs_a, zva, zqa = _build(q)
    obs_b, zvb, zqb = _build(-q)
    torch.testing.assert_close(obs_a, obs_b)
    torch.testing.assert_close(zva, zvb)
    torch.testing.assert_close(zqa, zqb)


def test_integrators_clamp_and_stale_sample_freezes():
    q = torch.tensor([[1.0, 0.2, -0.3, 0.4]], dtype=torch.float32)
    obs, z_v, z_q = _build(q, z=4.99, dt=1.0)
    torch.testing.assert_close(z_v, torch.tensor([[5.0, 2.99, 5.0]]))
    torch.testing.assert_close(z_q, torch.tensor([[5.0, 4.69, 5.0]]))
    assert obs.shape == (1, 16)

    frozen_obs, frozen_z_v, frozen_z_q = build_velocity_observation(
        quaternion_error_wxyz=q,
        velocity_error_body=torch.ones(1, 3),
        angular_velocity_body=torch.zeros(1, 3),
        integral_velocity=z_v,
        integral_attitude=z_q,
        dt=10.0,
        integrate=False,
    )
    torch.testing.assert_close(frozen_z_v, z_v)
    torch.testing.assert_close(frozen_z_q, z_q)
    torch.testing.assert_close(frozen_obs[:, 10:13], z_v)


def test_per_axis_mask_halts_only_the_saturated_component():
    """deploy_v6 anti-windup: a full (N,3) mask must gate each z_v/z_q
    component independently, not just per-env."""
    q = torch.tensor([[1.0, 0.1, 0.2, 0.3]], dtype=torch.float32)
    integral = torch.zeros(1, 3)
    # Only the y (pitch) axis is "saturated" (not integrated); x/z integrate.
    per_axis_mask = torch.tensor([[True, False, True]])
    obs, z_v, z_q = build_velocity_observation(
        quaternion_error_wxyz=q,
        velocity_error_body=torch.tensor([[1.0, 1.0, 1.0]]),
        angular_velocity_body=torch.zeros(1, 3),
        integral_velocity=integral,
        integral_attitude=integral.clone(),
        dt=1.0,
        integrate=True,
        integrate_velocity=per_axis_mask,
    )
    torch.testing.assert_close(z_v, torch.tensor([[1.0, 0.0, 1.0]]))
    # integrate_attitude was not overridden, so it still uses the scalar
    # integrate=True and advances all three components normally.
    torch.testing.assert_close(z_q, torch.tensor([[0.1, 0.2, 0.3]]))
    assert obs.shape == (1, 16)


def test_per_sample_mask_and_dt():
    q = canonicalize_quaternion(
        torch.tensor([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    )
    obs, z_v, _ = build_velocity_observation(
        quaternion_error_wxyz=q,
        velocity_error_body=torch.ones(2, 3),
        angular_velocity_body=torch.zeros(2, 3),
        integral_velocity=torch.zeros(2, 3),
        integral_attitude=torch.zeros(2, 3),
        dt=torch.tensor([0.02, 0.08]),
        integrate=torch.tensor([True, False]),
    )
    torch.testing.assert_close(z_v[0], torch.full((3,), 0.02))
    torch.testing.assert_close(z_v[1], torch.zeros(3))
    assert obs.shape == (2, 16)

