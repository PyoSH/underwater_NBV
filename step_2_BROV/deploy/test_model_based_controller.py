"""모터 출력 없는 model-based controller 수학 회귀 테스트."""

import math

import torch

from deploy.model_based_controller import (
    ModelBasedController,
    quaternion_error_rotation_vector,
)
from deploy.vendor.params import load_brov2_yaml, thruster_pos_dir_ned


def _controller() -> ModelBasedController:
    pos, direction = thruster_pos_dir_ned(load_brov2_yaml())
    return ModelBasedController(pos, direction)


def test_identity_is_zero_wrench():
    obs = torch.zeros(16)
    obs[0] = 1.0
    output = _controller().compute(obs)
    assert torch.allclose(output.wrench_zup, torch.zeros(6), atol=1e-6)
    assert torch.allclose(output.pwm, torch.zeros(8), atol=1e-6)


def test_velocity_error_uses_negative_feedback():
    obs = torch.zeros(16)
    obs[0] = 1.0
    obs[4:7] = torch.tensor([-0.1, 0.2, -0.3])
    output = _controller().compute(obs)
    assert torch.allclose(output.wrench_zup[:3], torch.tensor([2.5, -5.0, 10.5]))
    active = output.pwm != 0.0
    assert bool(active.any())
    assert bool((output.pwm[active].abs() >= 0.10).all())


def test_pitch_error_and_frame_transform_signs():
    obs = torch.zeros(16)
    angle = math.radians(5.0)
    obs[0] = math.cos(angle / 2.0)
    obs[2] = math.sin(angle / 2.0)
    output = _controller().compute(obs)
    assert output.wrench_zup[4] < 0.0
    assert output.wrench_sname[4] > 0.0


def test_quaternion_sign_has_same_rotation_vector():
    q = torch.tensor([0.9, 0.1, -0.2, 0.3])
    assert torch.allclose(
        quaternion_error_rotation_vector(q),
        quaternion_error_rotation_vector(-q),
        atol=1e-6,
    )
