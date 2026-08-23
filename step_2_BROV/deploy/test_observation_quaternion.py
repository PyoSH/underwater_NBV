"""Observation quaternion의 q/-q 불연속 방지 회귀 테스트."""

import torch

from deploy import math_utils as mu


def test_quat_unique_selects_positive_scalar_hemisphere():
    q = torch.tensor([-0.998, -0.01, 0.04, 0.02])
    unique = mu.quat_unique(q)
    assert unique[0] > 0.0
    assert torch.allclose(unique, -q)


def test_quat_unique_does_not_change_rotated_vector():
    q = torch.tensor([0.7, 0.1, -0.2, 0.67])
    q = q / q.norm()
    vector = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(mu.quat_apply(q, vector), mu.quat_apply(-q, vector))
