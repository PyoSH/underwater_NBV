"""
IsaacLab 의존성 없는 최소 쿼터니언 유틸
==========================================
`deploy/` 아래 코드는 topside PC(IsaacLab/Isaac Sim 미설치)에서 순수 torch만으로
돌아가야 한다 — `isaaclab.utils.math`를 그대로 가져다 쓸 수 없어서 실제로 쓰는
함수(quat_mul, quat_conjugate, quat_apply, quat_from_euler_xyz)만 여기 재구현한다.

쿼터니언 순서는 이 코드베이스 전체와 동일하게 [w, x, y, z].
"""

from __future__ import annotations

import torch


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2. (..., 4) [w,x,y,z]."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """쿼터니언 q로 벡터 v를 회전 (q ⊗ [0,v] ⊗ q⁻¹의 벡터부, 전개식으로 직접 계산)."""
    q_w = q[..., 0:1]
    q_vec = q[..., 1:4]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """ZYX(yaw-pitch-roll) 오일러각 → 쿼터니언. test_policy.py._quat_to_euler_zyx_deg의 역변환과 동일 규약."""
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.stack([w, x, y, z], dim=-1)


def sample_uniform(lo: float, hi: float, shape: tuple, device) -> torch.Tensor:
    return torch.rand(shape, device=device) * (hi - lo) + lo


def identity_quat(n: int, device) -> torch.Tensor:
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0
    return q
