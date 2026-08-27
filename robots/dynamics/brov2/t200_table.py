"""T200 thrust map backed by Blue Robotics' published performance data.

Replaces the two-stage ``PWM -> RPM -> thrust`` polynomial the thruster model
used to carry.  The measured table is exact at every supply voltage, holds the
dead zone and the forward/reverse asymmetry in the data itself rather than in
hand-tuned constants, and inverts by table search so ``force(pwm(f)) == f`` to
float precision.  The old quadratic inverse could not do that: for a requested
force inside the dead zone its discriminant clamp pinned the root at the
parabola vertex and returned an opposite-sign command.

Regenerate the table with ``build_t200_table.py``.  Depends only on numpy and
torch, so it vendorizes into the ROS runtime unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


DEFAULT_TABLE_PATH = Path(__file__).with_name("t200_table.npz")


class T200ThrustTable:
    """Bilinear ``(supply voltage, normalized PWM) -> thrust`` lookup.

    Every method takes ``voltage`` as a ``(N,)`` tensor of volts, one per
    environment, and ``(N, K)`` per-thruster tensors.
    """

    def __init__(
        self,
        npz_path: str | Path = DEFAULT_TABLE_PATH,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        data = np.load(str(npz_path))
        self._volts = torch.as_tensor(data["volts"], dtype=dtype, device=device)
        self._pwm = torch.as_tensor(data["pwm_norm"], dtype=dtype, device=device)
        self._force = torch.as_tensor(data["force_n"], dtype=dtype, device=device)
        if self._force.shape != (self._volts.numel(), self._pwm.numel()):
            raise ValueError("force_n must have shape (num_volts, num_pwm)")
        if not bool((self._pwm.diff() > 0).all()):
            raise ValueError("pwm_norm must be strictly increasing")
        if not bool((self._force.diff(dim=1) >= 0).all()):
            raise ValueError("force_n must be non-decreasing along the pwm axis")

        positive = torch.where(
            self._force > 0.0, self._force, self._force.new_full((), float("inf"))
        )
        negative = torch.where(
            self._force < 0.0, self._force, self._force.new_full((), float("-inf"))
        )
        self._min_forward = positive.min(dim=1).values   # (V,) smallest producible +thrust
        self._min_reverse = negative.max(dim=1).values   # (V,) smallest producible -thrust
        self._pwm_first = float(self._pwm[0])
        self._pwm_step = float(self._pwm[1] - self._pwm[0])
        self._num_pwm = self._pwm.numel()
        self._bracket_lo: torch.Tensor | None = None
        self._bracket_hi: torch.Tensor | None = None

    @property
    def voltage_range(self) -> tuple[float, float]:
        return float(self._volts[0]), float(self._volts[-1])

    @staticmethod
    def _as_batched(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Accept ``(K,)`` as well as ``(N, K)``.

        The ROS runtime allocates a single vehicle's wrench and passes the bare
        ``(8,)`` thruster vector (brov_control/policy_node.py), while training
        passes ``(num_envs, 8)``.  The old scalar force limits broadcast over
        both by accident; per-env limits do not, so normalize here instead of
        letting a ``(8,)`` input silently come back as ``(1, 8)``.
        """
        if values.dim() == 1:
            return values.unsqueeze(0), True
        return values, False

    def _curve(self, voltage: torch.Tensor) -> torch.Tensor:
        """Blend the two bracketing measured curves for each env -> ``(N, P)``."""
        volts = voltage.reshape(-1).clamp(self._volts[0], self._volts[-1])
        upper = torch.searchsorted(self._volts, volts.contiguous()).clamp(
            1, self._volts.numel() - 1
        )
        lower = upper - 1
        weight = (
            (volts - self._volts[lower]) / (self._volts[upper] - self._volts[lower])
        ).unsqueeze(-1)
        self._bracket_lo, self._bracket_hi = lower, upper
        return torch.lerp(self._force[lower], self._force[upper], weight)

    def force(self, pwm_norm: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        """``(K,)`` or ``(N, K)`` normalized PWM in ``[-1, 1]`` -> thrust in N."""
        pwm_norm, squeeze = self._as_batched(pwm_norm)
        curve = self._curve(voltage)
        index = (pwm_norm.clamp(-1.0, 1.0) - self._pwm_first) / self._pwm_step
        lower = index.floor().clamp(0, self._num_pwm - 2).long()
        weight = (index - lower.to(index.dtype)).clamp(0.0, 1.0)
        out = torch.lerp(curve.gather(1, lower), curve.gather(1, lower + 1), weight)
        return out.squeeze(0) if squeeze else out

    def pwm(self, force_n: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        """``(N, K)`` desired thrust in N -> ``(N, K)`` normalized PWM.

        A force inside the dead zone cannot be produced at all, so it maps to
        ``0`` instead of to a command the thruster would answer with either
        nothing or a full minimum-thrust step.
        """
        force_n, squeeze = self._as_batched(force_n)
        curve = self._curve(voltage)
        target = force_n.clamp(curve[:, :1], curve[:, -1:])
        upper = torch.searchsorted(curve.contiguous(), target.contiguous()).clamp(
            1, self._num_pwm - 1
        )
        lower = upper - 1
        force_lo = curve.gather(1, lower)
        force_hi = curve.gather(1, upper)
        weight = (
            (target - force_lo) / (force_hi - force_lo).clamp_min(1e-9)
        ).clamp(0.0, 1.0)
        pwm = self._pwm[lower] + weight * self._pwm_step

        # The producibility threshold comes from the two bracketing *measured*
        # curves, never from the voltage-blended one.  Blending curves whose
        # dead zones differ in width leaves a small non-zero force in the gap
        # that no real thruster can deliver.
        min_forward = torch.maximum(
            self._min_forward[self._bracket_lo], self._min_forward[self._bracket_hi]
        ).unsqueeze(-1)
        min_reverse = torch.minimum(
            self._min_reverse[self._bracket_lo], self._min_reverse[self._bracket_hi]
        ).unsqueeze(-1)
        producible = (force_n >= min_forward) | (force_n <= min_reverse)
        out = torch.where(producible, pwm, torch.zeros_like(pwm))
        return out.squeeze(0) if squeeze else out

    def force_limits(self, voltage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env ``(reverse, forward)`` limits as ``(N, 1)`` tensors."""
        curve = self._curve(voltage)
        return curve[:, :1], curve[:, -1:]

    def clamp_thrust(self, force_n: torch.Tensor, voltage: torch.Tensor) -> torch.Tensor:
        force_n, squeeze = self._as_batched(force_n)
        lower, upper = self.force_limits(voltage)
        out = force_n.clamp(lower, upper)
        return out.squeeze(0) if squeeze else out

    def dead_zone(self, voltage: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env ``(min_reverse, min_forward)`` producible thrust, ``(N, 1)``."""
        self._curve(voltage)
        return (
            torch.minimum(
                self._min_reverse[self._bracket_lo], self._min_reverse[self._bracket_hi]
            ).unsqueeze(-1),
            torch.maximum(
                self._min_forward[self._bracket_lo], self._min_forward[self._bracket_hi]
            ).unsqueeze(-1),
        )
