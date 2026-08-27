"""Thruster actuator dynamics — command to delivered thrust over time.

The static thrust map (``t200_table.py``) answers *how much* thrust a PWM
produces.  This module answers *when*: rotor inertia, motor electrical
dynamics, and the entrained water the propeller has to accelerate all delay
and shape the response.

Two models are available.

``first_order``
    ``H(s) = 1 / (tau*s + 1)`` with ``tau = 0.05 s``.  What this codebase used
    to carry.  Its corner frequency is only ``1/(2*pi*tau) = 3.2 Hz``.

``third_order`` (default)
    von Benzon et al. 2022 Eq. (19), identified from MathWorks' published
    thruster step data::

        H(s) = (6136 s + 108700) / (s^3 + 89 s^2 + 9258 s + 108700)

    Real pole at -13.16 (tau ~ 0.076 s), a complex pair at -37.92 +/- 82.60j
    (omega_n = 14.5 Hz, zeta = 0.417), and a zero at -17.72.  Effective
    bandwidth is about 14.5 Hz.

Why the difference matters more than it looks: both have unity DC gain, so
they agree exactly in steady state.  They diverge only on fast content, and
that is precisely where a chattering policy lives.  At the 12.5 Hz a 25 Hz
control loop produces when its action alternates every step, ``first_order``
passes 0.247 of the amplitude while ``third_order`` passes 0.983 -- a factor
of four.  Training against ``first_order`` therefore filters out chatter that
the real thruster reproduces almost intact, so the reward never charges the
policy for it and PPO gets no gradient pushing the actions smooth.

Because Eq. (19) has unity DC gain and was identified independently of Eq.
(18)'s force regression, it composes with any static map -- including the
measured Blue Robotics table this codebase now uses.  The identification data
is not a T200 though, so ``bandwidth_scale`` randomization is supported: it
scales every pole and zero (``s -> s/k``), moving the corner frequency while
preserving both the shape and the unity DC gain.

Placement: the filter runs on PWM, before the static map.  The lag is rotor
and entrained-water inertia, which sits between the command and the rotation
that actually produces thrust.
"""

from __future__ import annotations

import torch


# von Benzon et al. 2022, Eq. (19). Denominator s^3 + a2 s^2 + a1 s + a0.
_A2, _A1, _A0 = 89.0, 9258.0, 108700.0
_B1, _B0 = 6136.0, 108700.0

FIRST_ORDER_TAU = 0.05


class ThrusterDynamics:
    """Per-thruster actuator dynamics, batched over environments.

    Parameters
    ----------
    num_envs, num_thrusters, dt, device
        Batch shape and the integration step (the physics step, not the policy
        step -- the filter runs inside ``compute()``).
    model
        ``"third_order"`` or ``"first_order"``.
    bandwidth_scales
        Discretized bank of pole/zero scale factors available to
        :meth:`randomize`.  ``1.0`` is the identified transfer function; ``2.0``
        is a thruster twice as fast.  Defaults to a single entry (no
        randomization).
    """

    def __init__(
        self,
        num_envs: int,
        num_thrusters: int,
        dt: float,
        device: str | torch.device = "cpu",
        model: str = "third_order",
        bandwidth_scales: tuple[float, ...] = (1.0,),
    ):
        if model not in ("third_order", "first_order"):
            raise ValueError(f"unknown thruster dynamics model {model!r}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if not bandwidth_scales or any(k <= 0.0 for k in bandwidth_scales):
            raise ValueError("bandwidth_scales must be non-empty and positive")

        self.model = model
        self.num_envs = num_envs
        self.num_thrusters = num_thrusters
        self.dt = float(dt)
        self.device = device
        self._scales = torch.as_tensor(
            list(bandwidth_scales), dtype=torch.float32, device=device
        )
        self._order = 3 if model == "third_order" else 1

        matrices_a, matrices_b, output_c = [], [], []
        for scale in bandwidth_scales:
            a_d, b_d, c = self._discretize(float(scale))
            matrices_a.append(a_d)
            matrices_b.append(b_d)
            output_c.append(c)
        self._a_d = torch.stack(matrices_a).to(device)          # (K, n, n)
        self._b_d = torch.stack(matrices_b).to(device)          # (K, n)
        self._c = torch.stack(output_c).to(device)              # (K, n)

        self._state = torch.zeros(num_envs, num_thrusters, self._order, device=device)
        self._index = torch.zeros(num_envs, dtype=torch.long, device=device)

    def _discretize(self, scale: float):
        """Zero-order-hold discretization of the scaled transfer function.

        ``s -> s/scale`` multiplies every pole and zero by ``scale``.  In the
        controllable canonical form that means ``a_i -> a_i * scale^(n-i)`` and
        ``b_i -> b_i * scale^(n-i)``, which leaves ``b_0/a_0`` -- the DC gain --
        at exactly 1.
        """
        if self.model == "first_order":
            pole = scale / FIRST_ORDER_TAU
            a_c = torch.tensor([[-pole]], dtype=torch.float64)
            b_c = torch.tensor([pole], dtype=torch.float64)
            c = torch.tensor([1.0], dtype=torch.float64)
        else:
            a0 = _A0 * scale ** 3
            a1 = _A1 * scale ** 2
            a2 = _A2 * scale
            b0 = _B0 * scale ** 3
            b1 = _B1 * scale ** 2
            a_c = torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-a0, -a1, -a2]],
                dtype=torch.float64,
            )
            b_c = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
            c = torch.tensor([b0, b1, 0.0], dtype=torch.float64)

        a_d = torch.matrix_exp(a_c * self.dt)
        # Bd = A^-1 (Ad - I) B, valid because A is invertible here (a0 != 0).
        b_d = torch.linalg.solve(a_c, (a_d - torch.eye(a_c.shape[0], dtype=torch.float64)) @ b_c)
        return a_d.float(), b_d.float(), c.float()

    def randomize(self, env_ids: torch.Tensor, index: torch.Tensor | None = None) -> None:
        """Assign a bandwidth-scale bank entry per environment."""
        if self._scales.numel() == 1:
            return
        if index is None:
            index = torch.randint(
                0, self._scales.numel(), (env_ids.numel(),), device=self._index.device
            )
        self._index[env_ids] = index.reshape(-1).to(self._index.dtype)

    @property
    def bandwidth_scale(self) -> torch.Tensor:
        return self._scales[self._index]

    def reset(self, env_ids: torch.Tensor) -> None:
        self._state[env_ids] = 0.0

    def step(self, command: torch.Tensor) -> torch.Tensor:
        """Advance one ``dt`` and return the delivered (filtered) command."""
        a_d = self._a_d[self._index]                       # (N, n, n)
        b_d = self._b_d[self._index]                       # (N, n)
        c = self._c[self._index]                           # (N, n)
        # x[k+1] = Ad x[k] + Bd u[k];  y[k] = C x[k]
        self._state = torch.einsum("nij,nkj->nki", a_d, self._state) + (
            command.unsqueeze(-1) * b_d.unsqueeze(1)
        )
        return (self._state * c.unsqueeze(1)).sum(-1)

    @property
    def state(self) -> torch.Tensor:
        return self._state
