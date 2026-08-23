"""Versioned policy-action to thruster-allocation frame contract.

The policy action is ordered as ``[Fx, Fy, Fz, Mx, My, Mz]`` in the
Isaac/ROS body FLU (Z-up) frame.  The BlueROV allocation matrix is built in
SNAME/FRD (X forward, Y starboard, Z down).  A 180 degree rotation about the
body X axis therefore maps both force and moment with

``T6 = diag(1, -1, -1, 1, -1, -1)``.

Do not silently apply this transform to the frozen ``model_299`` policy.  It
was trained with the six policy values passed numerically to the SNAME
allocation matrix, so its compatibility adapter is deliberately named and
kept separate from the corrected contract.

This module only depends on PyTorch.  It can be copied/vendorized into the ROS
runtime without importing Isaac Lab.
"""

from __future__ import annotations

from typing import Final

import torch


WRENCH_AXIS_ORDER: Final[tuple[str, ...]] = (
    "surge",
    "sway",
    "heave",
    "roll",
    "pitch",
    "yaw",
)

# Correct contract for policies trained with FLU/Z-up wrench semantics.
EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1: Final[str] = (
    "explicit_flu_zup_to_sname_frd_v1"
)

# Compatibility-only contract for logs/artifacts produced by model_299.pt.
LEGACY_MODEL_299_NO_T6: Final[str] = "legacy_model_299_no_t6"

SUPPORTED_ACTION_FRAME_CONTRACTS: Final[tuple[str, ...]] = (
    EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1,
    LEGACY_MODEL_299_NO_T6,
)

T6_DIAGONAL: Final[tuple[float, ...]] = (
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
)


def _require_finite_float_tensor(
    value: torch.Tensor,
    *,
    name: str,
    last_dimension: int | None = None,
) -> torch.Tensor:
    """Validate tensor type, floating dtype, shape and finite values."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if last_dimension is not None:
        if value.ndim == 0 or value.shape[-1] != last_dimension:
            raise ValueError(
                f"{name} must have last dimension {last_dimension}; "
                f"got shape {tuple(value.shape)}"
            )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _t6_like(reference: torch.Tensor) -> torch.Tensor:
    """Return T6 diagonal signs on the reference tensor's dtype/device."""
    return reference.new_tensor(T6_DIAGONAL)


def _validate_contract(contract: str) -> None:
    if contract not in SUPPORTED_ACTION_FRAME_CONTRACTS:
        choices = ", ".join(SUPPORTED_ACTION_FRAME_CONTRACTS)
        raise ValueError(f"unknown action-frame contract {contract!r}; use {choices}")


def policy_flu_zup_to_sname_frd(policy_wrench: torch.Tensor) -> torch.Tensor:
    """Map a finite ``(..., 6)`` FLU/Z-up wrench into SNAME/FRD.

    ``T6`` is self-inverse, but the reverse direction has a separate named
    function so call sites document the source and destination frames.
    """
    policy_wrench = _require_finite_float_tensor(
        policy_wrench, name="policy_wrench", last_dimension=6
    )
    return policy_wrench * _t6_like(policy_wrench)


def sname_frd_to_policy_flu_zup(sname_wrench: torch.Tensor) -> torch.Tensor:
    """Map a finite ``(..., 6)`` SNAME/FRD wrench into FLU/Z-up."""
    sname_wrench = _require_finite_float_tensor(
        sname_wrench, name="sname_wrench", last_dimension=6
    )
    return sname_wrench * _t6_like(sname_wrench)


def policy_wrench_to_sname_frd(
    policy_wrench: torch.Tensor,
    *,
    contract: str,
) -> torch.Tensor:
    """Apply one explicitly selected, versioned action-frame contract.

    ``LEGACY_MODEL_299_NO_T6`` intentionally returns the policy values without
    a frame transform.  That reproduces the historical training/deployment
    path; it is not a claim that FLU and SNAME are the same frame.
    """
    policy_wrench = _require_finite_float_tensor(
        policy_wrench, name="policy_wrench", last_dimension=6
    )
    _validate_contract(contract)
    if contract == EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1:
        return policy_flu_zup_to_sname_frd(policy_wrench)
    return policy_wrench.clone()


def sname_frd_to_policy_wrench(
    sname_wrench: torch.Tensor,
    *,
    contract: str,
) -> torch.Tensor:
    """Inverse of :func:`policy_wrench_to_sname_frd` for the same contract."""
    sname_wrench = _require_finite_float_tensor(
        sname_wrench, name="sname_wrench", last_dimension=6
    )
    _validate_contract(contract)
    if contract == EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1:
        return sname_frd_to_policy_flu_zup(sname_wrench)
    return sname_wrench.clone()


def policy_action_to_sname_frd_wrench(
    policy_action: torch.Tensor,
    wrench_scale,
    *,
    contract: str,
) -> torch.Tensor:
    """Scale a finite ``(..., 6)`` policy action, then apply its frame contract.

    Action clipping/operational limits deliberately remain outside this helper:
    callers must pass the action that is actually being allocated.  Keeping
    limiting separate makes raw, limited, requested and achieved diagnostics
    unambiguous.
    """
    policy_action = _require_finite_float_tensor(
        policy_action, name="policy_action", last_dimension=6
    )
    scale = torch.as_tensor(
        wrench_scale, dtype=policy_action.dtype, device=policy_action.device
    )
    scale = _require_finite_float_tensor(
        scale, name="wrench_scale", last_dimension=6
    )
    if scale.ndim != 1:
        raise ValueError(
            f"wrench_scale must have shape (6,); got shape {tuple(scale.shape)}"
        )
    if bool((scale < 0.0).any()):
        raise ValueError("wrench_scale must contain non-negative magnitudes")
    return policy_wrench_to_sname_frd(
        policy_action * scale, contract=contract
    )


def build_policy_action_to_sname_frd_multiplier(
    wrench_scale,
    *,
    contract: str,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the cached ``(6,)`` action-to-SNAME multiplier for a hot loop.

    The functional adapter above validates each dynamic input and is suitable
    for runtime boundaries and offline diagnostics.  A vectorized Isaac Lab
    training loop must not perform a host/device finite-value synchronization
    every step.  Validate this constant once during environment construction,
    cache the returned tensor, then use ``sname_wrench = action * multiplier``.
    """
    _validate_contract(contract)
    scale = torch.as_tensor(wrench_scale, dtype=dtype, device=device)
    scale = _require_finite_float_tensor(
        scale, name="wrench_scale", last_dimension=6
    )
    if scale.ndim != 1:
        raise ValueError(
            f"wrench_scale must have shape (6,); got shape {tuple(scale.shape)}"
        )
    if bool((scale < 0.0).any()):
        raise ValueError("wrench_scale must contain non-negative magnitudes")
    if contract == EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1:
        return scale * _t6_like(scale)
    return scale.clone()


def allocate_sname_frd_wrench(
    sname_wrench: torch.Tensor,
    allocation_pinv: torch.Tensor,
) -> torch.Tensor:
    """Allocate ``(..., 6)`` SNAME/FRD wrench through a ``(N, 6)`` B+ matrix."""
    sname_wrench = _require_finite_float_tensor(
        sname_wrench, name="sname_wrench", last_dimension=6
    )
    allocation_pinv = _require_finite_float_tensor(
        allocation_pinv, name="allocation_pinv"
    )
    if allocation_pinv.ndim != 2 or allocation_pinv.shape[1] != 6:
        raise ValueError(
            "allocation_pinv must have shape (num_thrusters, 6); "
            f"got {tuple(allocation_pinv.shape)}"
        )
    matrix = allocation_pinv.to(
        dtype=sname_wrench.dtype, device=sname_wrench.device
    )
    return torch.matmul(sname_wrench, matrix.transpose(0, 1))


def thruster_forces_to_sname_frd_wrench(
    thruster_forces: torch.Tensor,
    allocation_matrix: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct achieved ``(..., 6)`` SNAME/FRD wrench with ``B(6, N)``."""
    thruster_forces = _require_finite_float_tensor(
        thruster_forces, name="thruster_forces"
    )
    allocation_matrix = _require_finite_float_tensor(
        allocation_matrix, name="allocation_matrix"
    )
    if allocation_matrix.ndim != 2 or allocation_matrix.shape[0] != 6:
        raise ValueError(
            "allocation_matrix must have shape (6, num_thrusters); "
            f"got {tuple(allocation_matrix.shape)}"
        )
    if thruster_forces.ndim == 0 or (
        thruster_forces.shape[-1] != allocation_matrix.shape[1]
    ):
        raise ValueError(
            "thruster_forces last dimension must match allocation_matrix's "
            f"num_thrusters={allocation_matrix.shape[1]}; "
            f"got shape {tuple(thruster_forces.shape)}"
        )
    matrix = allocation_matrix.to(
        dtype=thruster_forces.dtype, device=thruster_forces.device
    )
    return torch.matmul(thruster_forces, matrix.transpose(0, 1))


__all__ = [
    "WRENCH_AXIS_ORDER",
    "EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1",
    "LEGACY_MODEL_299_NO_T6",
    "SUPPORTED_ACTION_FRAME_CONTRACTS",
    "T6_DIAGONAL",
    "policy_flu_zup_to_sname_frd",
    "sname_frd_to_policy_flu_zup",
    "policy_wrench_to_sname_frd",
    "sname_frd_to_policy_wrench",
    "policy_action_to_sname_frd_wrench",
    "build_policy_action_to_sname_frd_multiplier",
    "allocate_sname_frd_wrench",
    "thruster_forces_to_sname_frd_wrench",
]
