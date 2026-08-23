"""Reset-safe mass domain randomization for IsaacLab articulations.

This module deliberately depends on :mod:`torch` only.  The helper follows
the tensor contract of ``Articulation.root_physx_view`` and can therefore be
unit-tested without starting Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


class _RootPhysXView(Protocol):
    def get_masses(self) -> torch.Tensor: ...

    def set_masses(self, masses: torch.Tensor, indices: torch.Tensor) -> None: ...

    def get_inertias(self) -> torch.Tensor: ...

    def set_inertias(self, inertias: torch.Tensor, indices: torch.Tensor) -> None: ...


class _ArticulationData(Protocol):
    default_mass: torch.Tensor
    default_inertia: torch.Tensor


class ArticulationMassTarget(Protocol):
    """Structural type accepted by :func:`randomize_articulation_mass`."""

    root_physx_view: _RootPhysXView
    data: _ArticulationData
    num_bodies: int


@dataclass(frozen=True)
class MassRandomizationResult:
    """Selected values written by one mass-domain-randomization call."""

    env_ids: torch.Tensor
    body_ids: torch.Tensor
    scale: torch.Tensor
    masses: torch.Tensor
    inertias: torch.Tensor


def _cpu_index_tensor(
    values: torch.Tensor | Sequence[int] | None,
    *,
    size: int,
    name: str,
) -> torch.Tensor:
    if values is None:
        result = torch.arange(size, dtype=torch.long, device="cpu")
    else:
        result = torch.as_tensor(values, dtype=torch.long, device="cpu").reshape(-1)

    if result.numel() == 0:
        return result
    if bool(((result < 0) | (result >= size)).any()):
        raise IndexError(f"{name} contains an index outside [0, {size})")
    if torch.unique(result).numel() != result.numel():
        raise ValueError(f"{name} must not contain duplicate indices")
    return result


def randomize_articulation_mass(
    asset: ArticulationMassTarget,
    env_ids: torch.Tensor | Sequence[int] | None,
    *,
    relative_range: tuple[float, float] = (0.95, 1.05),
    body_ids: torch.Tensor | Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> MassRandomizationResult:
    """Uniformly scale nominal articulation mass and inertia by one ratio.

    A single ratio is sampled per selected environment and applied to all its
    selected rigid bodies.  Scaling inertia by the same mass ratio preserves
    nominal geometry/radius of gyration.  Every call uses
    ``asset.data.default_mass`` and ``default_inertia`` rather than current
    PhysX values, so repeated resets cannot compound mass drift.

    ``root_physx_view`` setters require an initialized/playing simulation and
    CPU environment indices.  No ``write_data_to_sim`` call is required for
    these root PhysX property setters.
    """

    lower, upper = (float(relative_range[0]), float(relative_range[1]))
    if not (0.0 < lower <= upper):
        raise ValueError("relative_range must satisfy 0 < lower <= upper")

    default_mass = asset.data.default_mass
    default_inertia = asset.data.default_inertia
    if default_mass.ndim != 2:
        raise ValueError("default_mass must have shape (num_envs, num_bodies)")
    if default_inertia.ndim != 3 or default_inertia.shape[-1] != 9:
        raise ValueError("default_inertia must have shape (num_envs, num_bodies, 9)")
    if default_inertia.shape[:2] != default_mass.shape:
        raise ValueError("default_mass and default_inertia leading shapes differ")
    if not bool(torch.isfinite(default_mass).all()) or bool((default_mass <= 0.0).any()):
        raise ValueError("default_mass must contain finite positive values")
    if not bool(torch.isfinite(default_inertia).all()):
        raise ValueError("default_inertia must contain finite values")

    num_envs, num_bodies = default_mass.shape
    if int(asset.num_bodies) != num_bodies:
        raise ValueError("asset.num_bodies does not match default_mass")
    selected_envs = _cpu_index_tensor(env_ids, size=num_envs, name="env_ids")
    selected_bodies = _cpu_index_tensor(body_ids, size=num_bodies, name="body_ids")

    # Avoid empty PhysX setter calls: they are unsafe in some Isaac Sim builds.
    if selected_envs.numel() == 0 or selected_bodies.numel() == 0:
        return MassRandomizationResult(
            env_ids=selected_envs,
            body_ids=selected_bodies,
            scale=default_mass.new_empty((selected_envs.numel(), 1)),
            masses=default_mass.new_empty(
                (selected_envs.numel(), selected_bodies.numel())
            ),
            inertias=default_inertia.new_empty(
                (selected_envs.numel(), selected_bodies.numel(), 9)
            ),
        )

    scale = torch.empty(
        (selected_envs.numel(), 1),
        dtype=default_mass.dtype,
        device=default_mass.device,
    ).uniform_(lower, upper, generator=generator)

    env_grid = selected_envs[:, None]
    nominal_mass = default_mass[env_grid, selected_bodies].clone()
    nominal_inertia = default_inertia[env_grid, selected_bodies].clone()
    selected_mass = nominal_mass * scale
    selected_inertia = nominal_inertia * scale.unsqueeze(-1)

    masses = asset.root_physx_view.get_masses().clone()
    inertias = asset.root_physx_view.get_inertias().clone()
    if masses.shape != default_mass.shape:
        raise ValueError("root_physx_view mass shape does not match default_mass")
    if inertias.shape != default_inertia.shape:
        raise ValueError("root_physx_view inertia shape does not match default_inertia")

    masses[env_grid, selected_bodies] = selected_mass
    inertias[env_grid, selected_bodies] = selected_inertia
    asset.root_physx_view.set_masses(masses, selected_envs)
    asset.root_physx_view.set_inertias(inertias, selected_envs)

    return MassRandomizationResult(
        env_ids=selected_envs,
        body_ids=selected_bodies,
        scale=scale,
        masses=selected_mass,
        inertias=selected_inertia,
    )
