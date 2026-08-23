from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# The shared dynamics package lives at the repository root, while this test is
# launched from step_2_BROV inside the IsaacLab container.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.dynamics.brov2.mass_randomization import randomize_articulation_mass


class _FakeRootPhysXView:
    def __init__(self, masses: torch.Tensor, inertias: torch.Tensor):
        self.masses = masses.clone()
        self.inertias = inertias.clone()
        self.mass_indices = None
        self.inertia_indices = None

    def get_masses(self) -> torch.Tensor:
        return self.masses.clone()

    def get_inertias(self) -> torch.Tensor:
        return self.inertias.clone()

    def set_masses(self, masses: torch.Tensor, indices: torch.Tensor) -> None:
        assert indices.device.type == "cpu"
        self.mass_indices = indices.clone()
        self.masses[indices] = masses[indices]

    def set_inertias(self, inertias: torch.Tensor, indices: torch.Tensor) -> None:
        assert indices.device.type == "cpu"
        self.inertia_indices = indices.clone()
        self.inertias[indices] = inertias[indices]


def _fake_articulation(num_envs: int = 4, num_bodies: int = 1):
    default_mass = torch.arange(
        1, num_envs * num_bodies + 1, dtype=torch.float32
    ).reshape(num_envs, num_bodies) + 10.0
    default_inertia = torch.zeros(num_envs, num_bodies, 9)
    default_inertia[..., 0] = 1.0
    default_inertia[..., 4] = 2.0
    default_inertia[..., 8] = 3.0
    root = _FakeRootPhysXView(default_mass, default_inertia)
    return SimpleNamespace(
        root_physx_view=root,
        data=SimpleNamespace(
            default_mass=default_mass.clone(),
            default_inertia=default_inertia.clone(),
        ),
        num_bodies=num_bodies,
    )


def test_mass_is_bounded_and_inertia_uses_identical_scale():
    asset = _fake_articulation()
    result = randomize_articulation_mass(asset, torch.tensor([1, 3]))

    assert result.scale.shape == (2, 1)
    assert bool((result.scale >= 0.95).all())
    assert bool((result.scale <= 1.05).all())
    expected_mass = asset.data.default_mass[[1, 3]] * result.scale
    expected_inertia = asset.data.default_inertia[[1, 3]] * result.scale.unsqueeze(-1)
    torch.testing.assert_close(result.masses, expected_mass)
    torch.testing.assert_close(result.inertias, expected_inertia)
    torch.testing.assert_close(asset.root_physx_view.masses[[1, 3]], expected_mass)
    torch.testing.assert_close(asset.root_physx_view.inertias[[1, 3]], expected_inertia)
    torch.testing.assert_close(
        asset.root_physx_view.masses[[0, 2]], asset.data.default_mass[[0, 2]]
    )


def test_repeated_reset_uses_nominal_values_not_current_values():
    asset = _fake_articulation()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    first = randomize_articulation_mass(asset, [0, 2], generator=generator)

    generator.manual_seed(123)
    second = randomize_articulation_mass(asset, [0, 2], generator=generator)

    torch.testing.assert_close(second.scale, first.scale)
    torch.testing.assert_close(second.masses, first.masses)
    torch.testing.assert_close(second.inertias, first.inertias)


def test_body_subset_preserves_unselected_body_properties():
    asset = _fake_articulation(num_envs=3, num_bodies=2)
    before_mass = asset.root_physx_view.masses.clone()
    before_inertia = asset.root_physx_view.inertias.clone()
    randomize_articulation_mass(asset, [0, 2], body_ids=[1])
    torch.testing.assert_close(asset.root_physx_view.masses[:, 0], before_mass[:, 0])
    torch.testing.assert_close(asset.root_physx_view.inertias[:, 0], before_inertia[:, 0])


def test_empty_selection_does_not_call_physx_setters():
    asset = _fake_articulation()
    result = randomize_articulation_mass(asset, [])
    assert result.scale.shape == (0, 1)
    assert asset.root_physx_view.mass_indices is None
    assert asset.root_physx_view.inertia_indices is None


def test_bad_range_and_duplicate_indices_are_rejected():
    asset = _fake_articulation()
    with pytest.raises(ValueError, match="relative_range"):
        randomize_articulation_mass(asset, [0], relative_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="duplicate"):
        randomize_articulation_mass(asset, [0, 0])
