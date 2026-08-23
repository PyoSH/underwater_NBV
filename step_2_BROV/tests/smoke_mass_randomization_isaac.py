"""Manual headless smoke test against the real IsaacLab BROV articulation.

Run inside the IsaacLab container from ``step_2_BROV``.  This is intentionally
not named ``test_*.py`` because importing Isaac modules requires AppLauncher to
start first.
"""

from __future__ import annotations

from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch

from envs.vel_env import BROVVelEnv
from envs.vel_env_cfg import BROVVelEnvCfg
from robots.dynamics.brov2.mass_randomization import randomize_articulation_mass


def main() -> None:
    cfg = BROVVelEnvCfg()
    cfg.scene.num_envs = 4
    cfg.debug_vis = False
    env = BROVVelEnv(cfg)
    try:
        asset = env._robot
        default_mass = asset.data.default_mass.clone()
        default_inertia = asset.data.default_inertia.clone()
        env_ids = torch.tensor([0, 2], dtype=torch.long)

        generator = torch.Generator(device=default_mass.device)
        generator.manual_seed(20260817)
        result = randomize_articulation_mass(asset, env_ids, generator=generator)
        first_mass = result.masses.clone()
        first_inertia = result.inertias.clone()

        # Repeating a reset with the same draw must produce the same physical
        # properties, not multiply the previous randomized values again.
        generator.manual_seed(20260817)
        repeated = randomize_articulation_mass(asset, env_ids, generator=generator)
        torch.testing.assert_close(repeated.masses, first_mass)
        torch.testing.assert_close(repeated.inertias, first_inertia)

        actual_mass = asset.root_physx_view.get_masses()
        actual_inertia = asset.root_physx_view.get_inertias()
        other_ids = torch.tensor([1, 3], dtype=torch.long)

        torch.testing.assert_close(
            actual_mass[env_ids], default_mass[env_ids] * repeated.scale
        )
        torch.testing.assert_close(
            actual_inertia[env_ids],
            default_inertia[env_ids] * repeated.scale.unsqueeze(-1),
        )
        torch.testing.assert_close(actual_mass[other_ids], default_mass[other_ids])

        print(
            {
                "default_mass": default_mass[:, 0].tolist(),
                "scale": repeated.scale[:, 0].tolist(),
                "actual_mass": actual_mass[:, 0].tolist(),
                "mass_device": str(actual_mass.device),
                "default_device": str(default_mass.device),
                "inertia_shape": list(actual_inertia.shape),
            },
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
