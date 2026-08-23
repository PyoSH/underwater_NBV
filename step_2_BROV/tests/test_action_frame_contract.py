"""Golden tests for the policy-FLU to SNAME-thruster action contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


STEP_2_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STEP_2_ROOT.parent
for path in (STEP_2_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from action_frame_contract import (  # noqa: E402
    EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1,
    LEGACY_MODEL_299_NO_T6,
    allocate_sname_frd_wrench,
    build_policy_action_to_sname_frd_multiplier,
    policy_action_to_sname_frd_wrench,
    policy_flu_zup_to_sname_frd,
    policy_wrench_to_sname_frd,
    sname_frd_to_policy_flu_zup,
    sname_frd_to_policy_wrench,
    thruster_forces_to_sname_frd_wrench,
)
from robots.dynamics.brov2.params import (  # noqa: E402
    load_brov2_yaml,
    thruster_pos_dir_ned,
)
from robots.dynamics.brov2.thruster import (  # noqa: E402
    build_allocation_matrix,
)


class ActionFrameContractGoldenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        parameters = load_brov2_yaml()
        position, direction = thruster_pos_dir_ned(parameters)
        cls.allocation = build_allocation_matrix(
            torch.tensor(position, dtype=torch.float64),
            torch.tensor(direction, dtype=torch.float64),
        )
        cls.allocation_pinv = torch.linalg.pinv(cls.allocation)
        cls.wrench_scale = torch.tensor(
            [85.0, 85.0, 120.0, 26.0, 14.0, 22.0], dtype=torch.float64
        )

    def assertTensorClose(self, actual, expected, *, atol=1.0e-10) -> None:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)

    def test_positive_and_negative_six_axis_basis_matches_t6(self) -> None:
        positive = torch.eye(6, dtype=torch.float64)
        policy_basis = torch.cat((positive, -positive), dim=0)
        expected_t6 = torch.tensor(
            [1.0, -1.0, -1.0, 1.0, -1.0, -1.0], dtype=torch.float64
        )
        expected = policy_basis * expected_t6

        mapped = policy_flu_zup_to_sname_frd(policy_basis)

        self.assertTensorClose(mapped, expected)
        self.assertTensorClose(sname_frd_to_policy_flu_zup(mapped), policy_basis)

    def test_explicit_contract_round_trip_for_batched_values(self) -> None:
        policy_wrench = torch.tensor(
            [
                [12.0, -3.0, 7.5, -0.25, 4.0, -2.0],
                [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0],
            ],
            dtype=torch.float32,
        )
        sname = policy_wrench_to_sname_frd(
            policy_wrench, contract=EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1
        )
        recovered = sname_frd_to_policy_wrench(
            sname, contract=EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1
        )

        self.assertTensorClose(recovered, policy_wrench, atol=0.0)

    def test_cached_hot_loop_multiplier_matches_checked_adapter(self) -> None:
        actions = torch.tensor(
            [[0.25, -0.5, 0.75, -1.0, 0.125, -0.375]], dtype=torch.float64
        )
        for contract in (
            EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1,
            LEGACY_MODEL_299_NO_T6,
        ):
            with self.subTest(contract=contract):
                multiplier = build_policy_action_to_sname_frd_multiplier(
                    self.wrench_scale,
                    contract=contract,
                    dtype=actions.dtype,
                    device=actions.device,
                )
                expected = policy_action_to_sname_frd_wrench(
                    actions, self.wrench_scale, contract=contract
                )
                self.assertTensorClose(actions * multiplier, expected, atol=0.0)

    def test_current_allocation_recovers_all_explicit_axis_wrenches(self) -> None:
        # Low enough to isolate the ideal linear B/B+ contract from T200 clamps.
        actions = 0.1 * torch.cat(
            (torch.eye(6, dtype=torch.float64), -torch.eye(6, dtype=torch.float64))
        )
        requested_sname = policy_action_to_sname_frd_wrench(
            actions,
            self.wrench_scale,
            contract=EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1,
        )
        thruster_forces = allocate_sname_frd_wrench(
            requested_sname, self.allocation_pinv
        )
        achieved_sname = thruster_forces_to_sname_frd_wrench(
            thruster_forces, self.allocation
        )
        achieved_policy = sname_frd_to_policy_flu_zup(achieved_sname)

        self.assertEqual(int(torch.linalg.matrix_rank(self.allocation)), 6)
        self.assertTensorClose(achieved_sname, requested_sname, atol=1.0e-9)
        self.assertTensorClose(
            achieved_policy, actions * self.wrench_scale, atol=1.0e-9
        )

    def test_named_model_299_path_preserves_historical_no_t6_mapping(self) -> None:
        actions = 0.1 * torch.cat(
            (torch.eye(6, dtype=torch.float64), -torch.eye(6, dtype=torch.float64))
        )
        legacy_requested_sname = policy_action_to_sname_frd_wrench(
            actions,
            self.wrench_scale,
            contract=LEGACY_MODEL_299_NO_T6,
        )
        thruster_forces = allocate_sname_frd_wrench(
            legacy_requested_sname, self.allocation_pinv
        )
        achieved_sname = thruster_forces_to_sname_frd_wrench(
            thruster_forces, self.allocation
        )

        # This identity is the frozen model_299 training/deployment behavior.
        self.assertTensorClose(
            legacy_requested_sname, actions * self.wrench_scale, atol=0.0
        )
        self.assertTensorClose(achieved_sname, legacy_requested_sname, atol=1.0e-9)
        self.assertTensorClose(
            sname_frd_to_policy_wrench(
                achieved_sname, contract=LEGACY_MODEL_299_NO_T6
            ),
            actions * self.wrench_scale,
            atol=1.0e-9,
        )

        # The physically expressed FLU/Z-up wrench still carries the historical
        # sway/heave/pitch/yaw sign reversal.  The compatibility adapter must not
        # be mistaken for the corrected physical frame conversion.
        expected_physical_policy = policy_flu_zup_to_sname_frd(
            actions * self.wrench_scale
        )
        self.assertTensorClose(
            sname_frd_to_policy_flu_zup(achieved_sname),
            expected_physical_policy,
            atol=1.0e-9,
        )

    def test_zero_and_finite_batch_stay_finite(self) -> None:
        values = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0e-8, -1.0e5, 3.5, -2.25, 9.0e3, -4.0],
            ],
            dtype=torch.float32,
        )
        for contract in (
            EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1,
            LEGACY_MODEL_299_NO_T6,
        ):
            result = policy_wrench_to_sname_frd(values, contract=contract)
            self.assertTrue(bool(torch.isfinite(result).all()))
            self.assertTensorClose(result[0], torch.zeros(6), atol=0.0)

    def test_nonfinite_and_unknown_contract_fail_closed(self) -> None:
        bad = torch.zeros(6, dtype=torch.float32)
        bad[2] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            policy_flu_zup_to_sname_frd(bad)

        bad = torch.zeros(6, dtype=torch.float32)
        bad[4] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            policy_wrench_to_sname_frd(
                bad, contract=EXPLICIT_FLU_ZUP_TO_SNAME_FRD_V1
            )

        with self.assertRaisesRegex(ValueError, "unknown action-frame contract"):
            policy_wrench_to_sname_frd(
                torch.zeros(6, dtype=torch.float32), contract="implicit"
            )


if __name__ == "__main__":
    unittest.main()
