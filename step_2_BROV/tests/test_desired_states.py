"""CPU tests for the pure-Torch BROV desired-state contracts."""

from __future__ import annotations

import math

import pytest
import torch

from step_2_BROV.envs.desired_states import (
    DeployAttitudeMode,
    DeployTransitionMode,
    DeployV2Config,
    DeployV2Scheduler,
    DeployV3Config,
    DeployV3Scheduler,
    DeployV6Config,
    DeployV6Scheduler,
    DeployVelocityMode,
    PaperReferenceBatch,
    _heading_from_direction,
)


def _quat_dot_abs(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left * right).sum(dim=-1).abs()


def _quat_geodesic(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    dot = _quat_dot_abs(left, right).clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def test_paper_reference_has_exact_constant_half_meter_speed() -> None:
    refs = PaperReferenceBatch(4096, seed=7)
    refs.reset()

    velocity_zero, _ = refs.sample(0.0)
    velocity_late, _ = refs.sample(4.9)

    torch.testing.assert_close(
        torch.linalg.vector_norm(velocity_zero, dim=-1),
        torch.full((4096,), 0.5),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(velocity_late, velocity_zero, atol=0.0, rtol=0.0)


def test_paper_velocity_sampling_is_uniform_on_s2() -> None:
    refs = PaperReferenceBatch(30_000, dtype=torch.float64, seed=123)
    refs.reset()
    direction = refs.velocity_body / 0.5

    # For uniform S², E[x]=0 and E[xxᵀ]=I/3.  These loose deterministic
    # statistical bounds catch axis-biased samplers without flaky tail tests.
    assert torch.max(direction.mean(dim=0).abs()).item() < 0.015
    second_moment = direction.T @ direction / direction.shape[0]
    torch.testing.assert_close(
        second_moment,
        torch.eye(3, dtype=torch.float64) / 3.0,
        atol=0.012,
        rtol=0.0,
    )


def test_frenet_frame_is_analytic_orthonormal_and_right_handed() -> None:
    refs = PaperReferenceBatch(1, dtype=torch.float64)
    time = torch.linspace(0.0, 5.0, 501, dtype=torch.float64)
    frame = refs.frenet_frame(time)

    identity = frame.transpose(-1, -2) @ frame
    torch.testing.assert_close(
        identity,
        torch.eye(3, dtype=torch.float64).expand_as(identity),
        atol=2.0e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.det(frame),
        torch.ones(time.numel(), dtype=torch.float64),
        atol=2.0e-12,
        rtol=0.0,
    )

    # The first frame column must be Eq. (9)'s normalized velocity exactly.
    phase = 0.2 * time
    eq9_velocity = torch.stack(
        (torch.full_like(time, 0.5), 0.5 * torch.sin(phase), 0.3 * torch.cos(phase)),
        dim=-1,
    )
    expected_tangent = eq9_velocity / torch.linalg.vector_norm(
        eq9_velocity, dim=-1, keepdim=True
    )
    torch.testing.assert_close(frame[..., 0], expected_tangent, atol=2.0e-12, rtol=0.0)


def test_paper_attitude_starts_at_random_initial_and_stays_continuous() -> None:
    refs = PaperReferenceBatch(64, dtype=torch.float64, seed=91)
    refs.reset()
    _, at_zero = refs.sample(0.0)

    torch.testing.assert_close(
        _quat_dot_abs(at_zero, refs.initial_quaternion),
        torch.ones(64, dtype=torch.float64),
        atol=2.0e-12,
        rtol=0.0,
    )

    samples = []
    for time in torch.linspace(0.0, 5.0, 251, dtype=torch.float64):
        _, quaternion = refs.sample(time)
        samples.append(quaternion)
    trajectory = torch.stack(samples, dim=0)
    adjacent_dot = (trajectory[1:] * trajectory[:-1]).sum(dim=-1)

    assert torch.min(adjacent_dot).item() > 0.999
    assert torch.max(_quat_geodesic(trajectory[-1], trajectory[0])).item() > 0.05
    torch.testing.assert_close(
        torch.linalg.vector_norm(trajectory, dim=-1),
        torch.ones_like(trajectory[..., 0]),
        atol=2.0e-12,
        rtol=0.0,
    )


def test_paper_and_deploy_sampling_are_reproducible_without_global_rng() -> None:
    torch.manual_seed(999)
    paper_a = PaperReferenceBatch(33, seed=42)
    paper_a.reset()
    deploy_a = DeployV2Scheduler(33, seed=42)
    deploy_a.reset()

    # Perturb the process-global RNG: the modules must be unaffected because
    # they own fixed-seed CPU generators.
    _ = torch.rand(1000)
    paper_b = PaperReferenceBatch(33, seed=42)
    paper_b.reset()
    deploy_b = DeployV2Scheduler(33, seed=42)
    deploy_b.reset()

    torch.testing.assert_close(paper_a.velocity_body, paper_b.velocity_body)
    torch.testing.assert_close(paper_a.initial_quaternion, paper_b.initial_quaternion)
    torch.testing.assert_close(deploy_a.command_before, deploy_b.command_before)
    torch.testing.assert_close(deploy_a.command_after, deploy_b.command_after)
    torch.testing.assert_close(deploy_a.transition_time_s, deploy_b.transition_time_s)
    torch.testing.assert_close(deploy_a.initial_quaternion, deploy_b.initial_quaternion)
    torch.testing.assert_close(deploy_a.attitude_mode, deploy_b.attitude_mode)
    torch.testing.assert_close(deploy_a.transition_mode, deploy_b.transition_mode)
    torch.testing.assert_close(deploy_a.is_reversal, deploy_b.is_reversal)


def test_partial_reset_does_not_mutate_other_environments() -> None:
    refs = PaperReferenceBatch(8, seed=4)
    refs.reset()
    untouched_velocity = refs.velocity_body[4:].clone()
    untouched_quaternion = refs.initial_quaternion[4:].clone()
    refs.reset([0, 1, 2, 3])
    torch.testing.assert_close(refs.velocity_body[4:], untouched_velocity)
    torch.testing.assert_close(refs.initial_quaternion[4:], untouched_quaternion)

    scheduler = DeployV2Scheduler(8, seed=4)
    scheduler.reset()
    untouched_before = scheduler.command_before[4:].clone()
    untouched_after = scheduler.command_after[4:].clone()
    untouched_time = scheduler.transition_time_s[4:].clone()
    scheduler.reset([0, 1, 2, 3])
    torch.testing.assert_close(scheduler.command_before[4:], untouched_before)
    torch.testing.assert_close(scheduler.command_after[4:], untouched_after)
    torch.testing.assert_close(scheduler.transition_time_s[4:], untouched_time)


def test_deploy_v2_balances_velocity_and_attitude_modes() -> None:
    scheduler = DeployV2Scheduler(300, seed=12)
    scheduler.reset()

    velocity_counts = torch.bincount(scheduler.post_speed_bin, minlength=3)
    attitude_counts = torch.bincount(scheduler.attitude_mode, minlength=3)
    transition_counts = torch.bincount(scheduler.transition_mode, minlength=3)
    torch.testing.assert_close(velocity_counts, torch.full((3,), 100, dtype=torch.long))
    torch.testing.assert_close(attitude_counts, torch.full((3,), 100, dtype=torch.long))
    torch.testing.assert_close(transition_counts, torch.full((3,), 100, dtype=torch.long))

    after_speed = torch.linalg.vector_norm(scheduler.command_after, dim=-1)
    expected_speed = torch.tensor([0.0, 0.1, 0.5])[scheduler.post_speed_bin]
    torch.testing.assert_close(after_speed, expected_speed, atol=1.0e-6, rtol=0.0)
    assert scheduler.transition_time_s.min().item() >= 2.0
    assert scheduler.transition_time_s.max().item() < 3.0


def test_deploy_v2_has_exactly_one_command_transition_per_episode() -> None:
    scheduler = DeployV2Scheduler(96, seed=19)
    scheduler.reset()
    time = torch.arange(0.0, 5.0001, 0.01)
    commands = torch.stack([scheduler.sample(float(t)).velocity_body for t in time], dim=0)
    changed = torch.linalg.vector_norm(commands[1:] - commands[:-1], dim=-1) > 1.0e-8
    torch.testing.assert_close(changed.sum(dim=0), torch.ones(96, dtype=torch.long))

    # The edge mask is also emitted exactly once at regular policy cadence.
    policy_time = torch.arange(0.0, 5.0001, scheduler.config.policy_dt_s)
    edge = torch.stack([scheduler.sample(float(t)).transition_mask for t in policy_time], dim=0)
    torch.testing.assert_close(edge.sum(dim=0), torch.ones(96, dtype=torch.long))


def test_exact_reversal_pairs_velocity_and_180_degree_attitude_step() -> None:
    config = DeployV2Config(exact_reversal=True)
    num_envs = 90
    scheduler = DeployV2Scheduler(num_envs, seed=27, config=config)
    scheduler.reset()
    nonzero = scheduler.post_speed_bin != int(DeployVelocityMode.HOLD)
    reversal = scheduler.is_reversal
    restart = scheduler.transition_mode == int(DeployTransitionMode.RESTART)
    stop = scheduler.transition_mode == int(DeployTransitionMode.STOP)

    torch.testing.assert_close(
        scheduler.command_before[reversal],
        -scheduler.command_after[reversal],
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        scheduler.command_before[restart],
        torch.zeros_like(scheduler.command_before[restart]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        scheduler.command_after[stop],
        torch.zeros_like(scheduler.command_after[stop]),
        atol=0.0,
        rtol=0.0,
    )
    # Every non-zero speed bin is split evenly between restart and reversal.
    for mode in (DeployVelocityMode.LOW_0P1, DeployVelocityMode.CRUISE_0P5):
        in_bin = scheduler.post_speed_bin == int(mode)
        reversal_count = int((reversal & in_bin).sum())
        restart_count = int((restart & in_bin).sum())
        assert abs(reversal_count - restart_count) <= 1
    assert bool((stop | restart | reversal).all())
    assert bool((reversal <= nonzero).all())
    assert bool(
        (
            scheduler.attitude_mode[reversal]
            == int(DeployAttitudeMode.RUNTIME_YAW_180_STEP)
        ).all()
    )

    before_time = scheduler.transition_time_s - 1.0e-4
    after_time = scheduler.transition_time_s + 1.0e-4
    before = scheduler.sample(before_time, previous_time_s=before_time - 0.01)
    after = scheduler.sample(after_time, previous_time_s=before_time)
    angle = _quat_geodesic(before.desired_quaternion[reversal], after.desired_quaternion[reversal])
    torch.testing.assert_close(
        angle,
        torch.full_like(angle, math.pi),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert bool(after.transition_mask.all())
    assert bool((~before.transition_mask).all())
    torch.testing.assert_close(after.reversal_mask, reversal)
    torch.testing.assert_close(after.transition_mode, scheduler.transition_mode)
    assert after.velocity_mode.shape == (num_envs,)
    assert after.attitude_mode.shape == (num_envs,)


def test_exact_reversal_can_be_disabled_without_losing_balanced_speed_bins() -> None:
    scheduler = DeployV2Scheduler(
        60,
        seed=31,
        config=DeployV2Config(exact_reversal=False),
    )
    scheduler.reset()
    nonzero = scheduler.post_speed_bin != int(DeployVelocityMode.HOLD)

    assert not bool(scheduler.is_reversal.any())
    torch.testing.assert_close(
        scheduler.command_before[nonzero],
        torch.zeros_like(scheduler.command_before[nonzero]),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.bincount(scheduler.post_speed_bin, minlength=3),
        torch.full((3,), 20, dtype=torch.long),
    )


def test_deploy_attitude_modes_have_expected_continuity_contract() -> None:
    num_envs = 90
    scheduler = DeployV2Scheduler(num_envs, seed=8)
    scheduler.reset()
    modes = scheduler.attitude_mode
    time = torch.linspace(0.0, 5.0, 251)
    quaternions = torch.stack(
        [scheduler.sample(float(value)).desired_quaternion for value in time], dim=0
    )

    frenet = modes == int(DeployAttitudeMode.FRENET_CONTINUOUS)
    hold = modes == int(DeployAttitudeMode.RUNTIME_HOLD)
    step = modes == int(DeployAttitudeMode.RUNTIME_YAW_180_STEP)

    frenet_dot = (quaternions[1:, frenet] * quaternions[:-1, frenet]).sum(dim=-1)
    assert torch.min(frenet_dot).item() > 0.999
    torch.testing.assert_close(
        quaternions[:, hold],
        quaternions[:1, hold].expand_as(quaternions[:, hold]),
        atol=0.0,
        rtol=0.0,
    )

    for env_id in torch.nonzero(step, as_tuple=False).flatten().tolist():
        transition = scheduler.transition_time_s[env_id]
        before_index = int(torch.nonzero(time < transition, as_tuple=False)[-1])
        after_index = before_index + 1
        angle = _quat_geodesic(
            quaternions[before_index, env_id], quaternions[after_index, env_id]
        )
        assert angle.item() == pytest.approx(math.pi, abs=2.0e-6)

    assert quaternions.shape == (251, num_envs, 4)


def test_reset_and_config_validation_fail_closed_outside_hot_path() -> None:
    paper = PaperReferenceBatch(2)
    with pytest.raises(ValueError):
        paper.reset([0], initial_quaternion=torch.zeros(4))
    with pytest.raises(IndexError):
        paper.reset([2])
    with pytest.raises(ValueError):
        paper.reset([0, 0])

    with pytest.raises(ValueError):
        DeployV2Config(transition_time_range_s=(3.0, 2.0))
    with pytest.raises(ValueError):
        DeployV2Config(speed_bins_mps=(0.0, 0.2, 0.5))


def _quat_from_euler_xyz_reference(
    roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor
) -> torch.Tensor:
    """Independent transcription of isaaclab.utils.math.quat_from_euler_xyz
    (verified 2026-08-18 against /workspace/isaaclab/source/isaaclab/
    isaaclab/utils/math.py:274-298 inside the isaac-lab-base container --
    not importable here without a full AppLauncher/Kit boot, since
    isaaclab.utils.__init__ pulls in pxr/USD). Kept as a full roll/pitch/yaw
    formula (not specialized to roll=0) so it is a genuinely independent
    check on ``_heading_from_direction``'s closed-form specialization, not
    just the same three lines copied twice."""

    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack((qw, qx, qy, qz), dim=-1)


def _rotate_by_quat(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate ``(N,3)`` by ``(N,4)`` wxyz quaternions, independent of isaaclab."""

    w = quaternion[:, 0:1]
    axis = quaternion[:, 1:4]
    return (
        vector * (2.0 * w * w - 1.0)
        + 2.0 * axis * (axis * vector).sum(dim=-1, keepdim=True)
        + 2.0 * w * torch.cross(axis, vector, dim=-1)
    )


def test_heading_from_direction_points_the_nose_at_the_requested_direction() -> None:
    """The function's only contract: ``quat_apply(q, x_hat) == direction``.

    This is a frame-convention-free algebraic identity, which is exactly why
    it catches what the previous test could not. That test rebuilt ``expected``
    from the same ``pitch = asin(dz)`` the implementation used, so it agreed
    with a sign error that mirrored every non-horizontal heading (fixed
    2026-08-26; the correct expression is ``pitch = -asin(dz)``). A check that
    derives its expectation from the implementation's own formula can only ever
    confirm that the code matches itself.
    """

    torch.manual_seed(0)
    directions = torch.randn(2048, 3)
    directions[0] = torch.tensor([0.0, 0.0, 1.0])    # straight up
    directions[1] = torch.tensor([0.0, 0.0, -1.0])   # straight down
    directions[2] = torch.tensor([1.0, 0.0, 0.0])    # identity case
    directions[3] = torch.tensor([1.0, 0.0, 1.0])    # forward and up
    directions[4] = torch.tensor([1.0, 0.0, -1.0])   # forward and down
    directions = directions / directions.norm(dim=-1, keepdim=True)

    heading = _heading_from_direction(directions)
    torch.testing.assert_close(heading.norm(dim=-1), torch.ones(2048), atol=1.0e-5, rtol=0.0)

    forward = torch.tensor([1.0, 0.0, 0.0]).expand_as(directions)
    nose = _rotate_by_quat(heading, forward)
    torch.testing.assert_close(nose, directions, atol=1.0e-5, rtol=0.0)

    # roll = 0 means the body's lateral axis stays in the world horizontal plane.
    lateral = torch.tensor([0.0, 1.0, 0.0]).expand_as(directions)
    right = _rotate_by_quat(heading, lateral)
    torch.testing.assert_close(right[:, 2], torch.zeros(2048), atol=1.0e-5, rtol=0.0)

    identity_heading = _heading_from_direction(torch.tensor([[1.0, 0.0, 0.0]]))
    torch.testing.assert_close(
        identity_heading, torch.tensor([[1.0, 0.0, 0.0, 0.0]]), atol=1.0e-6, rtol=0.0
    )


def test_heading_from_direction_matches_the_general_euler_formula_at_roll_zero() -> None:
    """Closed-form specialization still equals the full roll/pitch/yaw formula.

    Complements the physical test above: that one pins the sign, this one pins
    the algebraic shortcut. Both are needed -- neither alone would have caught
    the 2026-08-26 sign error.
    """

    torch.manual_seed(0)
    directions = torch.randn(2048, 3)
    directions[0] = torch.tensor([0.0, 0.0, 1.0])   # pole case
    directions[1] = torch.tensor([0.0, 0.0, -1.0])  # pole case
    directions[2] = torch.tensor([1.0, 0.0, 0.0])   # identity case

    heading = _heading_from_direction(directions)

    d = directions / directions.norm(dim=-1, keepdim=True)
    yaw = torch.atan2(d[:, 1], d[:, 0])
    pitch = -torch.asin(d[:, 2].clamp(-1.0, 1.0))
    expected = _quat_from_euler_xyz_reference(torch.zeros_like(yaw), pitch, yaw)
    torch.testing.assert_close(heading, expected, atol=1.0e-6, rtol=0.0)


def test_deploy_v6_zero_coupling_probability_matches_deploy_v3() -> None:
    """With coupling probability 0, DeployV6Scheduler must reduce exactly to
    DeployV3Scheduler for the same seed -- the extra RNG draw it performs
    (coupled_roll) happens strictly after every value that feeds
    leg_starts/leg_velocity/leg_quaternion, so it must not perturb them."""

    v3 = DeployV3Scheduler(64, seed=101, config=DeployV3Config(max_legs=12))
    v3.reset()
    v6 = DeployV6Scheduler(
        64,
        seed=101,
        config=DeployV6Config(max_legs=12, los_coupled_retarget_probability=0.0),
    )
    v6.reset()

    torch.testing.assert_close(v3.leg_starts, v6.leg_starts, atol=0.0, rtol=0.0)
    torch.testing.assert_close(v3.leg_velocity, v6.leg_velocity, atol=0.0, rtol=0.0)
    torch.testing.assert_close(v3.leg_quaternion, v6.leg_quaternion, atol=0.0, rtol=0.0)
    assert not bool(v6.leg_los_coupled.any())


def test_deploy_v6_coupled_legs_are_los_heading_of_their_own_velocity() -> None:
    scheduler = DeployV6Scheduler(
        512,
        seed=17,
        config=DeployV6Config(
            episode_length_s=10.0, max_legs=16, los_coupled_retarget_probability=1.0
        ),
    )
    scheduler.reset()

    # probability=1.0: every retargeted (non-kept) leg beyond leg 0 is
    # coupled, since new_attitude_probability defaults to 0.5 (only legs
    # that get a *new* target are eligible for coupling at all).
    assert bool(scheduler.leg_los_coupled[:, 1:].any())
    coupled = scheduler.leg_los_coupled[:, 1:]
    # leg_velocity is direction*speed, and the scheduler couples off the
    # pre-scale unit direction -- reconstructing "direction" from velocity
    # here only works where speed != 0 (the HOLD bin's zero velocity is a
    # degenerate/undefined direction to normalize), so exclude it.
    velocity = scheduler.leg_velocity[:, 1:]
    moving = velocity.norm(dim=-1) > 1.0e-6
    expected = _heading_from_direction(velocity)
    mask = coupled & moving
    torch.testing.assert_close(
        scheduler.leg_quaternion[:, 1:][mask], expected[mask], atol=1.0e-5, rtol=0.0
    )
    assert bool(mask.any())


def test_deploy_v6_coupling_probability_is_approximately_respected() -> None:
    num_envs = 4096
    scheduler = DeployV6Scheduler(
        num_envs,
        seed=23,
        config=DeployV6Config(
            episode_length_s=10.0,
            max_legs=16,
            new_attitude_probability=1.0,
            los_coupled_retarget_probability=0.3,
        ),
    )
    scheduler.reset()
    # new_attitude_probability=1.0 makes every leg beyond 0 eligible, so the
    # coupled fraction should track los_coupled_retarget_probability directly.
    fraction = scheduler.leg_los_coupled[:, 1:].float().mean().item()
    assert fraction == pytest.approx(0.3, abs=0.03)


def test_deploy_v6_sample_reports_world_direction_and_coupling_mask() -> None:
    scheduler = DeployV6Scheduler(
        32,
        seed=5,
        config=DeployV6Config(
            episode_length_s=10.0, max_legs=8, los_coupled_retarget_probability=1.0
        ),
    )
    scheduler.reset()
    sample = scheduler.sample(0.0)
    assert sample.los_coupled_mask.shape == (32,)
    assert sample.world_direction.shape == (32, 3)
    torch.testing.assert_close(
        sample.world_direction, scheduler.leg_velocity[:, 0], atol=0.0, rtol=0.0
    )


def test_deploy_v6_config_validates_coupling_probability() -> None:
    with pytest.raises(ValueError):
        DeployV6Config(los_coupled_retarget_probability=1.5)
    with pytest.raises(ValueError):
        DeployV6Config(los_coupled_retarget_probability=-0.1)
