"""LOS 수평 구간/미션 완료 후 depth 및 position hold 회귀 테스트."""

import torch

from deploy.guidance_standalone import LOSGuidance


def _guidance():
    waypoints = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, -0.5], [1.0, 0.0, -0.5]]])
    guidance = LOSGuidance(
        waypoints,
        "cpu",
        cruise_speed=0.1,
        reach_threshold=0.15,
        heading_mode="straight",
        loop=False,
        depth_hold_kp=0.8,
        depth_speed_limit=0.1,
    )
    guidance.reset(torch.tensor([0]), initial_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    return guidance


def test_depth_hold_remains_active_after_switch_to_horizontal_segment():
    guidance = _guidance()
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # 첫 depth waypoint의 0.1 m 이내 → 수평 구간으로 전환된다.
    velocity, _ = guidance.compute(torch.tensor([[0.0, 0.0, -0.4]]), q)
    assert int(guidance._wp_idx[0]) == 1
    # NED Z 음수 속도는 상승. 남은 0.1 m 깊이 오차에 대해 -0.08 m/s.
    assert torch.allclose(velocity[0, 2], torch.tensor(-0.08), atol=1e-6)


def test_terminal_completion_continues_position_and_depth_hold():
    guidance = _guidance()
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    guidance.compute(torch.tensor([[0.0, 0.0, -0.5]]), q)  # idx 0 → 1
    guidance.compute(torch.tensor([[0.95, 0.0, -0.5]]), q)  # final 도달
    assert bool(guidance.mission_complete[0])

    # 이후 0.3 m 침강하고 X도 이탈하면 final waypoint로 복귀 속도가 생겨야 한다.
    velocity, _ = guidance.compute(torch.tensor([[0.8, 0.0, -0.2]]), q)
    assert velocity[0, 0] > 0.0
    assert torch.allclose(velocity[0, 2], torch.tensor(-0.1), atol=1e-6)
