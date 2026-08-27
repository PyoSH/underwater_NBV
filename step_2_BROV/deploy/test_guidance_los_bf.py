"""Breivik-Fossen 3D LOS 회귀 테스트 (배포 포팅본).

`test_guidance_depth_hold.py`를 대체한다 — 이전 구현의 별도 depth-hold P 제어기는
BF의 독립 vertical-track 축(υ_d)이 대체했다. 여기서는 그 축이 실제로 동작하는지와,
이전 구현이 깨뜨렸던 두 성질(||v_d|| 보존, waypoint 근방 비특이성)을 검사한다.
"""

import math

import torch

from deploy.guidance_standalone import LOSGuidance


_Q_ID = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
# NED. [0,0,0] → 0.5m 하강 → 수평 1m 전진.
_WP = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, -0.5], [1.0, 0.0, -0.5]]])


def _guidance(**kw):
    g = LOSGuidance(
        _WP, "cpu", cruise_speed=0.1, reach_threshold=0.15,
        heading_mode="straight", loop=False, **kw,
    )
    g.reset(torch.tensor([0]), initial_quat=_Q_ID)
    return g


def test_vertical_track_error_is_corrected_on_a_horizontal_segment():
    """수평 세그먼트에서도 경로 위/아래 이탈이 독립적으로 복원된다."""
    g = _guidance(lookahead_vert=0.2)
    # 첫 depth waypoint의 0.15m 이내 → 수평 세그먼트로 전환.
    v, _ = g.compute(torch.tensor([[0.0, 0.0, -0.4]]), _Q_ID)
    assert int(g._wp_idx[0]) == 1
    # NED에서 로봇(z=-0.4)이 경로(z=-0.5)보다 얕다 → 더 깊이(-z) 가야 한다.
    # h = +0.1, υ_d = atan(-0.1/0.2), v_z = 0.1·sin(υ_d).
    expected = 0.1 * math.sin(math.atan(-0.1 / 0.2))
    assert torch.allclose(v[0, 2], torch.tensor(expected), atol=1e-6)
    assert v[0, 2] < 0.0


def test_speed_magnitude_is_exactly_preserved():
    """||v_d|| = cruise_speed. 정책이 학습한 명령 크기는 이 값 하나뿐이다.

    이전 구현은 정규화 이후 v_d_world[2]를 depth-hold 출력으로 덮어써서 이 성질을
    깨뜨렸다 — 수평/수직 오차가 동시에 있을 때 크기가 학습 분포를 벗어났다.
    """
    g = _guidance(lookahead_vert=0.2)
    for p in ([0.0, 0.0, -0.4], [0.3, 0.7, -0.2], [0.9, -1.4, -1.1]):
        v, _ = g.compute(torch.tensor([p]), _Q_ID)
        assert torch.allclose(v.norm(dim=-1), torch.tensor([0.1]), atol=1e-6)


def test_course_depends_on_cross_track_only_not_on_along_track_progress():
    """BF의 구조적 성질: 조향각은 e, h만의 함수이고 진행률 s와 무관하다.

    이전 구현은 lookahead '지점'을 세그먼트 끝에 clamp했기 때문에, 같은 cross-track
    오차라도 세그먼트 끝에 가까울수록 명령 방향이 경로 방향에서 waypoint 방향으로
    끌려갔다(끝에서는 |to_los| → 0까지 퇴화). BF에는 lookahead 지점이 없어
    구간 어디에 있든 같은 이탈량이면 같은 조향을 낸다.
    """
    g = _guidance(lookahead_vert=0.2)
    g._loop = True                      # terminal hold 개입 배제
    g._wp_idx[:] = 1                    # 세그먼트 [0,0,-0.5] → [1,0,-0.5]
    # 같은 e(+0.2), 같은 h(+0.1), 다른 along-track s
    a, _ = g.compute(torch.tensor([[0.1, 0.2, -0.4]]), _Q_ID)
    g._wp_idx[:] = 1
    b, _ = g.compute(torch.tensor([[0.9, 0.2, -0.4]]), _Q_ID)
    assert torch.allclose(a, b, atol=1e-6)
    assert torch.allclose(a.norm(dim=-1), torch.tensor([0.1]), atol=1e-6)


def test_terminal_completion_continues_position_hold():
    g = _guidance()
    g.compute(torch.tensor([[0.0, 0.0, -0.5]]), _Q_ID)    # idx 0 → 1
    g.compute(torch.tensor([[0.95, 0.0, -0.5]]), _Q_ID)   # final 도달
    assert bool(g.mission_complete[0])

    # 0.3m 부상 + X 이탈 → final waypoint로 3D 복귀 속도.
    # terminal_hold_kp=0.5, terminal_speed_limit=cruise_speed=0.1.
    err = torch.tensor([0.2, 0.0, -0.3])
    want = 0.5 * err
    want = want * min(0.1 / float(want.norm()), 1.0)
    v, _ = g.compute(torch.tensor([[0.8, 0.0, -0.2]]), _Q_ID)
    assert v[0, 0] > 0.0
    assert torch.allclose(v[0], want, atol=1e-6)
