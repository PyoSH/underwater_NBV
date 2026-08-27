"""Breivik-Fossen 3D LOS 검증 (학습/평가측, IsaacLab Z-up).

배포 포팅본(`deploy/test_guidance_los_bf.py`)과 짝을 이룬다. 두 파일은 같은 법칙을
서로 다른 frame 규약에서 검사한다 — 배포측은 NED, 여기는 Z-up.
"""

import math

import pytest
import torch

# los_guidance.py는 isaaclab.utils.math에 의존하고, 그 패키지는 import 시점에 pxr을
# 끌어오므로 SimulationApp이 떠 있어야 한다. 컨테이너 밖(호스트 CPU pytest)에서는
# 통째로 skip한다 — 같은 법칙의 순수 torch 검증은 deploy/test_guidance_los_bf.py가
# 별도로 담당하므로 호스트 CI 커버리지가 0이 되지는 않는다.
pytest.importorskip("isaacsim", reason="IsaacLab 컨테이너에서만 실행")
try:
    import isaaclab.utils.math as mu
except ModuleNotFoundError:                       # pxr 미로드 상태
    from isaacsim import SimulationApp
    _APP = SimulationApp({"headless": True})
    import isaaclab.utils.math as mu

from guidance.los_guidance import LOSGuidance


_Q_ID = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
# 수평 → 상승 → 하강. 앙각 부호를 양방향 모두 검사하기 위함.
_WP = torch.tensor([[[0.0, 0.0, 5.0], [5.0, 0.0, 5.0], [8.0, 3.0, 8.0], [11.0, 3.0, 2.0]]])
_U = 0.5


def _los(**kw):
    return LOSGuidance(_WP, "cpu", lookahead_dist=1.0, cruise_speed=_U, **kw)


def _bf_reference(p, cur, nxt, dh=1.0, dv=1.0):
    """구현과 독립적인 폐형 참조 — Breivik & Fossen (2005) Sec. IV를 직접 옮긴 것."""
    seg = nxt - cur
    sd = seg / seg.norm()
    chi_p = math.atan2(float(sd[1]), float(sd[0]))
    ups_p = math.atan2(float(sd[2]), float(sd[:2].norm()))
    cs, sn, cu, su = math.cos(chi_p), math.sin(chi_p), math.cos(ups_p), math.sin(ups_p)
    d = p - cur
    e = -sn * float(d[0]) + cs * float(d[1])
    h = -su * (cs * float(d[0]) + sn * float(d[1])) + cu * float(d[2])
    chi_d = chi_p + math.atan(-e / dh)
    ups_d = ups_p + math.atan(-h / dv)
    v = _U * torch.tensor([math.cos(chi_d) * math.cos(ups_d),
                           math.sin(chi_d) * math.cos(ups_d),
                           math.sin(ups_d)])
    return v, e, h


@pytest.mark.parametrize("seg", [0, 1, 2])
@pytest.mark.parametrize("pos", [[0.5, -0.8, 5.6], [2.0, 1.3, 3.9], [7.0, 2.0, 6.5]])
def test_matches_the_closed_form_breivik_fossen_law(seg, pos):
    los = _los()
    los._wp_idx[:] = seg
    p = torch.tensor(pos)
    v_b, _ = los.compute(p.unsqueeze(0), _Q_ID)
    want, _, _ = _bf_reference(p, _WP[0, seg], _WP[0, (seg + 1) % 4])
    assert torch.allclose(v_b[0], want, atol=1e-5)


@pytest.mark.parametrize("rpy", [(0.0, 0.0, 0.0), (0.3, -0.4, 1.1), (-1.2, 0.9, -2.5)])
def test_command_is_attitude_independent_in_the_world_frame(rpy):
    """자세를 어떻게 바꿔도 월드 프레임 명령은 같다 (완전구동 전제).

    body frame 명령 v_d^b는 당연히 자세에 따라 회전하므로, 월드로 되돌려 비교한다.
    """
    los = _los()
    q = mu.quat_from_euler_xyz(*[torch.tensor([v]) for v in rpy])
    p = torch.tensor([[2.0, 1.3, 3.9]])
    v_b, _ = los.compute(p, q)
    v_w = mu.quat_apply(q, v_b)
    want, _, _ = _bf_reference(p[0], _WP[0, 0], _WP[0, 1])
    assert torch.allclose(v_w[0], want, atol=1e-5)


def test_speed_magnitude_is_exactly_preserved():
    """||v_d|| = cruise_speed. 정책이 학습한 명령 크기는 이 값 하나뿐이다."""
    los = _los()
    for seg, pos in ((0, [2.0, 1.3, 3.9]), (1, [7.0, 2.0, 6.5]), (2, [9.5, 3.4, 5.0])):
        los._wp_idx[:] = seg
        v_b, _ = los.compute(torch.tensor([pos]), _Q_ID)
        assert torch.allclose(v_b.norm(dim=-1), torch.tensor([_U]), atol=1e-6)


@pytest.mark.parametrize("seg,pos", [(0, [2.0, 1.3, 3.9]), (2, [9.5, 3.4, 5.0])])
def test_desired_attitude_nose_points_along_the_desired_velocity(seg, pos):
    """q_d의 기수가 v_d^w를 정확히 향하고 roll=0.

    pitch = -υ_d 부호가 여기서 잡힌다. Z-up에서 v_z = +U sin(υ_d)인데 roll=0 자세의
    기수 Z 성분은 -sin(pitch)이므로 pitch = -υ_d다. 원논문(NED)은 pitch = +υ_d다.
    수식을 자기 자신과 비교하면 이 부호를 영원히 못 잡으므로 물리 조건으로 검사한다.
    """
    los = _los(heading_mode="align")
    los._wp_idx[:] = seg
    p = torch.tensor([pos])
    v_b, q_d = los.compute(p, _Q_ID)
    v_hat = v_b[0] / v_b[0].norm()
    nose = mu.quat_apply(q_d, torch.tensor([[1.0, 0.0, 0.0]]))[0]
    left = mu.quat_apply(q_d, torch.tensor([[0.0, 1.0, 0.0]]))[0]
    assert torch.allclose(nose, v_hat, atol=1e-5)
    assert abs(float(left[2])) < 1e-5          # roll = 0


def test_vertical_and_horizontal_corrections_do_not_compete():
    """수평 오차를 키워도 수직 보정이 잠식되지 않는다.

    이전 구현("lookahead 지점을 향하는 3D 벡터 정규화")의 실제 결함 — 두 보정이
    고정 크기 U를 두고 경쟁해서, 실측 로그에서 cross-track 1.9m / vertical 1.24m
    상태의 수직 성분이 BF 대비 32% 작았다.
    """
    los = _los()
    base = torch.tensor([[2.0, 0.0, 4.0]])                  # h = -1.0, e = 0
    v0, _ = los.compute(base, _Q_ID)
    for lateral in (0.5, 1.5, 3.0):
        los._wp_idx[:] = 0
        p = base.clone(); p[0, 1] = lateral                  # e만 키움
        v, _ = los.compute(p, _Q_ID)
        assert torch.allclose(v[0, 2], v0[0, 2], atol=1e-6)


def test_course_depends_on_cross_track_only_not_on_along_track_progress():
    """조향각은 (e, h)만의 함수이고 진행률 s와 무관하다."""
    los = _los()
    a, _ = los.compute(torch.tensor([[0.5, 0.7, 4.6]]), _Q_ID)
    los._wp_idx[:] = 0
    b, _ = los.compute(torch.tensor([[4.5, 0.7, 4.6]]), _Q_ID)
    assert torch.allclose(a, b, atol=1e-6)
