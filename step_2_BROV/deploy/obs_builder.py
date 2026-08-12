"""
실기체 텔레메트리 → 16-dim 정책 관측 조립
=============================================
`envs/vel_env.py._get_observations()`와 정확히 같은 수식(q_e, v_e_b, ω_b, z_v, z_q)을
만들어야 하는데, 두 가지가 sim과 다르다:

1. **좌표계**: MAVLink는 NED(X=North,Y=East,Z=Down) world + SNAME 계열 body를 쓰고,
   학습된 정책은 IsaacLab의 Z-up world/body 관측으로 학습됨. world 쪽은 절대 방향이
   정책 동작에 영향이 없어(정책은 상대 오차만 봄) 그냥 NED를 그대로 "world"로 쓰고,
   body만 이 코드베이스 전체가 쓰는 `T3=diag(1,-1,-1)` 변환으로 Z-up으로 맞춘다.
   자세(쿼터니언)는 body만 바뀌므로 `q_zup = q_ned ⊗ Q_M`(Q_M=[0,1,0,0], X축 180도
   회전 — body 기저만 바꾸는 우측 곱) 하나로 충분하다. 유도 과정은 CLAUDE.md 논의
   참조. **아직 실기체 벤치 테스트로 부호를 검증하지 않았다** — `guidance/los_guidance.py`
   의 관례(`실제 부호 관례는 이 모듈 밖에서 test_rotation류로 검증 필요`)와 동일하게,
   비행 전 반드시 IMU를 손으로 돌려보며 obs의 각 성분이 예상 부호로 나오는지 확인할 것.
2. **적분 dt**: sim은 고정 `policy_dt`, 실기체는 텔레메트리 지연/지터가 있어 매 호출
   실측 경과시간을 받는다(`build()`의 `dt` 인자 — 호출부(`ros2_nodes/obs_node.py`)가
   `time.monotonic()` 차이로 재서 넘긴다).

EKF 게이팅: `RealRobotInterface.is_ekf_healthy()`가 False인 프레임은 이 클래스가
아니라 호출부가 걸러야 한다(정책에 먹이지 않고 안전정지) — 여기는 순수 조립만.
"""

from __future__ import annotations

import torch

from deploy import math_utils as mu

_T3 = torch.tensor([1.0, -1.0, -1.0])
_Q_M = torch.tensor([0.0, 1.0, 0.0, 0.0])   # NED body → Z-up body, X축 180도 회전


class ObservationBuilder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._t3 = _T3.to(device)
        self._q_m = _Q_M.to(device)
        self._z_v = torch.zeros(3, device=device)
        self._z_q = torch.zeros(3, device=device)
        self._origin_ned = None   # 미션 시작 시점 위치 — pos_env 기준점

    def reset(self, pos_ned: torch.Tensor) -> None:
        self._z_v.zero_()
        self._z_q.zero_()
        self._origin_ned = pos_ned.clone()

    def _ned_body_to_zup(self, q_ned: torch.Tensor) -> torch.Tensor:
        return mu.quat_mul(q_ned, self._q_m)

    def build(
        self,
        att_quat_ned  : torch.Tensor,   # (4,) [w,x,y,z] body→NED-world
        body_rates_ned: torch.Tensor,   # (3,) [p,q,r] NED body frame
        pos_ned       : torch.Tensor,   # (3,) NED world
        vel_ned       : torch.Tensor,   # (3,) NED world
        guidance,                        # deploy.guidance_standalone.LOSGuidance
        dt            : float,
    ) -> tuple[torch.Tensor, dict]:
        assert self._origin_ned is not None, "reset()으로 origin을 먼저 잡아야 함"

        q_b = att_quat_ned.unsqueeze(0)          # (1,4)
        pos_env = (pos_ned - self._origin_ned).unsqueeze(0)   # (1,3)

        # ── guidance는 NED 그대로 입력받아 NED-body 출력을 낸다 (guidance_standalone 참조) ──
        v_d_b_ned, q_d_ned = guidance.compute(pos_env, q_b)
        v_d_b_ned = v_d_b_ned.squeeze(0)
        q_d_ned = q_d_ned.squeeze(0)

        # ── 현재 상태: NED → Z-up body ──
        q_zup = self._ned_body_to_zup(att_quat_ned)
        omega_zup = body_rates_ned * self._t3
        v_body_ned = mu.quat_apply(mu.quat_conjugate(att_quat_ned), vel_ned)   # world→body 회전
        v_body_zup = v_body_ned * self._t3

        # ── 목표: NED → Z-up body ──
        q_d_zup = self._ned_body_to_zup(q_d_ned)
        v_d_b_zup = v_d_b_ned * self._t3

        # ── envs/vel_env.py._get_observations()와 동일 수식 ──
        q_e = mu.quat_mul(mu.quat_conjugate(q_d_zup), q_zup)
        v_e_b = v_body_zup - v_d_b_zup
        omega_b = omega_zup

        self._z_v = self._z_v + v_e_b * dt
        self._z_q = self._z_q + q_e[1:4] * dt

        obs = torch.cat([q_e, v_e_b, omega_b, self._z_v, self._z_q], dim=-1)   # (16,)

        debug = {
            "q_zup": q_zup, "q_d_zup": q_d_zup,
            "v_body_zup": v_body_zup, "v_d_b_zup": v_d_b_zup,
        }
        return obs, debug
