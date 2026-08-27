"""
3D LOS(Line-of-Sight) 경로 유도
================================
고전 제어 — RL 아님. 웨이포인트 경로를 따라가도록 v_d^b, q_d를 매 스텝 계산한다.
`BROVVelEnv`(RL 저수준 속도/자세 컨트롤러)의 명령 생성기로 쓰인다 — 정책이 고정된
뒤 평가/배포 단계에서만 사용하고, 학습 루프에는 관여하지 않는다.

유도 법칙은 Sim2Swim이 인용한 Breivik & Fossen (2005), "Principles of Guidance-Based
Path Following in 2D and 3D", CDC 2005, Sec. IV의 3D LOS다. 경로에 고정된 방위각 χ_p와
앙각 υ_p를 기준으로 두고, path-parallel frame에서 분해한 cross-track 오차 e(수평)와
vertical-track 오차 h(수직)에 각각 독립적인 lookahead 보정을 더한다:

    χ_d = χ_p + arctan(-e / Δ_h)          (방위각)
    υ_d = υ_p + arctan(-h / Δ_v)          (앙각)
    v_d^w = U_d · [cos χ_d cos υ_d,  sin χ_d cos υ_d,  sin υ_d]

BROV2는 완전구동(6DOF thruster)이라 이 v_d^w를 선수각 조향 명령으로 쓰지 않고 body
frame으로 그대로 변환해 속도 명령으로 넣는다 — 자세와 무관하게 어느 방향으로든 추력을
낼 수 있으므로(Sim2Swim이 "attitude-independent"라 부르는 것과 같은 아이디어).
q_d는 논문 서술("the desired heading and pitch equal to the desired course and
elevation angles calculated by the LOS guidance law") 그대로 (χ_d, υ_d)에서 직접
만든다(heading_mode="align"). "upright"/"random_at_waypoint"로 자세를 진행방향과
분리할 수도 있다(Sim2Swim Fig.4 (b)/(c) 재현용).

이전 구현은 "lookahead 지점을 향하는 3D 벡터를 정규화"하는 자체 설계였다. 직선 경로
초반에는 BF와 수식이 일치하지만(cross-track 0.24m에서 각도차 0.24°), 수평/수직 보정이
고정된 크기 U_d를 두고 경쟁하기 때문에 수평 오차가 커지면 수직 보정이 잠식된다 —
실측으로 cross-track 1.9m / vertical 1.24m 상태에서 수직 성분이 32% 작았다. 또한
lookahead 지점을 세그먼트 끝에 clamp하므로, 세그먼트 끝에 가까워질수록 명령 방향이
경로 방향에서 waypoint 방향으로 끌려갔다 — 같은 cross-track 오차라도 진행률 s에 따라
조향이 달라졌고, 경로 위 정확히 끝점에서는 |to_los|=0으로 퇴화했다(보통 reach 판정이
먼저 나서 가려짐). BF에는 lookahead "지점" 자체가 없어 조향각이 (e, h)만의 함수다.

좌표계 주의: 원논문은 NED(Z-down)라 v_z = -U_d sin υ_d 이지만, 여기(IsaacLab)는
Z-up이므로 +υ = 상승이고 v_z = +U_d sin υ_d 이다.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


class LOSGuidance:
    """웨이포인트 경로 기반 3D LOS 유도.

    Parameters
    ----------
    waypoints       : (num_envs, num_wp, 3) env-local 상대좌표 — envs/traj_env.py의
                      `_generate_trajectories`/`_waypoints`와 동일한 규약.
    device          : torch device
    lookahead_dist  : float, [m] — 수평 lookahead 거리 Δ_h. 경로 수렴의 시상수를
                      결정하는 주 튜닝 파라미터로, waypoint 간격과는 무관하다.
                      작을수록 공격적(cross-track e에 대한 조향 이득이 ~1/Δ rad/m).
                      Breivik & Fossen은 기체 길이의 수 배를 권한다 — BROV2 0.46m
                      기준 1~2.3m.
    lookahead_vert  : float | None, [m] — 수직 lookahead 거리 Δ_v. None이면
                      lookahead_dist와 동일. 깊이 응답을 수평과 따로 조율할 때 사용.
    cruise_speed    : float, [m/s] — v_d_b 크기. Sim2Swim 학습 속도(Vd=0.5)와
                      맞춰서 정책의 학습 분포를 벗어나지 않게 한다.
    reach_threshold : float, [m] — 다음 waypoint로 전환하는 거리.
    heading_mode    : "align" | "upright" | "random_at_waypoint" — q_d를 이동방향에
                      맞출지, 항상 수평 유지할지, waypoint 도달마다 새로 랜덤 샘플할지
                      (Sim2Swim 논문 Trial(c): roll,pitch~U(-π/2,π/2), yaw~U(-π,π)).
    """

    def __init__(
        self,
        waypoints      : torch.Tensor,
        device,
        lookahead_dist : float = 1.0,
        lookahead_vert : float | None = None,
        cruise_speed   : float = 0.5,
        reach_threshold: float = 0.5,
        heading_mode   : str = "align",
    ):
        self._wp = waypoints
        self.device = device
        self._lookahead = lookahead_dist
        self._lookahead_v = lookahead_dist if lookahead_vert is None else lookahead_vert
        self._speed = cruise_speed
        self._reach = reach_threshold
        self._heading_mode = heading_mode

        self.num_envs, self.num_wp, _ = waypoints.shape
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)

        # heading_mode="random_at_waypoint" 전용 상태 — waypoint 도달 전까지 값 유지.
        self._random_q_d = _identity_quat(self.num_envs, device)
        if heading_mode == "random_at_waypoint":
            self._random_q_d = self._sample_random_attitude(self.num_envs)

    def _sample_random_attitude(self, n: int) -> torch.Tensor:
        """Sim2Swim 논문 Trial(c): roll,pitch ~ U(-π/2,π/2), yaw ~ U(-π,π)."""
        roll  = math_utils.sample_uniform(-torch.pi / 2, torch.pi / 2, (n,), self.device)
        pitch = math_utils.sample_uniform(-torch.pi / 2, torch.pi / 2, (n,), self.device)
        yaw   = math_utils.sample_uniform(-torch.pi,     torch.pi,     (n,), self.device)
        return math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._wp_idx[env_ids] = 0
        if self._heading_mode == "random_at_waypoint":
            self._random_q_d[env_ids] = self._sample_random_attitude(len(env_ids))

    def _current_and_next(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_i = torch.arange(self.num_envs, device=self.device)
        return self._wp[env_i, idx], self._wp[env_i, (idx + 1) % self.num_wp]

    def compute(
        self,
        pos_env    : torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pos_env     : (N,3) env origin 기준 현재 위치
        root_quat_w : (N,4) 현재 자세 [w,x,y,z]

        Returns
        -------
        v_d_b : (N,3) body-frame 희망 속도 [m/s] — BROVVelEnv 관측에 그대로 주입
        q_d   : (N,4) 희망 자세 [w,x,y,z]
        """
        # waypoint 도달 판정 → 인덱스 전진. next_wp(지금 향하고 있는 목표) 기준으로
        # 봐야 한다 — cur_wp(이미 지나온 구간 출발점) 기준으로 체크하면 로봇이
        # 앞으로 나아갈수록 오히려 멀어지기만 해서 영원히 도달 판정이 안 남
        # (실제로 이 버그로 wp_idx가 안 넘어가서 lookahead 지점에 고착 → 목표속도
        # 발산까지 이어진 것을 로그로 확인함).
        _, next_wp = self._current_and_next(self._wp_idx)
        reached = torch.norm(next_wp - pos_env, dim=-1) < self._reach
        self._wp_idx = torch.where(reached, (self._wp_idx + 1) % self.num_wp, self._wp_idx)
        cur_wp, next_wp = self._current_and_next(self._wp_idx)   # 갱신된 인덱스로 재조회

        # "random_at_waypoint": 도달한 env만 새 자세 목표를 뽑고, 나머지는 이전 값 유지
        if self._heading_mode == "random_at_waypoint" and reached.any():
            idx = reached.nonzero(as_tuple=True)[0]
            self._random_q_d[idx] = self._sample_random_attitude(len(idx))

        # ── 경로 고정 방위각 χ_p / 앙각 υ_p (Breivik & Fossen 2005, Sec. IV) ──
        seg = next_wp - cur_wp                                          # (N,3)
        seg_dir = seg / seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        chi_p = torch.atan2(seg_dir[:, 1], seg_dir[:, 0])
        ups_p = torch.atan2(seg_dir[:, 2], seg_dir[:, :2].norm(dim=-1).clamp_min(1e-9))

        # ── path-parallel frame으로 오차 분해 ──
        # 기저: x_p = seg_dir, y_p = [-sin χ_p, cos χ_p, 0] (수평 좌측),
        #       z_p = x_p × y_p = [-sin υ_p cos χ_p, -sin υ_p sin χ_p, cos υ_p] (상방).
        # e>0 = 경로 좌측, h>0 = 경로 위쪽.
        cs, sn = torch.cos(chi_p), torch.sin(chi_p)
        cu, su = torch.cos(ups_p), torch.sin(ups_p)
        d = pos_env - cur_wp
        e = -sn * d[:, 0] + cs * d[:, 1]
        h = -su * (cs * d[:, 0] + sn * d[:, 1]) + cu * d[:, 2]

        # ── LOS 조향각 ──
        # 두 축이 독립이라 서로의 보정 크기를 잠식하지 않고, 각각 ±90°로 자연 포화한다.
        # 이전의 "lookahead 지점 정규화" 방식과 달리 lookahead '지점'이 없으므로
        # waypoint 근방에서 방향이 정의되지 않는 특이점도 없다.
        chi_d = chi_p + torch.atan(-e / self._lookahead)
        ups_d = ups_p + torch.atan(-h / self._lookahead_v)

        # ── 월드 프레임 희망 속도벡터 (Z-up: +υ = 상승) ──
        cd, sd = torch.cos(chi_d), torch.sin(chi_d)
        cv, sv = torch.cos(ups_d), torch.sin(ups_d)
        v_d_world = self._speed * torch.stack([cd * cv, sd * cv, sv], dim=-1)

        # ── body frame으로 변환 (자세와 무관 — attitude-independent LOS) ──
        v_d_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat_w), v_d_world)

        # ── 희망 자세 ──
        if self._heading_mode == "align":
            # 논문 서술 그대로 "desired heading and pitch equal to the desired course
            # and elevation angles". 방향벡터에서 역산(asin)하지 않고 χ_d, υ_d를 직접 쓴다.
            #
            # pitch = **-υ_d**. R = Rz(yaw)·Ry(pitch)이면 기수는
            # [cosθcosψ, cosθsinψ, -sinθ]이므로 기수의 world Z 성분은 -sin(pitch)인데,
            # Z-up에서 v_d^w의 Z 성분은 +sin(υ_d)다 → pitch = -υ_d.
            # 원논문(NED)은 v_z = -U sin υ_d 라서 pitch = +υ_d 로 그대로 쓴다.
            # 이 부호가 _heading_from_direction()의 pitch = -asin(dz)와 정확히 일치한다.
            zero = torch.zeros_like(chi_d)
            q_d = math_utils.quat_from_euler_xyz(zero, -ups_d, chi_d)
        elif self._heading_mode == "random_at_waypoint":
            q_d = self._random_q_d
        else:   # "upright"
            q_d = _identity_quat(self.num_envs, self.device)

        return v_d_b, q_d


def _heading_from_direction(direction_w: torch.Tensor, device) -> torch.Tensor:
    """월드 프레임 방향벡터를 바라보는 자세(roll=0)를 계산.

    정의는 하나다: ``quat_apply(q, [1,0,0]) == direction_w``. 이건 frame
    convention과 무관한 순수 대수 조건이라 Z-up이든 NED든 같은 식이 나온다.

    yaw = atan2(dy, dx),  pitch = **-asin(dz)**.

    부호에 주의. roll=0이면 회전은 ``R = Rz(yaw)·Ry(pitch)``이고
    ``R·x̂ = [cosθcosψ, cosθsinψ, -sinθ]``이므로 기수의 world Z 성분은
    ``-sin(pitch)``다. 이것이 ``dz``와 같아야 하므로 ``pitch = -asin(dz)``다.
    구면좌표의 elevation(``asin(z/r)``)과 부호가 반대인 지점이며,
    2026-08-26까지 ``+asin(dz)``로 잘못 구현되어 있었다(기수 Z가 뒤집힘).
    검증은 ``tests/test_desired_states.py``의 물리 기반 테스트가 한다 —
    수식을 자기 자신과 비교하면 이 오류를 영원히 못 잡는다.
    """
    d = direction_w / direction_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    yaw = torch.atan2(d[:, 1], d[:, 0])
    pitch = -torch.asin(d[:, 2].clamp(-1.0, 1.0))
    roll = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def _identity_quat(n: int, device) -> torch.Tensor:
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0
    return q
