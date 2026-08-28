"""
LOSGuidance — 배포용(IsaacLab 비의존) 포팅
==============================================
`guidance/los_guidance.py`와 로직은 동일하지만 `isaaclab.utils.math` 대신
`deploy/math_utils.py`(순수 torch)를 쓴다. topside PC에 IsaacLab이 없어서
원본 파일을 그대로 import할 수 없기 때문에 부득이하게 복제함 — 로직을 고치게
되면 `guidance/los_guidance.py`도 같이 고쳐야 한다(단일 정본 아님, 알려진 위험).

이 파일의 world frame은 호출자가 선택한다. 실기체에서는 NED 또는 제어 시작 yaw를
제거한 start-heading frame을 `obs_builder.py`가 일관되게 전달한다. body frame Z-up
변환 역시 `obs_builder.py`가 이 함수의 출력에 대해 별도로 수행한다.
"""

from __future__ import annotations

import torch

from deploy import math_utils as mu


class LOSGuidance:
    """`guidance/los_guidance.py`의 LOSGuidance와 동일 인터페이스/로직."""

    def __init__(
        self,
        waypoints      : torch.Tensor,
        device,
        lookahead_dist : float = 1.0,
        lookahead_vert : float | None = None,
        cruise_speed   : float = 0.5,
        reach_threshold: float = 0.5,
        heading_mode   : str = "align",
        loop           : bool = True,
        depth_hold_kp  : float | None = None,
        depth_speed_limit: float | None = None,
        terminal_hold_kp: float = 0.5,
        terminal_speed_limit: float | None = None,
    ):
        valid_heading_modes = {"align", "upright", "straight", "random_at_waypoint"}
        if heading_mode not in valid_heading_modes:
            raise ValueError(
                f"heading_mode={heading_mode!r} invalid; "
                f"expected one of {sorted(valid_heading_modes)}"
            )
        self._wp = waypoints
        self.device = device
        self._lookahead = lookahead_dist
        self._lookahead_v = lookahead_dist if lookahead_vert is None else lookahead_vert
        self._speed = cruise_speed
        self._reach = reach_threshold
        self._heading_mode = heading_mode
        self._loop = loop
        # depth_hold_kp / depth_speed_limit: **deprecated, no-op**.
        # 이전 구현("lookahead 지점을 향하는 3D 벡터 정규화")은 수평 보정과 수직
        # 보정이 고정 크기 U_d를 두고 경쟁했기 때문에 수평 오차가 커지면 깊이
        # 보정이 잠식됐고(실측 32%), 그 우회로 Z축에 별도 P 제어기를 덧댔었다.
        # 그 덧댐은 v_d_world[2]를 정규화 이후에 덮어써서 ||v_d||를 0.5에서
        # 이탈시켰다 — 정책이 학습한 명령 크기는 정확히 0.5 m/s 하나뿐이다.
        # Breivik-Fossen LOS는 수직축(υ_d)이 수평축(χ_d)과 독립이라 잠식이 없고
        # ||v_d|| = U_d를 정확히 보존하므로 두 우회가 모두 불필요해졌다.
        # 기존 config/launch가 계속 이 키를 넘겨도 깨지지 않도록 인자는 남긴다.
        self._depth_hold_kp = depth_hold_kp
        self._depth_speed_limit = depth_speed_limit
        self._terminal_hold_kp = float(terminal_hold_kp)
        self._terminal_speed_limit = float(
            cruise_speed if terminal_speed_limit is None else terminal_speed_limit
        )
        if self._terminal_hold_kp <= 0.0 or self._terminal_speed_limit <= 0.0:
            raise ValueError("terminal hold gain/speed limit은 양수여야 함")

        self.num_envs, self.num_wp, _ = waypoints.shape
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        # loop=False일 때만 의미 있음 — 마지막 웨이포인트 도달 후 True로 고정,
        # 이후 final waypoint position hold로 전환한다. sim 쪽
        # guidance/los_guidance.py에는 없는 개념(sim은 학습/평가 스크립트가 항상
        # 반복 경로를 원해서 loop 개념 자체가 없었음) — 실배포 미션은 보통 한 번
        # 돌고 종료 위치를 유지해야 해서 여기만 추가.
        self.mission_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        self._random_q_d = mu.identity_quat(self.num_envs, device)
        self._straight_q_d = mu.identity_quat(self.num_envs, device)
        if heading_mode == "random_at_waypoint":
            self._random_q_d = self._sample_random_attitude(self.num_envs)

    def _sample_random_attitude(self, n: int) -> torch.Tensor:
        roll  = mu.sample_uniform(-torch.pi / 2, torch.pi / 2, (n,), self.device)
        pitch = mu.sample_uniform(-torch.pi / 2, torch.pi / 2, (n,), self.device)
        yaw   = mu.sample_uniform(-torch.pi,     torch.pi,     (n,), self.device)
        return mu.quat_from_euler_xyz(roll, pitch, yaw)

    def reset(self, env_ids: torch.Tensor, initial_quat: torch.Tensor | None = None) -> None:
        self._wp_idx[env_ids] = 0
        self.mission_complete[env_ids] = False
        if self._heading_mode == "straight":
            if initial_quat is None:
                raise ValueError("straight heading reset에는 initial_quat가 필요함")
            yaw = mu.yaw_from_quat(initial_quat)
            zero = torch.zeros_like(yaw)
            # 시작 roll/pitch는 목표에 포함하지 않고, 시작 yaw만 고정한다.
            self._straight_q_d[env_ids] = mu.quat_from_euler_xyz(zero, zero, yaw)
        if self._heading_mode == "random_at_waypoint":
            self._random_q_d[env_ids] = self._sample_random_attitude(len(env_ids))

    def _current_and_next(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_i = torch.arange(self.num_envs, device=self.device)
        return self._wp[env_i, idx], self._wp[env_i, (idx + 1) % self.num_wp]

    def compute(
        self,
        pos_env    : torch.Tensor,
        root_quat_w: torch.Tensor,
        advance_waypoint: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """pos_env/root_quat_w는 world(NED) 기준. 반환 v_d_b/q_d도 NED-body 규약
        (obs_builder가 Z-up으로 변환).

        ``advance_waypoint=False``는 제어 시작 전 shadow observation을 만들 때 사용한다.
        이때 LOS 목표는 계산하되 waypoint index와 mission_complete는 변경하지 않는다.
        """
        cur_wp, next_wp = self._current_and_next(self._wp_idx)
        # 전환 조건 둘. **근접만으로는 부족하다.**
        #   ① 근접   ‖next_wp - pos‖ < reach
        #   ② 통과   along-track 진행률 s 가 세그먼트 길이를 넘음
        # BF LOS는 **무한 직선**을 따라 조향하므로, 끝점을 지나도 같은 방향을
        # 계속 가리킨다. ①만 보면 한 번 지나친 뒤 영원히 나아간다 -- 실제
        # SITL에서 0.20 m 하강 구간을 지나쳐 8.5 m까지 가라앉았다. 구
        # lookahead-point 법칙은 look_s를 세그먼트 끝에 clamp해서 끝점을 지나면
        # to_los가 뒤를 가리켜 스스로 되돌아왔는데, 그 성질이 BF에는 없다.
        _seg = next_wp - cur_wp
        _seg_len = _seg.norm(dim=-1)
        _seg_dir = _seg / _seg_len.clamp_min(1e-6).unsqueeze(-1)
        _s_along = ((pos_env - cur_wp) * _seg_dir).sum(-1)
        reached = (torch.norm(next_wp - pos_env, dim=-1) < self._reach) | (_s_along >= _seg_len)

        if advance_waypoint:
            if self._loop:
                self._wp_idx = torch.where(reached, (self._wp_idx + 1) % self.num_wp, self._wp_idx)
            else:
                # 마지막 세그먼트(idx == num_wp-2)에서 도달하면 그 이상 전진하지 않고
                # mission_complete만 세운다 — idx가 num_wp-1까지 가면 (idx+1)%num_wp가
                # 0으로 wrap해서 처음으로 되돌아가버리므로(요청한 "반복" 버그) 아예 막음.
                at_last_segment = self._wp_idx == (self.num_wp - 2)
                self.mission_complete = self.mission_complete | (reached & at_last_segment)
                advance = reached & ~at_last_segment
                self._wp_idx = torch.where(advance, self._wp_idx + 1, self._wp_idx)

        cur_wp, next_wp = self._current_and_next(self._wp_idx)

        if advance_waypoint and self._heading_mode == "random_at_waypoint" and reached.any():
            idx = reached.nonzero(as_tuple=True)[0]
            self._random_q_d[idx] = self._sample_random_attitude(len(idx))

        # ── Breivik & Fossen (2005) Sec. IV, 3D LOS ──
        # 경로 고정 방위각 χ_p / 앙각 υ_p를 기준으로 두고, path-parallel frame에서
        # 분해한 cross-track 오차 e(수평)와 vertical-track 오차 h에 각각 독립적인
        # lookahead 보정을 더한다. 두 축이 독립이라 서로의 보정을 잠식하지 않고,
        # 각각 ±90°로 자연 포화하며, ||v_d_world|| = cruise_speed가 정확히 보존된다.
        # 조향각이 (e, h)만의 함수이고 진행률 s와 무관하다 — 이전 구현은 lookahead
        # 지점을 세그먼트 끝에 clamp해서, 끝에 가까울수록 명령이 경로 방향에서
        # waypoint 방향으로 끌려갔다(경로 위 끝점에서는 |to_los|=0으로 퇴화).
        # 회귀 테스트: deploy/test_guidance_los_bf.py
        #
        # 이 블록은 frame-agnostic이다 — 호출자가 NED를 주든 Z-up을 주든 χ_p/υ_p와
        # e/h가 같은 축 규약으로 계산되므로 보정 부호가 자기일관적이다. "위/아래"의
        # 해석만 뒤집힌다. q_d도 _heading_from_direction(v_d_world)로 만들어
        # frame 가정을 두지 않는다(순수 대수 조건 quat_apply(q,x̂)==v̂).
        seg = next_wp - cur_wp
        seg_dir = seg / seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        chi_p = torch.atan2(seg_dir[:, 1], seg_dir[:, 0])
        ups_p = torch.atan2(seg_dir[:, 2], seg_dir[:, :2].norm(dim=-1).clamp_min(1e-9))

        cs, sn = torch.cos(chi_p), torch.sin(chi_p)
        cu, su = torch.cos(ups_p), torch.sin(ups_p)
        d = pos_env - cur_wp
        e = -sn * d[:, 0] + cs * d[:, 1]
        h = -su * (cs * d[:, 0] + sn * d[:, 1]) + cu * d[:, 2]

        chi_d = chi_p + torch.atan(-e / self._lookahead)
        ups_d = ups_p + torch.atan(-h / self._lookahead_v)

        current_speed = self._speed
        cd, sd = torch.cos(chi_d), torch.sin(chi_d)
        cv, sv = torch.cos(ups_d), torch.sin(ups_d)
        v_d_world = current_speed * torch.stack([cd * cv, sd * cv, sv], dim=-1)

        if not self._loop:
            # 마지막 waypoint에 한 번 도달했더라도 속도 목표를 0으로 고정하면
            # 음성부력/테더 외력으로 이탈한 뒤 복귀할 수 없다. 완주 상태에서는
            # 최종 waypoint에 대한 position outer-loop를 계속 유지한다.
            terminal_error = next_wp - pos_env
            terminal_velocity = self._terminal_hold_kp * terminal_error
            terminal_norm = terminal_velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            terminal_scale = torch.clamp(
                self._terminal_speed_limit / terminal_norm, max=1.0
            )
            terminal_velocity = terminal_velocity * terminal_scale
            v_d_world = torch.where(
                self.mission_complete.unsqueeze(-1), terminal_velocity, v_d_world
            )

        v_d_b = mu.quat_apply(mu.quat_conjugate(root_quat_w), v_d_world)

        if self._heading_mode == "align":
            q_d = _heading_from_direction(v_d_world, self.device)
        elif self._heading_mode == "straight":
            q_d = self._straight_q_d
        elif self._heading_mode == "random_at_waypoint":
            q_d = self._random_q_d
        else:   # "upright": NED/mission frame yaw=0
            q_d = mu.identity_quat(self.num_envs, self.device)

        return v_d_b, q_d


def _heading_from_direction(direction_w: torch.Tensor, device) -> torch.Tensor:
    d = direction_w / direction_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    yaw = torch.atan2(d[:, 1], d[:, 0])
    pitch = -torch.asin(d[:, 2].clamp(-1.0, 1.0))
    roll = torch.zeros_like(yaw)
    return mu.quat_from_euler_xyz(roll, pitch, yaw)
