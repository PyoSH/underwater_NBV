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
        cruise_speed   : float = 0.5,
        reach_threshold: float = 0.5,
        heading_mode   : str = "align",
        loop           : bool = True,
        depth_hold_kp  : float = 0.8,
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
        self._speed = cruise_speed
        self._reach = reach_threshold
        self._heading_mode = heading_mode
        self._loop = loop
        self._depth_hold_kp = float(depth_hold_kp)
        self._depth_speed_limit = float(
            cruise_speed if depth_speed_limit is None else depth_speed_limit
        )
        self._terminal_hold_kp = float(terminal_hold_kp)
        self._terminal_speed_limit = float(
            cruise_speed if terminal_speed_limit is None else terminal_speed_limit
        )
        if self._depth_hold_kp <= 0.0 or self._depth_speed_limit <= 0.0:
            raise ValueError("depth hold gain/speed limit은 양수여야 함")
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
        _, next_wp = self._current_and_next(self._wp_idx)
        reached = torch.norm(next_wp - pos_env, dim=-1) < self._reach

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

        seg = next_wp - cur_wp
        seg_len = seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        seg_dir = seg / seg_len

        s = ((pos_env - cur_wp) * seg_dir).sum(-1, keepdim=True).clamp(min=0.0)
        s = torch.minimum(s, seg_len)
        look_s = torch.minimum(s + self._lookahead, seg_len)
        los_point = cur_wp + look_s * seg_dir

        to_los = los_point - pos_env
        v_d_world = self._speed * to_los / to_los.norm(dim=-1, keepdim=True).clamp_min(1e-6)

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

        # 3D LOS 정규화에서는 긴 수평 lookahead가 작은 깊이 오차를 압도한다.
        # Z(NED)는 항상 현재 세그먼트의 next waypoint 깊이를 독립 추종한다.
        depth_error = next_wp[:, 2] - pos_env[:, 2]
        v_d_world[:, 2] = torch.clamp(
            self._depth_hold_kp * depth_error,
            -self._depth_speed_limit,
            self._depth_speed_limit,
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
    pitch = torch.asin(d[:, 2].clamp(-1.0, 1.0))
    roll = torch.zeros_like(yaw)
    return mu.quat_from_euler_xyz(roll, pitch, yaw)
