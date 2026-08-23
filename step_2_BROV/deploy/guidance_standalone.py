"""
LOSGuidance — 배포용(IsaacLab 비의존) 포팅
==============================================
`guidance/los_guidance.py`와 로직은 동일하지만 `isaaclab.utils.math` 대신
`deploy/math_utils.py`(순수 torch)를 쓴다. topside PC에 IsaacLab이 없어서
원본 파일을 그대로 import할 수 없기 때문에 부득이하게 복제함 — 로직을 고치게
되면 `guidance/los_guidance.py`도 같이 고쳐야 한다(단일 정본 아님, 알려진 위험).

이 파일이 다루는 "world frame"은 실기체 배포에서는 MAVLink NED(X=North,Y=East,
Z=Down)를 그대로 쓴다 — sim의 Isaac world와 달리 실세계엔 진짜 나침반 기준이
있지만, 정책은 q_e(상대오차)/v_e_b(body frame)만 관측하므로 world 기준의 절대
방향은 학습된 정책 동작에 영향이 없다. 즉 world를 NED로 두든 다른 기준으로
두든 상관없고, 여기서는 그냥 MAVLink가 이미 주는 NED를 그대로 쓴다(추가 변환
불필요) — body frame Z-up 변환은 `deploy/obs_builder.py`가 이 함수의 출력에
대해 별도로 수행한다.
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
    ):
        self._wp = waypoints
        self.device = device
        self._lookahead = lookahead_dist
        self._speed = cruise_speed
        self._reach = reach_threshold
        self._heading_mode = heading_mode
        self._loop = loop

        self.num_envs, self.num_wp, _ = waypoints.shape
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        # loop=False일 때만 의미 있음 — 마지막 웨이포인트 도달 후 True로 고정,
        # 이후 v_d_b=0(제자리 정지)으로 덮어씀. sim 쪽 guidance/los_guidance.py에는
        # 없는 개념(sim은 학습/평가 스크립트가 항상 반복 경로를 원해서 loop 개념
        # 자체가 없었음) — 실배포 미션은 보통 한 번 돌고 끝나야 해서 여기만 추가.
        self.mission_complete = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        self._random_q_d = mu.identity_quat(self.num_envs, device)
        if heading_mode == "random_at_waypoint":
            self._random_q_d = self._sample_random_attitude(self.num_envs)

    def _sample_random_attitude(self, n: int) -> torch.Tensor:
        roll  = mu.sample_uniform(-torch.pi / 2, torch.pi / 2, (n,), self.device)
        pitch = mu.sample_uniform(-torch.pi / 2, torch.pi / 2, (n,), self.device)
        yaw   = mu.sample_uniform(-torch.pi,     torch.pi,     (n,), self.device)
        return mu.quat_from_euler_xyz(roll, pitch, yaw)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._wp_idx[env_ids] = 0
        self.mission_complete[env_ids] = False
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
        """pos_env/root_quat_w는 world(NED) 기준. 반환 v_d_b/q_d도 NED-body 규약
        (obs_builder가 Z-up으로 변환)."""
        _, next_wp = self._current_and_next(self._wp_idx)
        reached = torch.norm(next_wp - pos_env, dim=-1) < self._reach

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

        if self._heading_mode == "random_at_waypoint" and reached.any():
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

        v_d_b = mu.quat_apply(mu.quat_conjugate(root_quat_w), v_d_world)
        if not self._loop:
            # 완주한 env는 제자리 정지(v_d=0) — "정지"는 액추에이션을 끊는 게
            # 아니라(그건 /brov/estop 몫) 목표 속도를 0으로 줘서 정책이 스스로
            # 제자리를 유지하도록 하는 것. q_d는 아래에서 계산된 값을 그대로 씀
            # (heading_mode="align"이면 종점 근방 방향벡터가 흔들려 q_d도 약간
            # 떨릴 수 있음 — "upright"/"random_at_waypoint"는 영향 없음).
            v_d_b = torch.where(self.mission_complete.unsqueeze(-1), torch.zeros_like(v_d_b), v_d_b)

        if self._heading_mode == "align":
            q_d = _heading_from_direction(v_d_world, self.device)
        elif self._heading_mode == "random_at_waypoint":
            q_d = self._random_q_d
        else:   # "upright"
            q_d = mu.identity_quat(self.num_envs, self.device)

        return v_d_b, q_d


def _heading_from_direction(direction_w: torch.Tensor, device) -> torch.Tensor:
    d = direction_w / direction_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    yaw = torch.atan2(d[:, 1], d[:, 0])
    pitch = torch.asin(d[:, 2].clamp(-1.0, 1.0))
    roll = torch.zeros_like(yaw)
    return mu.quat_from_euler_xyz(roll, pitch, yaw)
