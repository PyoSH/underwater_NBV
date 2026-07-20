"""
3D LOS(Line-of-Sight) 경로 유도
================================
고전 제어 — RL 아님. 웨이포인트 경로를 따라가도록 v_d^b, q_d를 매 스텝 계산한다.
`BROVVelEnv`(RL 저수준 속도/자세 컨트롤러)의 명령 생성기로 쓰인다 — 정책이 고정된
뒤 평가/배포 단계에서만 사용하고, 학습 루프에는 관여하지 않는다.

BROV2는 완전구동(6DOF thruster)이라 어뢰형 AUV의 표준 Fossen LOS(선수각 조향으로
경로에 수렴)를 그대로 쓰지 않는다. 대신 "월드 프레임에서 lookahead 지점을 향하는
속도벡터"를 직접 만들어 body frame으로 변환하는 방식을 쓴다 — 자세와 무관하게 어느
방향으로든 추력을 낼 수 있으므로(Sim2Swim 논문이 "attitude-independent" LOS라 부르는
것과 같은 아이디어). q_d(희망 자세)는 별도로, 진행방향에 맞춰 정렬한다(heading_mode
="align" 기본값 — roll=0, yaw/pitch를 이동방향에서 계산. 자세를 진행방향과 분리하고
싶으면 "upright"로 전환 가능, Sim2Swim 논문의 square-path 테스트처럼 임의 자세 명령을
따로 주입하고 싶을 때도 q_d를 이 클래스 밖에서 덮어쓰면 됨).

논문(arXiv:2512.08656)이 "3D LOS guidance"라고만 언급하고 구체 수식은 안 줘서,
정확한 lookahead/cross-track 처리 방식은 이 프로젝트에서 직접 설계한 것 — 표준
Fossen 2D/3D LOS의 lookahead 개념만 차용했다.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


class LOSGuidance:
    """웨이포인트 경로 기반 3D LOS 유도.

    Parameters
    ----------
    waypoints       : (num_envs, num_wp, 3) env-local 상대좌표 — env.py의
                      `_generate_trajectories`/`_waypoints`와 동일한 규약.
    device          : torch device
    lookahead_dist  : float, [m] — lookahead 거리 Δ. 현재 세그먼트 길이보다 작게
                      잡아야 한다 (기본 궤적 세그먼트 길이 ~1.57m 기준 1.0m).
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
        cruise_speed   : float = 0.5,
        reach_threshold: float = 0.5,
        heading_mode   : str = "align",
    ):
        self._wp = waypoints
        self.device = device
        self._lookahead = lookahead_dist
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

        # ── lookahead 지점 계산 (현재 세그먼트 cur→next 위) ──
        seg = next_wp - cur_wp                                          # (N,3)
        seg_len = seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        seg_dir = seg / seg_len

        # 현재 위치를 세그먼트에 투영한 진행률 s, 거기서 lookahead만큼 전진
        # (세그먼트를 넘어가면 다음 세그먼트를 고려하는 게 정석이지만, waypoint
        # 간격이 lookahead보다 충분히 크면 세그먼트 끝에서 clamp하는 것으로 충분)
        s = ((pos_env - cur_wp) * seg_dir).sum(-1, keepdim=True).clamp(min=0.0)
        s = torch.minimum(s, seg_len)
        look_s = torch.minimum(s + self._lookahead, seg_len)
        los_point = cur_wp + look_s * seg_dir

        # ── 월드 프레임 희망 속도벡터 ──
        to_los = los_point - pos_env
        v_d_world = self._speed * to_los / to_los.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # ── body frame으로 변환 (자세와 무관 — attitude-independent LOS) ──
        v_d_b = math_utils.quat_apply(math_utils.quat_conjugate(root_quat_w), v_d_world)

        # ── 희망 자세 ──
        if self._heading_mode == "align":
            q_d = _heading_from_direction(v_d_world, self.device)
        elif self._heading_mode == "random_at_waypoint":
            q_d = self._random_q_d
        else:   # "upright"
            q_d = _identity_quat(self.num_envs, self.device)

        return v_d_b, q_d


def _heading_from_direction(direction_w: torch.Tensor, device) -> torch.Tensor:
    """월드 프레임 방향벡터를 바라보는 자세(roll=0)를 계산.

    yaw = atan2(dy,dx), pitch = asin(dz) (Z-up world 기준 — 위로 향하면 +pitch).
    실제 부호 관례는 이 모듈 밖에서 test_rotation류로 검증 필요.
    """
    d = direction_w / direction_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    yaw = torch.atan2(d[:, 1], d[:, 0])
    pitch = torch.asin(d[:, 2].clamp(-1.0, 1.0))
    roll = torch.zeros_like(yaw)
    return math_utils.quat_from_euler_xyz(roll, pitch, yaw)


def _identity_quat(n: int, device) -> torch.Tensor:
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1.0
    return q
