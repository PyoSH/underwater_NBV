"""
NBV 목표점 상태 홀더
======================
2026-08-24 아키텍처 개정: 최초 설계는 `guidance/los_guidance.py`처럼 위치오차를
목표속도(v_d_b)로 변환하는 LOS 유도 레이어를 상정했으나, NBV는 "이산 목표점에
도달 후 정착"이 목적이라 LOS의 lookahead/velocity-생성 로직(연속 경로추종용)이
불필요하다는 결론에 도달했다(`.claude/plans/kind-launching-kahan.md` 참조).

이 클래스는 그래서 유도 계산을 하지 않는다 — NBV 정책이 매 정책스텝
`set_target()`으로 갱신하는 목표(p_target, q_target)를 다음 갱신까지 그대로
들고 있다가 `control/dp_controller.py::DPController.compute()`에 넘기는
얇은 상태 홀더일 뿐이다.
"""

from __future__ import annotations

import torch


class NBVGuidance:
    """NBV 정책이 명령한 목표 위치/자세를 유지하는 상태 홀더."""

    def __init__(self, num_envs: int, device):
        self.num_envs = num_envs
        self.device = device
        self._p_target = torch.zeros(num_envs, 3, device=device)
        self._q_target = torch.zeros(num_envs, 4, device=device)
        self._q_target[:, 0] = 1.0   # identity quaternion [w,x,y,z]

    def reset(self, env_ids: torch.Tensor) -> None:
        self._p_target[env_ids] = 0.0
        self._q_target[env_ids] = 0.0
        self._q_target[env_ids, 0] = 1.0

    def set_target(
        self,
        env_ids: torch.Tensor,
        p_target_w: torch.Tensor,
        q_target_w: torch.Tensor,
    ) -> None:
        """NBV 정책의 연속 액션이 매 정책스텝 호출 — 목표 위치/자세 갱신."""
        self._p_target[env_ids] = p_target_w
        self._q_target[env_ids] = q_target_w

    @property
    def p_target(self) -> torch.Tensor:
        return self._p_target

    @property
    def q_target(self) -> torch.Tensor:
        return self._q_target
