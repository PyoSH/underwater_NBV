"""
6-DOF 위치/자세 PID 컨트롤러 (Dynamic Positioning)
====================================================
NBV 정책이 명시한 목표점(p_target, q_target)을 "추종 후 steady-state(잔류오차 0)로
정착"시키는 저수준 실행부 — 2026-08-24 아키텍처 개정 결정
(`.claude/plans/kind-launching-kahan.md` 참조). RL이 아니라 고전 PID.

LOS 속도-cascade(guidance가 목표속도를 생성 → 별도 속도추종 루프)를 쓰지 않는
이유: 순수 P 기반 제어는 부력 트림/CoB 오프셋 같은 상수 외란 하에서
`외란/Kp`만큼의 잔류 정상상태 오차가 영구히 남는다 — steady-state 요구를
만족하려면 적분항이 필수다. 이 상수외란은 `project_step2_brov_retrain_spec`
메모리에 기록된 실기 pitch windup 사가에서 실측 확인된 실재 현상.

τ = Kp·e + Ki·∫e dt − Kd·v   (P: 추종, I: 정상상태오차 제거, D: 오버슈트 방지)

부력/중력 복원항(Fossen g(η))은 이 클래스가 아니라 물리 루프의
`robots.dynamics.fossen.Hydrodynamics.compute()`가 매 스텝 독립적으로 계산해
외력으로 직접 적용한다(step_2_BROV의 vel_env.py와 동일 구조) — 이 컨트롤러는
그 위에 얹히는 순수 오차궤환 항만 담당한다.

좌표계: 위치/속도 오차는 body frame(Z-up, IsaacLab)에서 계산 — B_pinv
할당행렬(robots.dynamics.brov2.thruster.build_allocation_matrix)이 SNAME/FRD
body frame을 기대하므로, 이 클래스가 반환하는 6-dim wrench는 step_2_BROV의
action_frame_contract 변환(FLU/Z-up 정책출력 → SNAME/FRD)과 동일한 후처리가
필요하다 — 이 클래스 자체는 축 변환 없이 Z-up body frame 그대로 반환한다
(호출부가 vel_env.py의 `_action_to_sname_multiplier`/`_sname_to_zup_sign`
패턴을 재사용해 변환할 것).

⚠ 부호 미검증: 자세오차(e_att) 부호는 Isaac Sim에서 실제 스텝응답으로
검증된 적 없음 — `guidance/los_guidance.py::_heading_from_direction`의
docstring이 같은 이유로 남긴 경고와 동일("실제 부호 관례는 이 모듈 밖에서
test_rotation류로 검증 필요"). Stage 1 스모크테스트(고정 목표점 1개로
이동+정착)에서 반드시 자세가 목표를 향해 수렴하는지, 발산하지 않는지
먼저 확인할 것 — 부호가 반대면 양의 되먹임(positive feedback)으로 즉시
발산하므로 스모크테스트에서 바로 드러난다.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def _canonicalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """q/-q(동일 회전) 중 w>=0 표현으로 통일 — 안 하면 ±180° 부근에서 오차
    벡터 부호가 불연속으로 뒤집혀 토크 명령이 채터링한다."""
    return torch.where(q[..., :1] < 0.0, -q, q)


class DPController:
    """6-DOF Dynamic Positioning PID. 목표(p_target, q_target)를 향해 추종 후
    잔류오차 0으로 정착하는 wrench(N, 6) [Fx,Fy,Fz,Tx,Ty,Tz] (Z-up body frame)를
    매 물리 스텝(dt=sim.dt, 100Hz 권장 — 정책 스텝이 아니라 물리 스텝 레이트로
    호출해야 적분/댐핑이 올바르게 동작함) 계산한다.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        device,
        kp_pos: tuple[float, float, float] = (15.0, 15.0, 15.0),
        ki_pos: tuple[float, float, float] = (1.5, 1.5, 1.5),
        kd_pos: tuple[float, float, float] = (20.0, 20.0, 20.0),
        kp_att: tuple[float, float, float] = (5.0, 5.0, 5.0),
        ki_att: tuple[float, float, float] = (0.5, 0.5, 0.5),
        kd_att: tuple[float, float, float] = (3.0, 3.0, 3.0),
        tau_max: tuple[float, float, float, float, float, float] = (
            85.0, 85.0, 120.0, 26.0, 14.0, 22.0,
        ),   # von Benzon et al. 2022 Table 4 실측 최대추력 (vel_env_cfg.py f_max와 동일 출처)
        integral_pos_limit: float = 2.0,
        integral_att_limit: float = 2.0,
    ):
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        self._kp_pos = torch.tensor(kp_pos, device=device)
        self._ki_pos = torch.tensor(ki_pos, device=device)
        self._kd_pos = torch.tensor(kd_pos, device=device)
        self._kp_att = torch.tensor(kp_att, device=device)
        self._ki_att = torch.tensor(ki_att, device=device)
        self._kd_att = torch.tensor(kd_att, device=device)
        self._tau_max = torch.tensor(tau_max, device=device)
        self._int_pos_limit = float(integral_pos_limit)
        self._int_att_limit = float(integral_att_limit)

        self._int_pos = torch.zeros(num_envs, 3, device=device)
        self._int_att = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._int_pos[env_ids] = 0.0
        self._int_att[env_ids] = 0.0

    def compute(
        self,
        pos_target_w: torch.Tensor,   # (N,3) world-frame 목표 위치
        quat_target_w: torch.Tensor,  # (N,4) world-frame 목표 자세 [w,x,y,z]
        root_pos_w: torch.Tensor,     # (N,3) 현재 위치
        root_quat_w: torch.Tensor,    # (N,4) 현재 자세
        lin_vel_b: torch.Tensor,      # (N,3) body-frame 선속도
        ang_vel_b: torch.Tensor,      # (N,3) body-frame 각속도
    ) -> torch.Tensor:
        """Returns wrench (N,6) [Fx,Fy,Fz,Tx,Ty,Tz], Z-up body frame, tau_max로 클램프됨."""
        q_conj = math_utils.quat_conjugate(root_quat_w)

        # 위치오차: world → body frame (Hydrodynamics._buoyancy, LOSGuidance.compute와
        # 동일한 world→body 회전 패턴)
        e_pos_w = pos_target_w - root_pos_w
        e_pos_b = math_utils.quat_apply(q_conj, e_pos_w)

        # 자세오차 — **부호 주의**(2026-08-25 실측으로 확인된 버그 수정):
        # `vel_env.py`가 쓰는 `q_e = q̄_d ⊗ q`는 "목표→현재" 회전이라, 이걸 그대로
        # 양의 게인으로 곱하면 현재 자세를 목표에서 *멀어지는* 방향으로 미는
        # 양의 되먹임이 된다. vel_env는 이 값을 신경망 *관측*으로만 쓰므로(정책이
        # 필요한 부호를 학습) 문제가 없지만, 손으로 짠 PID에서는 치명적이다.
        # 여기서는 반대 순서(`q⁻¹ ⊗ q_d` = "현재→목표" 회전)를 써서 벡터부가 곧
        # 교정 방향이 되게 한다 — 표준 쿼터니언 자세제어 법칙
        # `τ = −Kp·(q̄_d⊗q)_vec − Kd·ω`와 동치이며, 부호를 코드에 명시적으로
        # 드러내는 쪽이 재발 방지에 낫다고 판단.
        #
        # 실측 증상(수정 전): CoB 트림모멘트가 평형을 깨는 순간 자세오차가
        # 발산해 최대치(179°)에서 포화. CoB를 정확히 0으로 두면 오차가 0에
        # 머물렀는데, 그건 안정이 아니라 **불안정 평형점**이었다(증폭할 오차가
        # 0이라 가만히 있었을 뿐). Kd를 키우면 발산이 느려져 "개선"처럼
        # 보였던 것도 이 때문(증상 지연이지 해결이 아니었음).
        q_e = math_utils.quat_mul(math_utils.quat_conjugate(root_quat_w), quat_target_w)
        q_e = _canonicalize_quaternion(q_e)
        e_att = q_e[:, 1:4]

        tau_pos_raw = (
            self._kp_pos * e_pos_b + self._ki_pos * self._int_pos - self._kd_pos * lin_vel_b
        )
        tau_att_raw = (
            self._kp_att * e_att + self._ki_att * self._int_att - self._kd_att * ang_vel_b
        )
        tau_raw = torch.cat([tau_pos_raw, tau_att_raw], dim=-1)   # (N,6)

        tau_cmd = tau_raw.clamp(-self._tau_max, self._tau_max)

        # 축별 anti-windup(conditional integration): 그 스텝에 포화된 축만 적분 정지 —
        # deploy_v6(step_2_BROV)에서 실기로 검증된 컨셉을 RL 관측 적분이 아니라
        # classical PID 적분에 그대로 적용.
        not_saturated = tau_raw.abs() <= self._tau_max
        self._int_pos = (
            self._int_pos + e_pos_b * self.dt * not_saturated[:, 0:3]
        ).clamp(-self._int_pos_limit, self._int_pos_limit)
        self._int_att = (
            self._int_att + e_att * self.dt * not_saturated[:, 3:6]
        ).clamp(-self._int_att_limit, self._int_att_limit)

        return tau_cmd
