"""
BROV2 Heavy 8-thruster 모델 (BlueRobotics T200 기반)
======================================================
`step_2_BROV/hydrodynamics.py`에서 여기로 승격(2026-07) — 기종 무관 물리는
`robots/dynamics/fossen.py`로, BROV2 고유 액추에이터(이 파일)는 여기로 분리.
step_3(수중 인지+물리 통합)에서도 그대로 재사용.

좌표계
------
_POS, _DIR : SNAME b-frame (X=전방, Y=우현, Z=하방) — 직접 정의
compute() 출력: Z-up body frame (IsaacLab)
"""

from __future__ import annotations

import torch

from .t200_table import T200ThrustTable
from .thruster_dynamics import ThrusterDynamics

# 좌표계 변환 상수 (Z-up body ↔ NED body) — fossen.py와 동일 정의, 작은 상수라 중복 유지
# (thruster.py가 fossen.py에 의존하지 않게 하기 위한 의도적 선택).
_T3 = torch.tensor([1., -1., -1.])


def build_allocation_matrix(pos: torch.Tensor, dir_: torch.Tensor) -> torch.Tensor:
    """스러스터 위치/방향(SNAME body frame)으로부터 할당행렬 B(6,8)을 계산한다.

    B의 행 순서는 [Fx,Fy,Fz,Tx,Ty,Tz] (surge,sway,heave,roll,pitch,yaw),
    열 순서는 T1~T8 — Sim2Swim(arXiv:2512.08656) 논문의 6-dim wrench 액션을
    8-thruster로 할당할 때 B의 pseudo-inverse(B^+ = torch.linalg.pinv(B))를 쓴다.

    위치/방향이 바뀌면(YAML 갱신 등) 이 함수를 다시 호출하면 되고, 값 자체를
    하드코딩하지 않는다 — hydro_coef/coBM과 동일한 "단일 정본" 원칙.

    Parameters
    ----------
    pos  : (8,3) SNAME body frame
    dir_ : (8,3) SNAME body frame, 단위벡터

    Returns
    -------
    B : (6,8)
    """
    torque = torch.cross(pos, dir_, dim=-1)             # (8,3)
    return torch.cat([dir_.T, torque.T], dim=0)          # (6,8)


class BROV2ThrusterModel:
    """
    BROV2 Heavy 8-thruster 모델 (BlueRobotics T200 기반).

    입력  : PWM 명령 8채널 [-1, 1]  (Z-up body frame 기준 정의)
    출력  : Z-up body frame 합력 (N), 합토크 (N·m)
    내부  : NED body frame에서 합산

    추진기 배치 (Z-up body frame)
    ─────────────────────────────
    T1(ccw) 전우  T2(ccw) 전좌  ← 수평 4기: surge/sway/yaw
    T3(cw)  후우  T4(cw)  후좌
    T5(ccw) 전우  T6(cw)  전좌  ← 수직 4기: heave/roll/pitch
    T7(cw)  후우  T8(ccw) 후좌
    """

    # 추진기 위치 (SNAME b-frame: X=전방, Y=우현, Z=하방) 기본값 — brov2_custom_physics.usda 기반.
    # 실제 값은 robots/data/BROV2/brov2_heavy.yaml의 'thrusters.list'에서 params.py가 읽어
    # 생성자에 전달하므로, 이 값은 YAML 없이 클래스를 직접 생성할 때만 쓰이는 fallback이다
    # (hydro_coef와 동일 패턴). 2026-07-15: 좌우 쌍 Y좌표가 전부 ±3.5mm로 어긋나 있던 것을
    # 발견(body 원점 좌우대칭 오프셋으로 추정)하여 쌍별 평균으로 대칭화함.
    _POS = torch.tensor([
        [ 0.138,  0.1011, -0.002],   # T1: 전우 수평
        [ 0.138, -0.1011, -0.002],   # T2: 전좌 수평
        [-0.138,  0.1003, -0.002],   # T3: 후우 수평
        [-0.138, -0.1003, -0.002],   # T4: 후좌 수평
        [ 0.129,  0.2177, -0.061],   # T5: 전우 수직
        [ 0.129, -0.2177, -0.061],   # T6: 전좌 수직
        [-0.111,  0.2177, -0.061],   # T7: 후우 수직
        [-0.111, -0.2177, -0.061],   # T8: 후좌 수직
    ], dtype=torch.float32)

    # 추진기 추력 방향 (SNAME b-frame, 단위벡터, 양(+) PWM 시) 기본값 — _POS와 동일하게
    # brov2_heavy.yaml의 'thrusters.list[].axis'가 정본, 이건 fallback.
    # 규칙: 각 추진기 힘 방향 ⊥ 위치벡터 → yaw 토크 최대화
    _DIR = torch.tensor([
        [-0.7071,  0.7071,  0.0],   # T1: 전우 → 후우현 방향
        [-0.7071, -0.7071,  0.0],   # T2: 전좌 → 후좌현 방향
        [ 0.7071,  0.7071,  0.0],   # T3: 후우 → 전우현 방향
        [ 0.7071, -0.7071,  0.0],   # T4: 후좌 → 전좌현 방향
        [ 0.0,     0.0,     1.0],   # T5~T8: 하방 (음수 명령 → 상승)
        [ 0.0,     0.0,     1.0],
        [ 0.0,     0.0,     1.0],
        [ 0.0,     0.0,     1.0],
    ], dtype=torch.float32)

    # T200 파라미터. 추력 곡선은 제조사 공개 실측 테이블(t200_table.npz)이
    # 정본이다 — PWM→RPM→추력 다항식은 20V 곡선만 맞았고(RMSE 0.55N), 4S팩이
    # 부하에서 실제로 내는 14V에서는 추력을 44% 과대평가했다. 또 affine RPM
    # 절편이 deadband 직후를 620RPM으로 점프시켜 최소 유효추력을 실측 0.44N
    # 대신 1.44N으로 3배 부풀렸다. deadband/정역비대칭/전압의존성은 이제 전부
    # 테이블 데이터 안에 있고, 액추에이터 동특성은 thruster_dynamics.py가 갖는다.
    # 4S Li-ion: 방전 종지 ~12.6V, 만충 ~16.8V, 공칭 14.8V.
    NOMINAL_VOLTAGE = 14.8
    # 참고용 공칭 deadband(정규화 PWM). 추력 계산에는 더 이상 쓰이지 않는다 —
    # 실제 dead zone은 전압 의존이고(실측 ±26us@20V ~ ±40us@10V) 테이블 안에
    # 있으므로 dead_zone()을 쓸 것. 이 상수는 validator/physics_tests가 참조하는
    # 16V 근방 공칭값으로만 남긴다.
    _DEADBAND = 0.075

    def __init__(
        self,
        num_envs: int,
        dt      : float,
        device  : str,
        pos     : list | tuple | torch.Tensor | None = None,
        dir     : list | tuple | torch.Tensor | None = None,
        voltage : float = NOMINAL_VOLTAGE,
        table   : "T200ThrustTable | None" = None,
        dynamics_model: str = "third_order",
        dynamics_bandwidth_scales: tuple = (1.0,),
    ):
        self.num_envs = num_envs
        self.dt       = dt
        self.device   = device

        # 액추에이터 동특성. 기본값이 von Benzon Eq.(19) 3차인 이유는
        # thruster_dynamics.py 참조 — 기존 1차(tau=0.05)는 대역폭이 3.2Hz뿐이라
        # 25Hz 제어에서 나오는 12.5Hz chatter를 25%로 깎아 sim에서 안 보이게
        # 만든다. 실제 스러스터는 그 대역을 98% 통과시킨다.
        # dynamics_model="first_order"로 예전 거동을 재현할 수 있다.
        self._dynamics = ThrusterDynamics(
            num_envs=num_envs, num_thrusters=8, dt=dt, device=device,
            model=dynamics_model, bandwidth_scales=tuple(dynamics_bandwidth_scales),
        )
        self._last_thrust = torch.zeros(num_envs, 8, device=device)

        # 제조사 실측 추력 테이블. 전압은 env별 상태라 clamp_thrust/inverse_thrust의
        # 시그니처가 바뀌지 않는다 — brov_ros2의 호출부가 그대로 동작해야 한다.
        self._table = table if table is not None else T200ThrustTable(device=device)
        self._voltage = torch.full((num_envs,), float(voltage), device=device)

        # 도메인 랜덤화 대상 — env×thruster별 승수, compute()의 최종 추력에
        # 곱해진다(테이블은 안 건드림). 1.0 = 실측 곡선 그대로. 전압 sag는 이제
        # 이 승수가 아니라 _voltage로 표현하므로, 여기 남는 것은 개체 편차뿐이다.
        # randomize() 참조 — Hydrodynamics.randomize()와 동일 패턴.
        self._thrust_scale = torch.ones(num_envs, 8, device=device)

        # SNAME b-frame (직접 정의, 변환 불필요). pos/dir 미전달 시 클래스 기본값(_POS/_DIR)
        # 사용 — YAML 없이 직접 생성할 때만 쓰이는 fallback (coBM/hydro_coef와 동일 패턴).
        self._t3  = _T3.to(device)
        self._pos = torch.as_tensor(pos if pos is not None else self._POS, dtype=torch.float32).to(device)
        self._dir = torch.as_tensor(dir if dir is not None else self._DIR, dtype=torch.float32).to(device)

    def randomize(
        self,
        env_ids: torch.Tensor,
        thrust_scale: torch.Tensor | None = None,
        voltage: torch.Tensor | None = None,
    ) -> None:
        """도메인 랜덤화 — env._reset_idx()에서 env_ids만 호출.

        Parameters
        ----------
        thrust_scale : (M, 8) 절대 승수. **개체 편차 전용**이다. 전압 sag는
            이제 ``voltage``로 표현하므로 여기 곱하지 않는다 — 예전처럼 두
            성분을 곱해 넘기면 전압 효과가 두 번 들어간다.
        voltage : (M,) 공급 전압 [V]. 4S 팩의 방전~만충 범위(12.6~16.8V)에서
            샘플링하면 추력 한계·deadband 폭·정역 비대칭이 실측대로 함께
            움직인다. 추상 승수와 달리 정방향/역방향 계산이 서로 일관된다.
        """

        if thrust_scale is not None:
            self._thrust_scale[env_ids] = thrust_scale
        if voltage is not None:
            self._voltage[env_ids] = voltage.to(self._voltage.dtype).reshape(-1)
        # 동특성 대역폭도 DR 대상. Eq.(19)의 식별 데이터는 T200이 아니므로
        # 극점 위치 자체가 불확실하다 — 범위를 열어 정책이 견디게 한다.
        self._dynamics.randomize(env_ids)

    @property
    def _pwm_state(self) -> torch.Tensor:
        """지연 필터를 통과한 실효 PWM.

        예전에는 1차 필터의 상태 그 자체였다. 3차로 바뀌면서 내부 상태는
        (N, 8, 3)이 됐으므로, 외부에서 참조하던 "실효 PWM" 의미만 유지한다
        (physics_tests/bottom_up.py).
        """
        state = self._dynamics.state
        gains = self._dynamics._c[self._dynamics._index]
        return (state * gains.unsqueeze(1)).sum(-1)

    @property
    def voltage(self) -> torch.Tensor:
        return self._voltage

    def set_voltage(self, voltage: torch.Tensor | float) -> None:
        """실기 배터리 텔레메트리를 물릴 때 쓰는 진입점 (deploy 측)."""
        if isinstance(voltage, torch.Tensor):
            self._voltage.copy_(voltage.to(self._voltage.dtype).reshape(-1))
        else:
            self._voltage.fill_(float(voltage))

    def dead_zone(self) -> tuple[torch.Tensor, torch.Tensor]:
        """현재 전압에서 낼 수 있는 최소 (역, 정) 추력 [N], 각 (num_envs, 1)."""
        return self._table.dead_zone(self._voltage)

    def compute(self, pwm_commands: torch.Tensor) -> tuple:
        """
        PWM → NED body frame 계산 → Z-up body frame 반환.

        Parameters
        ----------
        pwm_commands : (num_envs, 8), [-1, 1]

        Returns
        -------
        forces_zup  : (num_envs, 3) [N]
        torques_zup : (num_envs, 3) [N·m]
        """
        # 액추에이터 동특성 (로터 관성 + 모터 전기 + 끌려오는 물의 부가질량)
        pwm = self._dynamics.step(pwm_commands)

        # PWM → Thrust [N] (제조사 실측 테이블, env별 공급 전압으로 보간)
        thrust = self._table.force(pwm, self._voltage)   # (N, 8)
        thrust = thrust * self._thrust_scale   # 개체 편차 DR (randomize() 참조)

        # SNAME b-frame에서 합력/합토크
        f_each = thrust.unsqueeze(-1) * self._dir.unsqueeze(0)          # (N, 8, 3)
        pos_ex = self._pos.unsqueeze(0).expand(self.num_envs, -1, -1)   # (N, 8, 3)
        t_each = torch.cross(pos_ex, f_each, dim=-1)                    # (N, 8, 3)

        forces_ned  = f_each.sum(dim=1)   # (N, 3)
        torques_ned = t_each.sum(dim=1)   # (N, 3)

        # 추력 시각화(디버그 화살표)용 — 개별 스러스터 크기 캐시
        self._last_thrust = thrust   # (N, 8) [N], 부호 있음(+PWM=+_DIR 방향)

        # SNAME → Z-up 변환 후 반환
        return forces_ned * self._t3, torques_ned * self._t3

    @property
    def force_limits_n(self) -> tuple[float, float]:
        """Widest per-thruster force limits over the whole table, in newtons.

        Deliberately voltage-independent: the only consumer is the reward's
        ``clamp_residual`` normalizer (vel_env.py), which must not change when
        domain randomization samples a different battery voltage per env.  For
        the actual limit at an env's voltage use :meth:`force_limits`.
        """

        return (
            float(self._table._force.min()),
            float(self._table._force.max()),
        )

    def force_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env ``(reverse, forward)`` limits at the current voltage, ``(N, 1)``."""

        return self._table.force_limits(self._voltage)

    def clamp_thrust(self, force: torch.Tensor) -> torch.Tensor:
        """Apply the same physical force limit used by :meth:`inverse_thrust`."""

        return self._table.clamp_thrust(force, self._voltage)

    def inverse_thrust(self, force: torch.Tensor) -> torch.Tensor:
        """희망 추력(N, 부호 있음) → pwm([-1,1]) 역산.

        compute()의 순방향 테이블을 그대로 이진탐색으로 되짚는다. 따라서
        ``compute(inverse_thrust(f)) == f``가 부동소수 정밀도로 성립한다 —
        예전 근의공식 역산은 그러지 못했다. dead zone 안의 힘을 요청하면
        판별식이 음수가 되어 ``clamp_min(0)``이 근을 포물선 꼭짓점에 고정했고,
        그 결과 **요청과 반대 부호의 pwm**(작은 역추력 요청 → +0.064)을
        돌려줬다. 테이블 역산은 낼 수 없는 힘을 0으로 되돌린다.

        Sim2Swim 스타일 6-dim wrench 액션을 B_pinv로 할당한 개별 스러스터
        희망 추력을 실제 pwm 명령으로 바꿀 때 사용 (velEnv._apply_action에서 호출).

        Parameters
        ----------
        force : (num_envs, 8) [N] 부호 있음 (+dir 방향)

        Returns
        -------
        pwm : (num_envs, 8) [-1, 1]
        """
        return self._table.pwm(force, self._voltage).clamp(-1.0, 1.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._dynamics.reset(env_ids)
        self._last_thrust[env_ids] = 0.0
