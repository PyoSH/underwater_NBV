"""
BROV2 Heavy 8-thruster 모델 (BlueRobotics T200 기반) — vendored 사본
======================================================================
원본은 `robots/dynamics/brov2/thruster.py` — `deploy/`가 IsaacLab 없이 단독
복사돼도 동작하도록 `deploy/vendor/`에 내장한 사본(로직 동일). 원본이 바뀌면
`deploy/sync_vendor.sh`로 재동기화할 것 — 단일 정본은 여전히 `robots/` 쪽.
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

    # T200 파라미터 (BlueRobotics 실험값)
    _KF       = 4.4e-7   # force constant [N·s²/rad²]
    _MAX_RPM  = 3900.0
    _DEADBAND = 0.075
    _TAU      = 0.05     # 1차 지연 시정수 [s]

    def __init__(
        self,
        num_envs: int,
        dt      : float,
        device  : str,
        pos     : list | tuple | torch.Tensor | None = None,
        dir     : list | tuple | torch.Tensor | None = None,
    ):
        self.num_envs = num_envs
        self.dt       = dt
        self.device   = device

        self._pwm_state  = torch.zeros(num_envs, 8, device=device)
        self._last_thrust = torch.zeros(num_envs, 8, device=device)

        # SNAME b-frame (직접 정의, 변환 불필요). pos/dir 미전달 시 클래스 기본값(_POS/_DIR)
        # 사용 — YAML 없이 직접 생성할 때만 쓰이는 fallback (coBM/hydro_coef와 동일 패턴).
        self._t3  = _T3.to(device)
        self._pos = torch.as_tensor(pos if pos is not None else self._POS, dtype=torch.float32).to(device)
        self._dir = torch.as_tensor(dir if dir is not None else self._DIR, dtype=torch.float32).to(device)

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
        # 1차 지연 필터
        alpha = self.dt / (self._TAU + self.dt)
        self._pwm_state += alpha * (pwm_commands - self._pwm_state)
        pwm = self._pwm_state

        # PWM → RPM (T200 다항식)
        db  = self._DEADBAND
        rpm = torch.where(
            pwm >  db,  3659.9 * pwm + 345.21,
            torch.where(
                pwm < -db, 3494.4 * pwm - 433.50,
                torch.zeros_like(pwm),
            ),
        ).clamp(-self._MAX_RPM, self._MAX_RPM)

        # RPM → Thrust [N]
        k = self._KF / 4.4e-7 * 9.81
        thrust = torch.where(
            rpm > 0,
            k * ( 4.7368e-7 * rpm**2 - 1.9275e-4 * rpm + 8.4452e-2),
            torch.where(
                rpm < 0,
                k * (-3.8442e-7 * rpm**2 - 1.6186e-4 * rpm - 3.9139e-2),
                torch.zeros_like(rpm),
            ),
        )   # (N, 8)

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

    def inverse_thrust(self, force: torch.Tensor) -> torch.Tensor:
        """희망 추력(N, 부호 있음) → pwm([-1,1]) 역산.

        compute()의 순방향 다항식(PWM→RPM 선형, RPM→추력 2차)을 역순으로 푼다:
        추력 2차식을 근의공식으로 역산(양/음 분기 각각, 물리적으로 유효한 증가
        구간의 근 — 두 분기 모두 '+sqrt' 근이 맞는 근이 되도록 부호까지 확인함)
        한 뒤, RPM→PWM은 선형이라 바로 역산한다.

        Sim2Swim 스타일 6-dim wrench 액션을 B_pinv로 할당한 개별 스러스터
        희망 추력을 실제 pwm 명령으로 바꿀 때 사용 (velEnv._apply_action에서 호출).

        Parameters
        ----------
        force : (num_envs, 8) [N] 부호 있음 (+dir 방향)

        Returns
        -------
        pwm : (num_envs, 8) [-1, 1]
        """
        k = self._KF / 4.4e-7 * 9.81

        # 안전 clamp — brov2_heavy.yaml의 실측 최대추력(정/역방향 비대칭) 근방으로 제한.
        # 기본 _KF에서는 k=9.81이라 그대로 64.1/-51.5가 되고, _KF가 바뀌면 함께 스케일된다.
        force = force.clamp(-51.5 * k / 9.81, 64.1 * k / 9.81)

        # ── 추력 → RPM (2차식 역산) ──
        # rpm>0: k*(4.7368e-7·rpm² - 1.9275e-4·rpm + 8.4452e-2) = force
        a_p, b_p, c_p = 4.7368e-7 * k, -1.9275e-4 * k, 8.4452e-2 * k
        disc_p = (b_p**2 - 4 * a_p * (c_p - force)).clamp_min(0.0)
        rpm_pos = (-b_p + torch.sqrt(disc_p)) / (2 * a_p)

        # rpm<0: k*(-3.8442e-7·rpm² - 1.6186e-4·rpm - 3.9139e-2) = force
        a_n, b_n, c_n = -3.8442e-7 * k, -1.6186e-4 * k, -3.9139e-2 * k
        disc_n = (b_n**2 - 4 * a_n * (c_n - force)).clamp_min(0.0)
        rpm_neg = (-b_n + torch.sqrt(disc_n)) / (2 * a_n)

        rpm = torch.where(
            force > 0, rpm_pos,
            torch.where(force < 0, rpm_neg, torch.zeros_like(force)),
        ).clamp(-self._MAX_RPM, self._MAX_RPM)

        # ── RPM → PWM (선형 역산) ──
        pwm = torch.where(
            rpm > 0, (rpm - 345.21) / 3659.9,
            torch.where(rpm < 0, (rpm + 433.50) / 3494.4, torch.zeros_like(rpm)),
        )

        # print(f"inverse_thrust: {pwm}")
        return pwm.clamp(-1.0, 1.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._pwm_state[env_ids] = 0.0
        self._last_thrust[env_ids] = 0.0
