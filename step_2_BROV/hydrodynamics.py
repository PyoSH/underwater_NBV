"""
BROV2 수중 동역학 계산 모듈
=====================================
Fossen (2011) 6-DOF 수중 동역학 구현.

좌표계
------
- _POS, _DIR  : SNAME b-frame  (X=전방, Y=우현, Z=하방) — 직접 정의
- BROV2Hydrodynamics 입출력 : Z-up body frame (IsaacLab)
- 내부 계산   : SNAME/NED body frame
- 출력 변환   : T₃=diag(1,-1,-1)  (SNAME b → Z-up b)

IsaacLab(PhysX) 담당: M_RB·ν̇, C_RB·ν, F_gravity
이 모듈 담당       : M_A·ν̇, C_A·ν, D·ν, 부력+복원모멘트, 추진력

주의: brov_rigid.py 에서 linear_damping=angular_damping=0 필수.
"""

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate

# 좌표계 변환 상수 (Z-up body ↔ NED body)
_T3 = torch.tensor([1., -1., -1.])
_T6 = torch.tensor([1., -1., -1., 1., -1., -1.])


# ==============================================================================
# BROV2ThrusterModel
# ==============================================================================

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
    # 실제 값은 ../robots/data/BROV2/brov2_heavy.yaml의 'thrusters.list'에서 env.py가 읽어
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

    def reset(self, env_ids: torch.Tensor) -> None:
        self._pwm_state[env_ids] = 0.0
        self._last_thrust[env_ids] = 0.0


# ==============================================================================
# BROV2Hydrodynamics
# ==============================================================================

class BROV2Hydrodynamics:
    """
    Fossen (2011) 기반 6-DOF 수중 유체역학.

    내부 계산: NED body frame (수식 직접 적용)
    입력/출력: Z-up body frame (IsaacLab 인터페이스)

    적용 항목
    ----------
    τ_ext = τ_thrust + g(η) - M_A·ν̇ - C_A(ν)·ν - D(ν)·ν

    g(η) = [F_buoy_body; r_COB × F_buoy_body]   (부력 + 복원모멘트)
    IsaacLab이 중력(F_gravity)과 강체 항(M_RB, C_RB)을 처리하므로 제외.
    """

    _WATER_DENSITY : float = 997.0
    _GRAVITY       : float = 9.81
    _VOLUME        : float = 0.0134      # [m³] 기본값 (런타임에 cfg.volume 으로 오버라이드)
    # COM → COB 벡터 (SNAME body frame: X=전방,Y=우현,Z=하방) 기본값.
    # 실제 값은 USD asset(base_link의 'buoyancy:comToCob' custom attribute)에서
    # env.py가 읽어 생성자에 명시적으로 전달하므로, 이 값은 CAD 데이터 없이 클래스를
    # 직접 생성할 때만 쓰이는 대략적인 fallback이다 (+Z로 1cm, 순수 스칼라 가정).
    _COB_VECTOR    : list  = [0.0, 0.0, -0.01]  # [m] SNAME frame, fallback only
    _ACC_ALPHA     : float = 0.3         # 가속도 저역필터 계수

    # Fossen 6-DOF 유체역학 계수 (NED body frame 기준, 모두 양수 크기값) 기본값.
    # [surge, sway, heave, roll, pitch, yaw]
    # 실제 값은 ../robots/data/BROV2/brov2_heavy.yaml의 'hydro_coef'에서 env.py가 읽어
    # 생성자에 명시적으로 전달하므로, 이 값은 YAML 없이 클래스를 직접 생성할 때만 쓰이는
    # fallback이다 (von Benzon et al. 2022 Table A1과 동일).
    _ADDED_MASS        = [6.36,  7.12,  18.68, 0.189, 0.135, 0.222]
    _LINEAR_DAMPING    = [13.70, 0.00,  33.00, 0.00,  0.80,  0.00 ]
    _QUADRATIC_DAMPING = [141.0, 217.0, 190.0, 1.19,  0.47,  1.50 ]

    def __init__(
        self,
        num_envs         : int,
        dt                : float,
        device            : str,
        volume            : float | None = None,
        cob_vector        : list | tuple | None = None,
        water_density     : float | None = None,
        added_mass        : list | tuple | None = None,
        linear_damping    : list | tuple | None = None,
        quadratic_damping : list | tuple | None = None,
    ):
        self.num_envs = num_envs
        self.dt       = dt
        self.device   = device

        rho = water_density or self._WATER_DENSITY
        V   = volume        or self._VOLUME
        cob = cob_vector    or self._COB_VECTOR

        self._buoy_mag = rho * self._GRAVITY * V   # [N] 실제 부력 크기

        # 변환 벡터
        self._t3 = _T3.to(device)
        self._t6 = _T6.to(device)

        # 부력 기준벡터 (Z-up world)
        self._world_up = torch.tensor([0., 0., 1.], device=device)

        # COB 위치벡터 — NED(SNAME) body frame, COM 기준
        self._r_cob_ned = torch.tensor(cob, dtype=torch.float32, device=device)

        # 6-DOF 대각 행렬 (num_envs, 6, 6)
        def _diag(coeffs):
            return (
                torch.diag(torch.tensor(coeffs, dtype=torch.float32))
                .unsqueeze(0).repeat(num_envs, 1, 1).to(device)
            )
        self._Ma = _diag(added_mass        or self._ADDED_MASS)
        self._Dl = _diag(linear_damping    or self._LINEAR_DAMPING)
        self._Dq = _diag(quadratic_damping or self._QUADRATIC_DAMPING)

        # 가속도 추정 버퍼 (NED body frame)
        self._prev_vel_ned = torch.zeros(num_envs, 6, device=device)
        self._prev_acc_ned = torch.zeros(num_envs, 6, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        root_quat_w : torch.Tensor,   # (N, 4) [w,x,y,z]
        lin_vel_b   : torch.Tensor,   # (N, 3) Z-up body
        ang_vel_b   : torch.Tensor,   # (N, 3) Z-up body
    ) -> tuple:
        """
        유체역학 합력/합토크 계산.

        Returns
        -------
        forces_zup  : (N, 3) [N]    Z-up body frame
        torques_zup : (N, 3) [N·m]  Z-up body frame
        """
        # ── 입력 변환: Z-up body → NED body ──────────────────────────
        vel_zup = torch.cat([lin_vel_b, ang_vel_b], dim=-1)   # (N, 6)
        vel_ned = vel_zup * self._t6                           # (N, 6)

        # 가속도 추정 (NED body frame)
        acc_ned = self._update_acc(vel_ned)

        # ── Fossen 계산 (전부 NED body frame) ─────────────────────────
        g_ned    = self._buoyancy(root_quat_w)   # (N, 6)
        fd_ned   = self._damping(vel_ned)         # (N, 6)
        fam_ned  = self._added_mass(acc_ned)      # (N, 6)
        fcor_ned = self._coriolis(vel_ned)        # (N, 6)

        # ── NED body frame 합산 ────────────────────────────────────────
        total_ned = g_ned - (fd_ned + fam_ned + fcor_ned)   # (N, 6)

        # ── 출력 변환: NED body → Z-up body ──────────────────────────
        total_zup = total_ned * self._t6

        return total_zup[:, :3], total_zup[:, 3:]

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev_vel_ned[env_ids] = 0.0
        self._prev_acc_ned[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Private: Fossen 각 항 계산 (모두 NED body frame, 6-벡터 반환)
    # ------------------------------------------------------------------

    def _buoyancy(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """
        g(η) = [F_buoy_ned; r_COB_ned × F_buoy_ned]

        F_buoy_W(Z-up) → Z-up body(쿼터니언 회전) → NED body(T₃)
        τ_restore = r_COB_NED × F_buoy_NED
        """
        # 1. 부력: Z-up world → Z-up body
        f_world = self._world_up * self._buoy_mag
        f_zup   = quat_apply(
            quat_conjugate(root_quat_w),
            f_world.unsqueeze(0).expand(self.num_envs, -1),
        )   # (N, 3)

        # 2. Z-up body → NED body
        f_ned = f_zup * self._t3   # (N, 3)

        # 3. 복원 모멘트: r_COB_NED × F_buoy_NED
        r_cob = self._r_cob_ned.unsqueeze(0).expand(self.num_envs, -1)
        t_ned = torch.cross(r_cob, f_ned, dim=-1)   # (N, 3)

        return torch.cat([f_ned, t_ned], dim=-1)   # (N, 6)

    def _damping(self, vel_ned: torch.Tensor) -> torch.Tensor:
        """
        D(ν)·ν = (D_l + D_q·diag(|ν|))·ν   (NED body frame, 6-벡터)
        """
        vel_diag = torch.diag_embed(vel_ned)                        # (N, 6, 6)
        D = self._Dl + self._Dq * torch.abs(vel_diag)
        return (D @ vel_ned.unsqueeze(-1)).squeeze(-1)               # (N, 6)

    def _added_mass(self, acc_ned: torch.Tensor) -> torch.Tensor:
        """
        M_A·ν̇   (NED body frame, 6-벡터)
        """
        return (self._Ma @ acc_ned.unsqueeze(-1)).squeeze(-1)        # (N, 6)

    def _coriolis(self, vel_ned: torch.Tensor) -> torch.Tensor:
        """
        C_A(ν)·ν   (NED body frame, 6-벡터)

        f_CA = -ω × (A₁₁·v_lin)
        τ_CA = -(A₁₁·v_lin × v_lin + A₂₂·v_ang × v_ang)
        """
        v_lin = vel_ned[:, :3]
        v_ang = vel_ned[:, 3:]

        Mav   = (self._Ma @ vel_ned.unsqueeze(-1)).squeeze(-1)
        Mav_l = Mav[:, :3]   # A₁₁·v_lin
        Mav_a = Mav[:, 3:]   # A₂₂·v_ang

        f_cor = -torch.cross(v_ang, Mav_l, dim=-1)
        t_cor = -(torch.cross(Mav_l, v_lin, dim=-1) + torch.cross(Mav_a, v_ang, dim=-1))

        return torch.cat([f_cor, t_cor], dim=-1)   # (N, 6)

    def _update_acc(self, vel_ned: torch.Tensor) -> torch.Tensor:
        """저역 필터 기반 NED body frame 가속도 추정."""
        raw = (vel_ned - self._prev_vel_ned) / self.dt
        acc = (1. - self._ACC_ALPHA) * self._prev_acc_ned + self._ACC_ALPHA * raw
        self._prev_vel_ned = vel_ned.detach().clone()
        self._prev_acc_ned = acc.detach().clone()
        return acc
