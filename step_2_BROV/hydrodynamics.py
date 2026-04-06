"""
BROV2 수중 동역학 계산 모듈
=====================================
MarineGym (marinegym/robots/drone/underwaterVehicle.py) 방식을
IsaacLab DirectRLEnv 에 맞게 독립적으로 재구현한 모듈.

좌표계 규약
-----------
- IsaacLab 기본: Z-up, 우수 좌표계 (ENU-like)
- Body frame : X = 전방, Y = 좌방, Z = 상방
- 힘/토크     : body frame 기준 계산 → set_external_force_and_torque 로 적용

포함 클래스
-----------
BROV2ThrusterModel  : 8-thruster PWM → body-frame 합력/합토크
BROV2Hydrodynamics  : 부력 · 항력 · 추가질량 · Coriolis 계산 (MarineGym 동등)
"""

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate


# ==============================================================================
# BROV2ThrusterModel
# ==============================================================================

class BROV2ThrusterModel:
    """
    BROV2 Heavy 8-thruster 모델.

    입력  : 8채널 PWM 명령 [-1, 1]
    출력  : body frame 합력 (N), 합토크 (N·m)

    추진기 배치 (body frame, X=전방 Y=좌방 Z=상방) [근사값 — USD 확인 후 수정]
    ┌──────────────────────────────────────────────────────────────┐
    │  수평 4기 (T1~T4) : surge / sway / yaw 제어                 │
    │    T1(ccw) 전우  T2(ccw) 전좌   (45° 배치)                  │
    │    T3(cw)  후우  T4(cw)  후좌                                │
    │  수직 4기 (T5~T8) : heave / roll / pitch 제어               │
    │    T5(ccw) 전우  T6(cw)  전좌  T7(cw)  후우  T8(ccw) 후좌  │
    └──────────────────────────────────────────────────────────────┘

    Note
    ----
    * 실제 추진기 위치·방향은 BlueROV2_buoyancy.usd 를 확인하여 수정할 것.
    * BlueRobotics T200 thruster 의 RPM-Thrust 다항식 사용 (MarineGym t200.py 기반).
    """

    # ------------------------------------------------------------------
    # 추진기 위치 (body frame, m) — BlueROV2 Heavy 근사
    # ------------------------------------------------------------------
    _POS = torch.tensor([
        [ 0.14, -0.10,  0.00],   # T1: 전우 수평
        [ 0.14,  0.10,  0.00],   # T2: 전좌 수평
        [-0.14, -0.10,  0.00],   # T3: 후우 수평
        [-0.14,  0.10,  0.00],   # T4: 후좌 수평
        [ 0.11, -0.11,  0.00],   # T5: 전우 수직
        [ 0.11,  0.11,  0.00],   # T6: 전좌 수직
        [-0.11, -0.11,  0.00],   # T7: 후우 수직
        [-0.11,  0.11,  0.00],   # T8: 후좌 수직
    ], dtype=torch.float32)  # (8, 3)

    # ------------------------------------------------------------------
    # 추진기 추력 방향 단위벡터 (body frame) — 양(+) PWM 시 추력 방향
    # ------------------------------------------------------------------
    _DIR = torch.tensor([
        [ 0.7071, -0.7071,  0.0],   # T1: +45°
        [ 0.7071,  0.7071,  0.0],   # T2: -45°
        [-0.7071, -0.7071,  0.0],   # T3: 225°
        [-0.7071,  0.7071,  0.0],   # T4: 135°
        [ 0.0,     0.0,    -1.0],   # T5: 하방 (-Z)
        [ 0.0,     0.0,    -1.0],   # T6: 하방
        [ 0.0,     0.0,    -1.0],   # T7: 하방
        [ 0.0,     0.0,    -1.0],   # T8: 하방
    ], dtype=torch.float32)  # (8, 3)

    # T200 파라미터 (BlueRobotics 실험값)
    _KF      = 4.4e-7   # force constant [N·s²/rad²]
    _MAX_RPM = 3900.0   # 최대 RPM
    _DEADBAND = 0.075   # PWM 데드밴드
    _TAU      = 0.05    # 1차 지연 시정수 [s]

    def __init__(self, num_envs: int, dt: float, device: str):
        self.num_envs = num_envs
        self.dt       = dt
        self.device   = device

        # 1차 지연 필터 상태
        self._pwm_state = torch.zeros(num_envs, 8, device=device)

        # 상수 텐서를 device 로 이동
        self._pos = self._POS.to(device)   # (8, 3)
        self._dir = self._DIR.to(device)   # (8, 3)

    # ------------------------------------------------------------------

    def compute(
        self,
        pwm_commands: torch.Tensor,   # (num_envs, 8), [-1, 1]
    ) -> tuple:
        """
        PWM 명령 → body-frame 합력/합토크.

        Returns
        -------
        forces_b  : (num_envs, 3)  [N]
        torques_b : (num_envs, 3)  [N·m]
        """
        # --- 1차 지연 필터 (τ = 0.05 s) ---
        alpha = self.dt / (self._TAU + self.dt)
        self._pwm_state = self._pwm_state + alpha * (pwm_commands - self._pwm_state)
        pwm = self._pwm_state  # (num_envs, 8)

        # --- PWM → RPM (T200 다항식, 데드밴드 포함) ---
        db = self._DEADBAND
        rpm = torch.where(
            pwm >  db,  3659.9 * pwm + 345.21,
            torch.where(
                pwm < -db, 3494.4 * pwm - 433.50,
                torch.zeros_like(pwm)
            )
        )
        rpm = torch.clamp(rpm, -self._MAX_RPM, self._MAX_RPM)   # (num_envs, 8)

        # --- RPM → Thrust [N] (T200 실험 기반 다항식) ---
        # (KF / 4.4e-7) * 9.81 = 9.81 (KF 가 4.4e-7 인 경우)
        k = self._KF / 4.4e-7 * 9.81
        thrust = torch.where(
            rpm > 0,
            k * ( 4.7368e-7 * rpm**2 - 1.9275e-4 * rpm + 8.4452e-2),
            k * (-3.8442e-7 * rpm**2 - 1.6186e-4 * rpm - 3.9139e-2),
        )  # (num_envs, 8)

        # --- 각 추진기의 3D 힘 벡터 (body frame) ---
        # thrust: (num_envs, 8), _dir: (8, 3)
        f_each = thrust.unsqueeze(-1) * self._dir.unsqueeze(0)   # (num_envs, 8, 3)

        # --- 토크: r × F ---
        pos_expanded = self._pos.unsqueeze(0).expand(self.num_envs, -1, -1)  # (num_envs, 8, 3)
        t_each = torch.cross(pos_expanded, f_each, dim=-1)                   # (num_envs, 8, 3)

        forces_b  = f_each.sum(dim=1)   # (num_envs, 3)
        torques_b = t_each.sum(dim=1)   # (num_envs, 3)

        return forces_b, torques_b

    def reset(self, env_ids: torch.Tensor) -> None:
        """지정 환경의 추진기 PWM 상태 초기화."""
        self._pwm_state[env_ids] = 0.0


# ==============================================================================
# BROV2Hydrodynamics
# ==============================================================================

class BROV2Hydrodynamics:
    """
    MarineGym 방식 수중 유체역학 계산기.

    계산 요소 (Fossen 2011 기반)
    --------------------------------
    1. 부력 + 복원 토크   (Buoyancy restoring forces)
    2. 선형 + 이차 항력   (Linear & Quadratic Damping)
    3. 추가 질량력        (Added Mass)
    4. Coriolis / 원심력  (Coriolis/Centripetal)

    모든 힘/토크는 **body frame** 기준으로 반환.
    IsaacLab 의 set_external_force_and_torque 는 local-frame 을 받으므로
    별도 변환 없이 바로 사용 가능.

    참조
    ----
    MarineGym/marinegym/robots/drone/underwaterVehicle.py
    MarineGym/marinegym/robots/assets/usd/BlueROV/BlueROV.yaml

    파라미터 (BROV2 Heavy 기준 — BlueROV.yaml 값 사용, 필요시 수정)
    ------------------------------------------------------------------
    volume            = 0.022747843 m³  (중성 부력 기준)
    cob_offset        = 0.01 m          (COB 가 COM 보다 위에 있는 거리)
    added_mass        = [5.5, 12.7, 14.57, 0.12, 0.12, 0.12]  6-DOF
    linear_damping    = [4.03, 6.22, 5.18, 0.07, 0.07, 0.07]
    quadratic_damping = [18.18, 21.66, 36.99, 1.55, 1.55, 1.55]
    """

    # --- 기본 물리 상수 ---
    _WATER_DENSITY : float = 997.0    # [kg/m³]
    _GRAVITY       : float = 9.81     # [m/s²]

    # --- BROV2 Heavy 유체역학 계수 ---
    _VOLUME : float = 0.022747843     # [m³]
    _COB_OFFSET : float = 0.01        # [m]  COM → COB (+Z body)

    _ADDED_MASS        = [5.5,   12.7,  14.57, 0.12, 0.12, 0.12]
    _LINEAR_DAMPING    = [4.03,  6.22,  5.18,  0.07, 0.07, 0.07]
    _QUADRATIC_DAMPING = [18.18, 21.66, 36.99, 1.55, 1.55, 1.55]

    # 가속도 저역 필터 alpha (MarineGym 기본값)
    _ACC_ALPHA : float = 0.3

    def __init__(
        self,
        num_envs    : int,
        dt          : float,
        device      : str,
        volume      : float | None = None,
        cob_offset  : float | None = None,
        water_density: float | None = None,
    ):
        self.num_envs = num_envs
        self.dt       = dt
        self.device   = device

        rho   = water_density or self._WATER_DENSITY
        V     = volume        or self._VOLUME
        cob   = cob_offset    or self._COB_OFFSET

        self._buoy_mag = rho * self._GRAVITY * V   # [N] 부력 크기
        self._cob_offset = cob

        # --- 6-DOF 행렬 (num_envs, 6, 6) ---
        def _diag_mat(coeffs):
            return (
                torch.diag(torch.tensor(coeffs, dtype=torch.float32))
                .unsqueeze(0)
                .repeat(num_envs, 1, 1)
                .to(device)
            )

        self._Ma  = _diag_mat(self._ADDED_MASS)         # Added mass
        self._Dl  = _diag_mat(self._LINEAR_DAMPING)     # Linear damping
        self._Dq  = _diag_mat(self._QUADRATIC_DAMPING)  # Quadratic damping

        # --- 가속도 추정 버퍼 ---
        self._prev_vel_b = torch.zeros(num_envs, 6, device=device)
        self._prev_acc_b = torch.zeros(num_envs, 6, device=device)

        # --- 상수 텐서 ---
        # 월드 상방 벡터 (부력 계산용)
        self._world_up = torch.tensor([0.0, 0.0, 1.0], device=device)
        # COB 위치벡터 (body frame)
        self._r_cob = torch.tensor([0.0, 0.0, cob], device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        root_quat_w : torch.Tensor,   # (num_envs, 4) [w, x, y, z]
        lin_vel_b   : torch.Tensor,   # (num_envs, 3) body frame
        ang_vel_b   : torch.Tensor,   # (num_envs, 3) body frame
    ) -> tuple:
        """
        수중 유체역학 합력/합토크 계산 (body frame).

        Returns
        -------
        forces_b  : (num_envs, 3)  [N]
        torques_b : (num_envs, 3)  [N·m]
        """
        vel_b = torch.cat([lin_vel_b, ang_vel_b], dim=-1)   # (num_envs, 6)

        # 가속도 추정 (저역 필터)
        acc_b = self._update_acc(vel_b)   # (num_envs, 6)

        # 각 요소별 계산
        f_b, t_b     = self._buoyancy(root_quat_w)
        f_d, t_d     = self._damping(vel_b)
        f_am, t_am   = self._added_mass(acc_b)
        f_cor, t_cor = self._coriolis(vel_b)

        # 합산
        # 부력: 복원력(+), 항력·추가질량·Coriolis: 운동 저항(-) 으로 적용
        forces_b  = f_b  - (f_d  + f_am  + f_cor)
        torques_b = t_b  - (t_d  + t_am  + t_cor)

        return forces_b, torques_b

    def reset(self, env_ids: torch.Tensor) -> None:
        """지정 환경의 동역학 필터 상태 초기화."""
        self._prev_vel_b[env_ids] = 0.0
        self._prev_acc_b[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Private: 각 요소 계산
    # ------------------------------------------------------------------

    def _buoyancy(self, root_quat_w: torch.Tensor) -> tuple:
        """
        부력 + 복원 토크 (body frame).

        부력 벡터 = ρgV · ẑ_world → body frame 으로 회전
        복원 토크 = r_COB × f_buoy  (body frame)
        """
        # 부력 벡터: world frame → body frame
        # quat_conjugate(q) 로 역회전 적용
        f_buoy_world = self._world_up * self._buoy_mag         # (3,)
        f_buoy_b = quat_apply(
            quat_conjugate(root_quat_w),
            f_buoy_world.unsqueeze(0).expand(self.num_envs, -1),
        )  # (num_envs, 3)

        # 복원 토크: r_cob × f_buoy
        r_cob = self._r_cob.unsqueeze(0).expand(self.num_envs, -1)  # (num_envs, 3)
        t_restore = torch.cross(r_cob, f_buoy_b, dim=-1)             # (num_envs, 3)

        return f_buoy_b, t_restore

    def _damping(self, vel_b: torch.Tensor) -> tuple:
        """
        선형 + 이차 항력 (body frame).

        D(v) = D_lin + D_quad * |v|   (대각 행렬)
        F_damp = D(v) · v
        """
        # |v| 대각 행렬 (num_envs, 6, 6)
        vel_diag = torch.diag_embed(vel_b)

        damp_mat = self._Dl + self._Dq * torch.abs(vel_diag)   # (num_envs, 6, 6)

        damp_6 = (damp_mat @ vel_b.unsqueeze(-1)).squeeze(-1)   # (num_envs, 6)

        return damp_6[:, :3], damp_6[:, 3:]

    def _added_mass(self, acc_b: torch.Tensor) -> tuple:
        """
        추가 질량력 (body frame).

        F_added = M_a · a
        """
        added_6 = (self._Ma @ acc_b.unsqueeze(-1)).squeeze(-1)  # (num_envs, 6)
        return added_6[:, :3], added_6[:, 3:]

    def _coriolis(self, vel_b: torch.Tensor) -> tuple:
        """
        Coriolis / 원심력 (body frame).

        Fossen (2011) C_A(v) · v 식 기반:
          f_cor = -ω × (M_a · v_lin)
          t_cor = -(M_a·v_lin × v_lin + M_a·v_ang × ω)
        """
        v_lin = vel_b[:, :3]   # (num_envs, 3)
        v_ang = vel_b[:, 3:]   # (num_envs, 3)

        # M_a · v  (6-DOF)
        Mav    = (self._Ma @ vel_b.unsqueeze(-1)).squeeze(-1)  # (num_envs, 6)
        Mav_l  = Mav[:, :3]
        Mav_a  = Mav[:, 3:]

        f_cor = -torch.cross(v_ang, Mav_l, dim=-1)
        t_cor = -(torch.cross(Mav_l, v_lin, dim=-1) + torch.cross(Mav_a, v_ang, dim=-1))

        return f_cor, t_cor

    def _update_acc(self, vel_b: torch.Tensor) -> torch.Tensor:
        """
        저역 필터로 body-frame 가속도 추정.

        filtered_acc = (1-α)·prev_acc + α·((v - v_prev) / dt)
        α = 0.3  (MarineGym 기본값)
        """
        raw_acc     = (vel_b - self._prev_vel_b) / self.dt
        filtered    = (1.0 - self._ACC_ALPHA) * self._prev_acc_b \
                      + self._ACC_ALPHA * raw_acc

        self._prev_vel_b = vel_b.detach().clone()
        self._prev_acc_b = filtered.detach().clone()

        return filtered
