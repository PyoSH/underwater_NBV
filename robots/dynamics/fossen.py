"""
Fossen (2011) 6-DOF 수중 유체역학 — 기종 무관 코어
=====================================================
`step_2_BROV/hydrodynamics.py`의 `BROV2Hydrodynamics`를 여기로 승격(2026-07).
기종 이름이 안 붙은 이유: 모든 계수(added mass/damping/부력/CoB)를 생성자
인자로 받으므로 BlueROV2든 다른 수중차량이든 계수만 바꾸면 그대로 재사용
가능 — step_2_BROV(현재)와 향후 step_3(수중 인지+물리 통합) 양쪽에서 공유.

기종별로 다른 것(스러스터/핀 등 액추에이터 모델, 위치·방향, YAML 파라미터
로더)은 여기 두지 않는다 — `robots/dynamics/<기종>/` 아래에 따로 둔다
(예: `robots/dynamics/brov2/thruster.py`).

좌표계
------
- 입출력 : Z-up body frame (IsaacLab)
- 내부 계산 : SNAME/NED body frame (X=전방, Y=우현, Z=하방)
- 변환 : T₃=diag(1,-1,-1) (SNAME b → Z-up b)

IsaacLab(PhysX) 담당: M_RB·ν̇, C_RB·ν, F_gravity
이 모듈 담당       : M_A·ν̇, C_A·ν, D·ν, 부력+복원모멘트

주의: 이 클래스를 쓰는 로봇의 USD에서 linear_damping=angular_damping=0 필수
(PhysX 기본 damping과 중복 적용 방지).
"""

import torch
from isaaclab.utils.math import quat_apply, quat_conjugate

# 좌표계 변환 상수 (Z-up body ↔ NED body)
_T3 = torch.tensor([1., -1., -1.])
_T6 = torch.tensor([1., -1., -1., 1., -1., -1.])


class Hydrodynamics:
    """
    Fossen (2011) 기반 6-DOF 수중 유체역학. 기종 무관 — 계수는 생성자 인자.

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
    _VOLUME        : float = 0.0134      # [m³] fallback 기본값
    _COB_VECTOR    : list  = [0.0, 0.0, -0.01]  # [m] SNAME frame, fallback only

    # fallback 계수 (인자 없이 직접 생성할 때만 사용 — von Benzon et al. 2022 Table A1)
    # 강체 질량/관성 fallback (brov2_heavy.yaml expect 절). M_total = M_RB + M_A 용.
    _RIGID_MASS        = 14.635
    _RIGID_INERTIA     = [0.289, 0.329, 0.337]

    _ADDED_MASS        = [6.36,  7.12,  18.68, 0.189, 0.135, 0.222]
    _LINEAR_DAMPING    = [13.70, 0.00,  33.00, 0.00,  0.80,  0.00 ]
    _QUADRATIC_DAMPING = [141.0, 217.0, 190.0, 1.19,  0.47,  1.50 ]

    def __init__(
        self,
        num_envs         : int,
        dt                : float,
        device            : str,
        volume            : float | torch.Tensor | None = None,
        cob_vector        : list | tuple | torch.Tensor | None = None,
        water_density     : float | None = None,
        added_mass        : list | tuple | None = None,
        linear_damping    : list | tuple | None = None,
        quadratic_damping : list | tuple | None = None,
        rigid_mass        : float | None = None,
        rigid_inertia     : list | tuple | None = None,
    ):
        self.num_envs = num_envs
        self.dt       = dt
        self.device   = device

        self._water_density = water_density or self._WATER_DENSITY
        V   = volume        if volume     is not None else self._VOLUME
        cob = cob_vector    if cob_vector is not None else self._COB_VECTOR

        # 변환 벡터
        self._t3 = _T3.to(device)
        self._t6 = _T6.to(device)

        # 부력 기준벡터 (Z-up world)
        self._world_up = torch.tensor([0., 0., 1.], device=device)

        # ── 부력/COB — env별 배치 텐서 (도메인 랜덤화 대상, randomize() 참조) ──
        # volume/cob_vector로 스칼라/리스트를 넘기면 전체 env에 broadcast, 이미
        # (num_envs,)/(num_envs,3) 텐서를 넘기면 그대로 사용 (velEnv 등에서 env별
        # 다른 값을 초기 지정하고 싶을 때 대비).
        self._volume = (
            V.to(device).float() if isinstance(V, torch.Tensor)
            else torch.full((num_envs,), float(V), device=device)
        )
        self._buoy_mag = self._water_density * self._GRAVITY * self._volume   # (num_envs,) [N]

        if isinstance(cob, torch.Tensor):
            self._r_cob_ned = cob.to(device).float()
            if self._r_cob_ned.dim() == 1:
                self._r_cob_ned = self._r_cob_ned.unsqueeze(0).repeat(num_envs, 1)
        else:
            self._r_cob_ned = (
                torch.tensor(cob, dtype=torch.float32, device=device)
                .unsqueeze(0).repeat(num_envs, 1)
            )   # (num_envs,3) NED(SNAME) body frame, COM 기준
        self._nominal_r_cob_ned = self._r_cob_ned.clone()   # randomize()의 오프셋 기준점

        # 6-DOF 대각 행렬 (num_envs, 6, 6)
        def _diag(coeffs):
            return (
                torch.diag(torch.tensor(coeffs, dtype=torch.float32))
                .unsqueeze(0).repeat(num_envs, 1, 1).to(device)
            )
        self._Ma = _diag(added_mass        or self._ADDED_MASS)
        self._Dl = _diag(linear_damping    or self._LINEAR_DAMPING)
        self._Dq = _diag(quadratic_damping or self._QUADRATIC_DAMPING)

        # ── M_total = M_RB + M_A — added mass를 암묵적으로 풀기 위한 것 ──
        # compute()가 ν̇ = M_total⁻¹·F 를 풀고 −M_A·ν̇ 를 외력으로 돌려준다.
        # 자세한 이유는 compute() docstring 참조.
        m = float(rigid_mass    if rigid_mass    is not None else self._RIGID_MASS)
        I = list(rigid_inertia  if rigid_inertia is not None else self._RIGID_INERTIA)
        if m <= 0.0 or any(v <= 0.0 for v in I) or len(I) != 3:
            raise ValueError("rigid_mass는 양수, rigid_inertia는 양수 3개여야 함")
        self._M_total = self._Ma + _diag([m, m, m] + I)

    def randomize(
        self,
        env_ids        : torch.Tensor,
        volume         : torch.Tensor | None = None,
        cob_offset     : torch.Tensor | None = None,
        added_mass_rot : torch.Tensor | None = None,
    ) -> None:
        """도메인 랜덤화 — env._reset_idx()에서 env_ids만 호출. None인 인자는 그대로 둔다.

        Parameters
        ----------
        volume         : (M,) [m^3] 절대값
        cob_offset     : (M,3) __init__ 시점 cob_vector(nominal) 기준 오프셋, NED/SNAME frame
        added_mass_rot : (M,3) Kṗ, Mq̇, Nṙ 절대값 (회전축 added mass)
        """
        if volume is not None:
            self._volume[env_ids]   = volume
            self._buoy_mag[env_ids] = self._water_density * self._GRAVITY * volume
        if cob_offset is not None:
            self._r_cob_ned[env_ids] = self._nominal_r_cob_ned[env_ids] + cob_offset
        if added_mass_rot is not None:
            for k in range(3):
                # M_total = M_RB + M_A 이므로 M_A를 흔들면 M_total도 따라가야 한다.
                # 안 그러면 암묵 해법이 흔들리기 전 질량으로 ν̇ 를 푼다.
                delta = added_mass_rot[:, k] - self._Ma[env_ids, 3 + k, 3 + k]
                self._Ma[env_ids, 3 + k, 3 + k] = added_mass_rot[:, k]
                self._M_total[env_ids, 3 + k, 3 + k] += delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        root_quat_w : torch.Tensor,   # (N, 4) [w,x,y,z]
        lin_vel_b   : torch.Tensor,   # (N, 3) Z-up body
        ang_vel_b   : torch.Tensor,   # (N, 3) Z-up body
        other_wrench_b: torch.Tensor, # (N, 6) Z-up body — 이 모듈 밖에서 몸체에
                                      #        작용하는 모든 것(추력 + 중력)
    ) -> tuple:
        """
        유체역학 합력/합토크 계산.

        added mass를 **암묵적으로** 푼다
        ------------------------------
        Fossen 방정식에서 M_A는 좌변의 질량행렬에 속한다:

            (M_RB + M_A)·ν̇ + C(ν)ν + D(ν)ν = τ_thrust + g(η) + F_grav

        2026-08-28까지는 이를 우변으로 옮겨 −M_A·ν̇ 를 외력으로 넣고 ν̇ 를
        **직전 스텝의 속도차**로 추정했다. 그 되먹임은 M_A ≳ M_RB 에서 발산하므로
        가속도에 저역필터(α=0.3, ~5.7 Hz)를 걸어 막고 있었다. 필터는 물리적
        근거가 없고 added mass를 주파수 의존으로 만든다 — 측정하니 0.5 Hz 이하
        1% 미만이지만 5 Hz에서 −21%, 10 Hz에서 −43%로, 10 Hz에서는 유효질량이
        18.74 kg(added mass가 거의 사라진 값)이 된다.

        지금은 ν̇ 를 추정하지 않고 **푼다**:

            ν̇        = (M_RB + M_A)⁻¹ · [τ_other + g(η) − C(ν)ν − D(ν)ν]
            τ_am     = −M_A · ν̇

        PhysX가 M_RB·ν̇_physx = τ_other + (이 함수의 반환값) 을 적분하므로,
        반환값에 τ_am 이 들어가면 ν̇_physx = (M_RB+M_A)⁻¹·F 가 정확히 나온다.
        가속도 되먹임이 없으니 불안정할 수 없고 필터도 필요 없다. 전 대역에서
        해석해와 0.5% 이내로 일치한다.

        ``other_wrench_b`` 는 **이 모듈 밖에서 몸체에 작용하는 전부**여야 한다 —
        추력과 중력. 빠뜨리면 ν̇ 가 틀리고 added mass가 그만큼 어긋난다.
        (중력은 PhysX가 따로 적용하므로 반환값에 넣지 않는다. 여기서는 ν̇ 를
        푸는 데만 쓴다.)

        Returns
        -------
        forces_zup  : (N, 3) [N]    Z-up body frame
        torques_zup : (N, 3) [N·m]  Z-up body frame
        """
        # ── 입력 변환: Z-up body → NED body ──────────────────────────
        vel_zup = torch.cat([lin_vel_b, ang_vel_b], dim=-1)   # (N, 6)
        vel_ned = vel_zup * self._t6                           # (N, 6)
        other_ned = other_wrench_b * self._t6                  # (N, 6)

        # ── Fossen 계산 (전부 NED body frame) ─────────────────────────
        g_ned    = self._buoyancy(root_quat_w)   # (N, 6)
        fd_ned   = self._damping(vel_ned)         # (N, 6)
        fcor_ned = self._coriolis(vel_ned)        # (N, 6)

        # ── added mass: ν̇ 를 풀어서 구한다 ───────────────────────────
        rhs      = other_ned + g_ned - fd_ned - fcor_ned          # (N, 6)
        acc_ned  = torch.linalg.solve(self._M_total, rhs.unsqueeze(-1)).squeeze(-1)
        fam_ned  = self._added_mass(acc_ned)      # (N, 6)

        # ── NED body frame 합산 ────────────────────────────────────────
        total_ned = g_ned - (fd_ned + fam_ned + fcor_ned)   # (N, 6)

        # ── 출력 변환: NED body → Z-up body ──────────────────────────
        total_zup = total_ned * self._t6

        return total_zup[:, :3], total_zup[:, 3:]

    def reset(self, env_ids: torch.Tensor) -> None:
        # 가속도 되먹임이 없어져 남길 상태가 없다.
        return

    # ------------------------------------------------------------------
    # Private: Fossen 각 항 계산 (모두 NED body frame, 6-벡터 반환)
    # ------------------------------------------------------------------

    def _buoyancy(self, root_quat_w: torch.Tensor) -> torch.Tensor:
        """
        g(η) = [F_buoy_ned; r_COB_ned × F_buoy_ned]

        F_buoy_W(Z-up) → Z-up body(쿼터니언 회전) → NED body(T₃)
        τ_restore = r_COB_NED × F_buoy_NED
        """
        # 1. 부력: Z-up world → Z-up body (env별 배치 크기 self._buoy_mag)
        f_world = self._world_up.unsqueeze(0) * self._buoy_mag.unsqueeze(-1)   # (N, 3)
        f_zup   = quat_apply(quat_conjugate(root_quat_w), f_world)             # (N, 3)

        # 2. Z-up body → NED body
        f_ned = f_zup * self._t3   # (N, 3)

        # 3. 복원 모멘트: r_COB_NED × F_buoy_NED (env별 배치)
        t_ned = torch.cross(self._r_cob_ned, f_ned, dim=-1)   # (N, 3)

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

        # 힘 항은 −(A₁₁ν₁) × ν₂ 다. 2026-08-28까지 −(ν₂ × A₁₁ν₁), 즉 **부호가
        # 반대**였다. 두 가지로 확인했다:
        #   ① 스큐대칭  ν·(C_A ν) = 0 이어야 하는데 30.07이 나왔다(에너지 생성).
        #      힘 항만 뒤집으면 1.4e-06으로 떨어진다.
        #   ② 물리 유도  전진 u·우선회 r 에서 부가질량 운동량 p = A₁₁ν₁ 를 회전
        #      시키는 힘은 ω × p 이고, 반작용으로 물이 기체를 미는 힘은 −(ω × p)
        #      즉 **선회 바깥**이다(원심 반작용). 적용값이 −C_A·ν 이므로
        #      C_A·ν|force = +(ω × p) 여야 한다. 구 구현은 −(ω × p)라 기체를
        #      선회 **안쪽으로 당겼다**.
        # 크기는 선회율에 비례한다 — 순항 0.5 m/s에서 sway 예산의 0.4%(0.05 rad/s)
        # ~30%(4.05 rad/s, trial (a) 실측 최대). 직진에서는 무시할 만하다.
        f_cor = -torch.cross(Mav_l, v_ang, dim=-1)
        t_cor = -(torch.cross(Mav_l, v_lin, dim=-1) + torch.cross(Mav_a, v_ang, dim=-1))

        return torch.cat([f_cor, t_cor], dim=-1)   # (N, 6)

