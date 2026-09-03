"""
step_3_NBV_BROV — 물리 기반 능동인지(NBV) 환경
=================================================
step_1_NBV(카메라 순간이동, TSDF/coverage 관측)과 step_2_BROV(BROV2 물리,
Fossen 동역학+8-스러스터)를 통합한 Stage 1 스켈레톤. 아키텍처 결정 배경은
`.claude/plans/kind-launching-kahan.md` 참조.

계층 구조
---------
NBV 정책 action(연속 3-dim) → 목표 구면좌표(θ,φ,ψ) 갱신 → 목표점(p_target,
q_target) → NBVGuidance(상태 홀더) → DPController(6-DOF 위치/자세 PID,
Dynamic Positioning) → B_pinv 할당 → 8-thruster PWM → BROV2ThrusterModel +
Hydrodynamics(robots/dynamics/, step_2와 완전 동일 재사용)

step_1과의 핵심 차이: 카메라가 로봇 동체에 물리적으로 고정 부착돼 있어
"시점을 바꾸려면 로봇 전체가 이동/회전"해야 한다 — `write_root_state_to_sim()`
순간이동 없음. 저수준은 RL이 아니라 classical PID(DP)라 frozen 정책의
분포이탈 위험이 없다.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Sequence

import torch
import torch.nn.functional as F
import omni.usd
from pxr import UsdLux

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

# 한 단계 위(step_3_NBV_BROV) = envs./guidance./control. 절대 dotted-import 루트,
# 두 단계 위(repo root) = robots.dynamics.* 루트 — step_2_BROV/envs/vel_env.py와
# 동일한 sys.path 관례(상대 import `.` 대신 절대 dotted import 사용).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from envs.env_cfg import NBVBROVEnvCfg
from envs.env_reward import EnvRewardMixin
from envs.env_utils import EnvUtilsMixin
from envs.scene_cfg import _CAMERA_FRAME_POS
from guidance.nbv_guidance import NBVGuidance
from control.dp_controller import DPController

from robots.dynamics.brov2.thruster import BROV2ThrusterModel, build_allocation_matrix
from robots.dynamics.brov2.mass_randomization import randomize_articulation_mass
from robots.dynamics.brov2.params import load_brov2_yaml, coBM_vector_ned, thruster_pos_dir_ned
from robots.dynamics.fossen import Hydrodynamics

# Z-up body(IsaacLab, FLU) → SNAME/FRD body(할당행렬 B가 기대하는 프레임) 부호
# 변환. robots/dynamics/fossen.py의 `_T6`, step_2_BROV/action_frame_contract.py의
# `T6_DIAGONAL`과 동일한 상수(diag(1,-1,-1) Y/Z 반전 = X축 기준 180도 고유회전이라
# 힘/토크 둘 다 같은 부호로 변환됨) — step_3가 step_2_BROV에 직접 의존하지
# 않도록 여기 독립 재선언(값 자체가 바뀔 일이 없는 좌표계 상수라 중복 비용 낮음).
_ZUP_TO_SNAME_SIGN = (1.0, -1.0, -1.0, 1.0, -1.0, -1.0)


def _sample_from_sphere(n: int, r: float, device) -> torch.Tensor:
    """반경 r 구 내부 균등(부피 기준) 샘플. robots/dynamics 도메인랜덤화 공통 패턴."""
    coords = torch.randn(n, 3, device=device)
    coords = coords / coords.norm(dim=1, keepdim=True)
    radii = r * torch.rand(n, 1, device=device).pow(1 / 3)
    return radii * coords


class NBVBROVEnv(EnvUtilsMixin, EnvRewardMixin, DirectRLEnv):
    """물리 기반 BROV2 NBV 능동인지 환경."""

    cfg: NBVBROVEnvCfg

    def __init__(self, cfg: NBVBROVEnvCfg, render_mode: str | None = None):
        # 카메라 viewport는 씬 생성(super().__init__) 시점에 카메라가 스폰되므로
        # 반드시 그 **이전에** 설정해야 한다 — step_1_NBV/env/env.py의 debug_vis
        # 분기와 동일한 이유/순서.
        if cfg.enable_camera_viewport or cfg.debug_vis:
            cfg.scene.camera.enable_viewport = True
            cfg.scene.camera.viewport_env_id = cfg.camera_viewport_env_id

        # Stage 4 다중 대상 물체 — 씬 생성 **이전**에 스포너를 교체해야 한다.
        # `use_mesh_pool()`이 replicate_physics=False도 함께 설정한다.
        if cfg.mesh_pool_manifest:
            from envs.mesh_pool import load_mesh_pool
            # 풀 크기를 env 수로 제한하는 것이 **기본**이다.
            #
            # IsaacLab의 `spawn_multi_usd_file()`은 (1) 리스트의 **모든** 자산을
            # 템플릿 프림으로 먼저 스폰한 뒤 (2) env i에 `pool[i % N]`을 복사한다.
            # 즉 초과분은 학습에 한 번도 등장하지 않으면서 startup에 USD 로딩
            # 비용만 얹는다(GSO 700개 = 14 GB). 또한 이 배정은 씬 생성 시
            # **한 번**뿐이라, 학습 중 실제로 보는 물체 수 = min(num_envs, 풀).
            limit = cfg.mesh_pool_limit or cfg.scene.num_envs
            _pool = load_mesh_pool(
                cfg.mesh_pool_manifest,
                filter_flat=cfg.mesh_pool_filter_flat,
                min_aspect=cfg.mesh_pool_min_aspect,
                limit=limit,
                offset=cfg.mesh_pool_offset,
                split=cfg.mesh_pool_split,
                n_holdout=cfg.mesh_pool_holdout,
            )
            cfg.scene.use_mesh_pool(_pool)
            # 학습이 실제로 만나는 물체 종류 수. env i는 `pool[i % N]`을 씬
            # 생성 시 한 번 배정받고 끝이므로 이 값이 다양성의 상한이다.
            self._n_mesh_objects = min(len(_pool), cfg.scene.num_envs)
        else:
            self._n_mesh_objects = 1   # 단일 rock — 모든 env가 같은 물체

        # 소나 비활성화도 씬 생성 전에 결정해야 한다(센서가 여기서 스폰됨).
        # 관측에 안 쓰이면서 카메라의 32배 픽셀을 렌더링하므로 기본 off —
        # `env_cfg.enable_sonar` 주석 참조.
        if not cfg.enable_sonar:
            cfg.scene.sonar = None

        super().__init__(cfg, render_mode)

        self._robot: Articulation = self.scene.articulations["robot"]
        self._policy_dt = cfg.sim.dt * cfg.decimation

        # ── 물리 (step_2_BROV/envs/vel_env.py와 동일 구성) ──
        yaml_params = load_brov2_yaml()
        hydro_coef = yaml_params["hydro_coef"]
        cob_vector = coBM_vector_ned(yaml_params)
        thr_pos, thr_dir = thruster_pos_dir_ned(yaml_params)

        self._thruster = BROV2ThrusterModel(
            self.num_envs, cfg.sim.dt, self.device, pos=thr_pos, dir=thr_dir,
        )
        self._hydro = Hydrodynamics(
            self.num_envs, cfg.sim.dt, self.device,
            volume=yaml_params["volume"],
            cob_vector=cob_vector,
            water_density=yaml_params["environment"]["fluid_density"],
            added_mass=hydro_coef["added_mass"],
            linear_damping=hydro_coef["linear_damping"],
            quadratic_damping=hydro_coef["quadratic_damping"],
        )
        # `_apply_action()`이 중력 wrench를 만들 때 쓴다 — `Hydrodynamics.compute()`가
        # ν̇ 를 풀려면 추력과 중력이 모두 필요하다(step_2 `vel_env.py`와 동일 관례).
        self._rigid_mass = float(yaml_params["expect"]["mass"])
        B = build_allocation_matrix(self._thruster._pos, self._thruster._dir)
        self._B_pinv = torch.linalg.pinv(B.to(self.device))
        self._zup_to_sname_sign = torch.tensor(_ZUP_TO_SNAME_SIGN, device=self.device)
        self._nominal_added_mass_rot = torch.tensor(hydro_coef["added_mass"][3:], device=self.device)

        # ── 저수준: NBV 목표 상태홀더 + DP(PID) 컨트롤러 ──
        self._guidance = NBVGuidance(self.num_envs, self.device)
        self._dp = DPController(
            self.num_envs, cfg.sim.dt, self.device,
            kp_pos=cfg.dp_kp_pos, ki_pos=cfg.dp_ki_pos, kd_pos=cfg.dp_kd_pos,
            kp_att=cfg.dp_kp_att, ki_att=cfg.dp_ki_att, kd_att=cfg.dp_kd_att,
            tau_max=cfg.dp_tau_max,
            integral_pos_limit=cfg.dp_integral_pos_limit,
            integral_att_limit=cfg.dp_integral_att_limit,
        )

        # ── NBV 목표 구면좌표 (rock 중심 기준) ──
        self._sph_theta = torch.zeros(self.num_envs, device=self.device)
        self._sph_phi = torch.zeros(self.num_envs, device=self.device)
        self._sph_psi = torch.zeros(self.num_envs, device=self.device)
        self._actions = torch.zeros(self.num_envs, cfg.action_space, device=self.device)

        rock_local = torch.tensor([0.0, 0.0, -3.0], device=self.device)
        self.rock_pos = self.scene.env_origins + rock_local

        # 차량 조명 단계 (step_1과 동일 구조) — scene_cfg는 intensity=0.0으로
        # 선언하고 실제 값은 `_update_light_intensity()`가 런타임에 설정한다.
        self._light_level = torch.full(
            (self.num_envs,), cfg.light_level_init, dtype=torch.long, device=self.device
        )

        # ── 관측 버퍼 ──
        self._image_buffer = torch.zeros(
            (self.num_envs, cfg.visual.num_seq_actor, cfg.visual.h, cfg.visual.w), device=self.device
        )
        self._depth_buffer = torch.zeros(
            (self.num_envs, cfg.visual.num_seq_critic, cfg.visual.h, cfg.visual.w), device=self.device
        )

        # ── TSDF/coverage ──
        Nx, Ny, Nz = cfg.tsdf.vol_dim
        self._tsdf_vol = torch.zeros(self.num_envs, Nx, Ny, Nz, device=self.device)
        self._weight_vol = torch.zeros(self.num_envs, Nx, Ny, Nz, device=self.device)
        self._vol_origin = torch.zeros(self.num_envs, 3, device=self.device)
        self._total_surf_voxels = torch.ones(self.num_envs, device=self.device)
        self._surf_vol = torch.zeros(self.num_envs, Nx, Ny, Nz, dtype=torch.bool, device=self.device)

        self.curr_coverage = torch.zeros(self.num_envs, device=self.device)
        # ── 적응형 커리큘럼 상태 ──
        # 임계값은 시작값에서 출발해 성공률이 게이트를 넘을 때만 오른다.
        self._curriculum_level = cfg.curriculum_coverage_terminal_start
        self.curriculum_success_ema = 0.0
        # `_get_dones()`가 매 스텝 갱신. 첫 `_reset_idx()`는 `_get_dones()` 이전에
        # 불릴 수 있으므로(초기 reset) 여기서 False로 초기화해 둔다.
        self._last_coverage_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # 에피소드 종료 시점의 coverage (리셋 전 값) — 학습 스크립트 로깅용.
        # `_reset_idx()` 주석 참조.
        self.terminal_coverage = torch.zeros(self.num_envs, device=self.device)
        self._prev_robot_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # 진단 지표 (보상에는 미기여, `_get_rewards()`가 매 스텝 갱신)
        self.last_dist_moved = torch.zeros(self.num_envs, device=self.device)
        self.last_reward_terms: dict[str, torch.Tensor] = {}

        # ── Quality-weighted coverage (env_cfg.use_quality_coverage 참조) ──
        self._quality_vol = torch.zeros(self.num_envs, Nx, Ny, Nz, device=self.device)
        # 보상 delta의 기준 — binary/quality 중 실제로 쓰는 쪽의 정규화값을 담는다.
        self._prev_coverage_norm = torch.zeros(self.num_envs, device=self.device)
        self.curr_coverage_q = torch.zeros(self.num_envs, device=self.device)
        self.terminal_coverage_q = torch.zeros(self.num_envs, device=self.device)
        # ②a 오염기. `cfg.corruption.enabled=False`면 모든 경로가 항등이라
        # 학습 경로의 동작은 변하지 않는다.
        from envs.depth_corruption import DepthCorruptor
        self._corruptor = DepthCorruptor(cfg.corruption, self.num_envs, self.device)

        # μ는 카메라 실제 감쇠계수에서 유도한다 — `_sync_quality_water()`.
        # 여기서는 형태만 잡아두고 첫 리셋에서 실제 값으로 덮인다.
        self._quality_mu = torch.full((self.num_envs,), 0.1, device=self.device)
        # **voxel별 달성 가능 최대 품질** q*(v) — (A) 정규화의 분모.
        # 전역 스칼라 Q_sat=exp(-μ·psi_min)을 대체한다(`_update_q_star()` 참조).
        self._q_star = torch.ones(self.num_envs, Nx, Ny, Nz, device=self.device)
        # GT surface voxel의 품질 분포 진단 (step_1 diag/gt_* 대응).
        # binary coverage로는 안 보이는 "봤지만 멀어서 흐릿함"을 드러낸다.
        self._diag_gt_never = torch.zeros(self.num_envs, device=self.device)
        self._diag_gt_partial = torch.zeros(self.num_envs, device=self.device)
        self._diag_gt_full = torch.zeros(self.num_envs, device=self.device)
        # voxel **중심** 오프셋 (원점 기준 격자, 리셋 무관 상수).
        # `+ voxel_size/2`가 필요하다 — 인덱스 i의 voxel이 덮는 구간은
        # [origin+i·vox, origin+(i+1)·vox)이므로 중심은 origin+(i+0.5)·vox다.
        # TSDF 적분(`env_reward.py::_vox_local`)은 처음부터 이 보정을 갖고
        # 있었는데 품질 계산 쪽만 빠져 있어 반 voxel 어긋나 있었다. voxel이
        # 10 cm로 커지면서 편차가 5 cm가 되므로 여기서 맞춘다.
        gx, gy, gz = torch.meshgrid(
            torch.arange(Nx, device=self.device),
            torch.arange(Ny, device=self.device),
            torch.arange(Nz, device=self.device),
            indexing="ij",
        )
        self._voxel_offset = (
            torch.stack([gx, gy, gz], dim=-1).float() * cfg.tsdf.voxel_size
            + cfg.tsdf.voxel_size / 2.0
        )

        self._mass_scale = torch.ones(self.num_envs, 1, device=self.device)

        # RigidObject 핸들 (InteractiveScene이 자동 생성) — EnvUtilsMixin의
        # `_build_cam_pose()`, `_get_observations/_get_rewards`가 사용
        self._camera = self.scene["camera"]
        self._sonar = self.scene["sonar"] if cfg.enable_sonar else None

        # `_build_cam_pose()`가 로봇 root pose에서 카메라 pose를 유도할 때 쓰는
        # 고정 오프셋 — scene_cfg.py의 카메라 offset.pos와 반드시 일치해야 한다.
        self._camera_offset_pos = torch.tensor(
            _CAMERA_FRAME_POS, device=self.device, dtype=torch.float32
        )

    # ── 씬 구성 ──────────────────────────────────────────────────────────────
    def _setup_scene(self) -> None:
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        stage = omni.usd.get_context().get_stage()
        from pxr import Usd, UsdGeom
        from isaacsim.core.utils.semantics import add_update_semantics

        for env_idx in range(self.num_envs):
            env_ns = f"/World/envs/env_{env_idx}"
            for light_path in (f"{env_ns}/Robot/SphereLight_L", f"{env_ns}/Robot/SphereLight_R"):
                light = stage.GetPrimAtPath(light_path)
                if light.IsValid():
                    shaping = UsdLux.ShapingAPI.Apply(light)
                    shaping.GetShapingConeAngleAttr().Set(40.0)
                    shaping.GetShapingConeSoftnessAttr().Set(0.1)

            obj_prim = stage.GetPrimAtPath(f"{env_ns}/Object")
            if obj_prim.IsValid():
                for prim in Usd.PrimRange(obj_prim):
                    if UsdGeom.Mesh(prim):
                        add_update_semantics(prim, type_label="reflectivity", semantic_label="1.0")

    def _update_light_intensity(self, light_level: torch.Tensor) -> None:
        """차량 SphereLight 2개의 intensity를 단계값에 따라 설정.

        `step_1_NBV/env/env.py::_update_light_intensity()` 이식 — 조명 프림 경로만
        step_3 구조(Robot의 자식)에 맞게 바꿨다. USD 속성 직접 쓰기라 env 수만큼
        Python 루프를 돌지만, 리셋 시에만 호출되므로 학습 루프 비용은 아니다.
        """
        stage = omni.usd.get_context().get_stage()
        for env_idx in range(self.num_envs):
            intensity = float(
                light_level[env_idx].item() * self.cfg.light_intensity_per_level
            )
            env_ns = f"/World/envs/env_{env_idx}"
            for name in ("SphereLight_L", "SphereLight_R"):
                prim = stage.GetPrimAtPath(f"{env_ns}/Robot/{name}")
                if not prim.IsValid():
                    continue
                attr = prim.GetAttribute("inputs:intensity")
                if attr.IsValid():
                    attr.Set(intensity)

    # ── 사전 물리 스텝 (정책스텝당 1회): NBV 목표 갱신 ────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clamp(-1.0, 1.0)

        delta_theta = self._actions[:, 0] * self.cfg.max_rate_theta
        delta_phi = self._actions[:, 1] * self.cfg.max_rate_phi
        delta_psi = self._actions[:, 2] * self.cfg.max_rate_psi

        self._sph_theta = (self._sph_theta + delta_theta) % (2 * math.pi)
        self._sph_phi = (self._sph_phi + delta_phi).clamp(self.cfg.phi_min, self.cfg.phi_max)
        self._sph_psi = (self._sph_psi + delta_psi).clamp(self.cfg.psi_min, self.cfg.psi_max)

        offset = torch.stack([
            self._sph_psi * torch.sin(self._sph_phi) * torch.cos(self._sph_theta),
            self._sph_psi * torch.sin(self._sph_phi) * torch.sin(self._sph_theta),
            self._sph_psi * torch.cos(self._sph_phi),
        ], dim=-1)

        p_target = self.rock_pos + offset
        q_target = self._look_at_quat(p_target, self.rock_pos)

        all_ids = torch.arange(self.num_envs, device=self.device)
        self._guidance.set_target(all_ids, p_target, q_target)

    # ── 물리 서브스텝(decimation회 반복): DP → B_pinv → 스러스터/유체역학 ──────
    def _apply_action(self) -> None:
        wrench_zup = self._dp.compute(
            pos_target_w=self._guidance.p_target,
            quat_target_w=self._guidance.q_target,
            root_pos_w=self._robot.data.root_pos_w,
            root_quat_w=self._robot.data.root_quat_w,
            lin_vel_b=self._robot.data.root_lin_vel_b,
            ang_vel_b=self._robot.data.root_ang_vel_b,
        )   # (N,6) Z-up body frame

        tau_cmd_sname = wrench_zup * self._zup_to_sname_sign
        f_desired = (self._B_pinv @ tau_cmd_sname.unsqueeze(-1)).squeeze(-1)
        f_limited = self._thruster.clamp_thrust(f_desired)
        pwm = self._thruster.inverse_thrust(f_limited)

        f_thrust, t_thrust = self._thruster.compute(pwm)

        # `Hydrodynamics.compute()`는 ν̇ 를 **풀기** 위해 이 모듈 밖에서 몸체에
        # 작용하는 전부(추력 + 중력)를 받아야 한다(2026-08-28 fossen.py 개정).
        # 빠뜨리면 ν̇ 가 틀리고 added mass가 그만큼 어긋난다. 중력은 PhysX가 따로
        # 적용하므로 여기서는 ν̇ 를 푸는 데만 쓰이고 반환값에는 포함되지 않는다.
        #
        # step_2 `vel_env.py`는 공칭 질량 상수를 쓰지만, step_3는 질량 DR이
        # 있으므로 실제 스케일된 질량을 쓴다(`dr_enable_mass=False`면 scale=1.0).
        g_world = torch.zeros_like(f_thrust)
        g_world[:, 2] = -self._rigid_mass * self._mass_scale.squeeze(-1) * 9.81
        g_body = math_utils.quat_apply(
            math_utils.quat_conjugate(self._robot.data.root_quat_w), g_world
        )
        other_wrench_b = torch.cat((f_thrust + g_body, t_thrust), dim=-1)

        f_hydro, t_hydro = self._hydro.compute(
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            other_wrench_b,
        )

        total_forces = (f_thrust + f_hydro).unsqueeze(1)
        total_torques = (t_thrust + t_hydro).unsqueeze(1)
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=total_forces, torques=total_torques, body_ids=[0]
        )

    # ── 관측 ─────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        raw_rgb = self._camera.data.output["uw_rgb"][:, :, :, :3]
        raw_depth = self._camera.data.output["distance_to_camera"]

        curr_obs = torch.mean(raw_rgb.float(), dim=-1) / 255.0
        curr_state = raw_depth.squeeze(-1)

        curr_obs = F.interpolate(
            curr_obs.unsqueeze(1), size=(self.cfg.visual.h, self.cfg.visual.w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        curr_state = F.interpolate(
            curr_state.unsqueeze(1), size=(self.cfg.visual.h, self.cfg.visual.w),
            mode="nearest",
        ).squeeze(1)

        self._image_buffer = torch.roll(self._image_buffer, shifts=-1, dims=1)
        self._image_buffer[:, -1, :, :] = curr_obs
        self._depth_buffer = torch.roll(self._depth_buffer, shifts=-1, dims=1)
        self._depth_buffer[:, -1, :, :] = curr_state

        # 스칼라 관측은 **명령한** 구면좌표(`_sph_*`)가 아니라 로봇이 **실제로
        # 도달한** pose를 역산해서 넣는다(2026-08-26 수정).
        #
        # 이유: 보상은 `_integrate_depth()`→`_build_cam_pose()`로 실제 카메라
        # pose에서 계산되는데, 관측만 명령값을 주면 컨트롤러가 목표에 정확히
        # 도달하지 못했을 때 "같은 관측+같은 액션인데 보상이 다른" 상태가 된다
        # — RL의 credit assignment가 깨지는 전형적 형태. 관측과 보상이 같은
        # pose를 가리키게 만드는 것이 옳다.
        #
        # `_apply_action()`의 구면→직교 변환(offset = [ψsinφcosθ, ψsinφsinθ,
        # ψcosφ])의 정확한 역변환.
        # ②a: pose 드리프트가 켜져 있으면 로봇이 **믿는** 위치를 쓴다. 융합과
        # 같은 pose를 써야 "관측과 보상이 같은 pose를 가리킨다"는 아래 원칙이
        # 오염 하에서도 유지된다 — 관측만 참값이면 정책은 실기에 없는 정보를
        # 공짜로 받게 되고, ②a가 실제보다 관대한 시험이 된다.
        rel = (self._robot.data.root_pos_w
               + self._corruptor.pos_offset) - self.rock_pos        # (N,3)
        psi_actual = rel.norm(dim=-1).clamp_min(1e-6)
        phi_actual = torch.acos((rel[:, 2] / psi_actual).clamp(-1.0, 1.0))
        theta_actual = torch.atan2(rel[:, 1], rel[:, 0]) % (2 * math.pi)

        cfg = self.cfg
        scalar_obs = torch.stack([
            theta_actual / (2 * math.pi),
            (phi_actual - cfg.phi_min) / (cfg.phi_max - cfg.phi_min),
            (psi_actual - cfg.psi_min) / (cfg.psi_max - cfg.psi_min),
        ], dim=-1)

        # ── voxel 관측 (2026-08-26 추가, 이식 누락 정정) ──
        # step_1의 실제 성과 구성(`env_GenNBV.py::_get_vox_actor()`)을 이식.
        # 초기 이식 때 step_1의 *베이스* env.py 관측(이미지만)을 가져오고 이
        # 오버라이드를 빠뜨렸는데, **voxel 그리드가 "무엇을 이미 봤는지"를 정책에
        # 알려주는 유일한 수단**이라 이게 없으면 NBV 학습 자체가 성립하지 않는다
        # (정책이 최근 이미지와 자기 좌표만 보고 커버리지 상태를 모름).
        vox_actor = self._get_vox_actor()

        return {
            "policy": self._image_buffer,
            "extra_info": scalar_obs,
            "critic": self._depth_buffer,
            # step_1과 동일: critic scalar만 coverage를 추가로 받아 4-dim
            # (actor는 3-dim 유지 — 커버리지 스칼라를 직접 주면 voxel에서
            #  읽어내야 할 정보를 우회로로 흘리게 됨)
            # coverage는 **보상·종료가 실제로 쓰는 쪽**(quality 모드면 정규화
            # coverage_q)을 줘야 한다 — 여기에 binary를 주면 critic이 리턴을
            # 좌우하는 양과 다른 양을 보게 돼 value 예측이 불필요하게 어려워진다.
            "critic_scalar": torch.cat(
                [scalar_obs, self._coverage_for_reward().unsqueeze(-1)], dim=-1
            ),
            "vox_actor": vox_actor,
            # step_1은 최근 2프레임만 semantic 입력으로 씀 — 전체 버퍼(6프레임)를
            # 그대로 주는 것보다 입력 차원이 작고, step_1이 실제로 학습에 쓴 형태다.
            "img_semantic": self._image_buffer[:, -2:, :, :].clone(),
        }

    def _get_vox_actor(self) -> torch.Tensor:
        """3-state voxel 관측 (E, 3, Nx, Ny, Nz).

        `step_1_NBV/env/env_GenNBV.py::_get_vox_actor()` 무수정 이식 — TSDF/weight
        볼륨은 step_3도 동일하게 갖고 있어 로직 변경이 필요 없다.

        ch0 unknown  : weight == 0          (아직 관측 안 됨)
        ch1 free     : weight > 0, tsdf > 0 (카메라 앞 빈 공간)
        ch2 occupied : weight > 0, tsdf ≤ 0 (표면/물체)
        """
        observed = self._weight_vol > 0                      # (E, Nx, Ny, Nz)
        if self.cfg.use_quality_coverage:
            # ch2를 이진 occupied가 아니라 **연속 관측 품질**로 준다 — 정책이
            # "봤다/못 봤다"뿐 아니라 "얼마나 가까이서 봤나"를 구분할 수 있어야
            # 거리를 줄이는 행동을 학습할 수 있다(step_1 ch2와 동일).
            ch2 = (self._quality_vol / self._q_star).clamp(0.0, 1.0)
        else:
            ch2 = (observed & (self._tsdf_vol <= 0)).float()
        return torch.stack([
            (~observed).float(),                             # ch0: unknown
            (observed & (self._tsdf_vol > 0)).float(),       # ch1: free
            ch2,                                             # ch2: 품질 또는 occupied
        ], dim=1)

    def _current_coverage_terminal(self) -> float:
        """커리큘럼: coverage 성공 임계값.

        **적응형(기본)**: 최근 성공률이 `curriculum_success_gate`를 넘을 때만
        임계값을 올린다. 스텝 기반 선형 상향은 정책 성능과 무관하게 진행돼
        구조적으로 정책을 앞지를 수 있고, 실제로 그랬다 — 2026-08-26 9.3시간
        런에서 임계값은 0.450→0.649(+0.199) 올라갔는데 coverage는 0.499→0.560
        (+0.061)에 그쳐 **3.3배 앞질렀고**, 후반에는 도달 불가능한 목표가 됐다.
        성공률로 게이팅하면 정의상 앞지를 수 없고, 종료값이 맞는지 추측할
        필요도 없어진다.

        `curriculum_adaptive=False`면 기존 스텝 기반 선형 상향으로 되돌아간다
        (A/B 비교용). 둘 다 비활성이면 고정값 — `vel_env.py::
        _current_action_envelope()`와 동일 관례.
        """
        cfg = self.cfg
        if not cfg.curriculum_enabled:
            return cfg.coverage_terminal
        if cfg.curriculum_adaptive:
            return self._curriculum_level
        if cfg.curriculum_total_steps <= 0:
            return cfg.coverage_terminal
        progress = min(1.0, self.common_step_counter / cfg.curriculum_total_steps)
        return (
            cfg.curriculum_coverage_terminal_start
            + progress
            * (cfg.curriculum_coverage_terminal_end - cfg.curriculum_coverage_terminal_start)
        )

    def _update_curriculum(self, env_ids: torch.Tensor) -> None:
        """방금 끝난 에피소드들의 성공 여부로 난이도를 조절한다.

        상승폭을 `len(env_ids)/num_envs`로 비례시키는 이유: `_reset_idx()`는 매
        스텝 "그때 끝난 env들"에 대해 불리므로 호출 횟수가 에피소드 길이와
        env 수에 따라 크게 달라진다. 호출당 고정폭으로 올리면 난이도 상승 속도가
        그 부수적 요인에 좌우된다. 비례시키면 `curriculum_rate`가 "num_envs개
        에피소드가 끝날 때마다의 상승폭"이라는 안정된 의미를 갖는다.

        임계값은 **단조 증가**만 한다(내려가지 않음) — 난이도가 오르내리면
        보상 분포가 비정상(non-stationary)이 돼 critic이 쫓아가기 어렵다.
        """
        cfg = self.cfg
        succ = self._last_coverage_reached[env_ids].float().mean().item()
        a = cfg.curriculum_success_ema_alpha
        self.curriculum_success_ema = (1.0 - a) * self.curriculum_success_ema + a * succ

        if self.curriculum_success_ema >= cfg.curriculum_success_gate:
            frac = len(env_ids) / self.num_envs
            self._curriculum_level = min(
                cfg.curriculum_coverage_terminal_end,
                self._curriculum_level + cfg.curriculum_rate * frac,
            )

    def _sync_quality_water(self, env_ids: torch.Tensor) -> None:
        """μ를 카메라의 실제 감쇠계수에서 유도한다.

        μ = atten_coeff의 채널 평균. 렌더러가 실제로 쓰는 감쇠와 품질 모델의
        감쇠가 어긋나면 "보상이 재는 것"과 "화면에 보이는 것"이 달라지므로
        같은 출처에서 유도한다.

        `_atten_coeff_np`는 (N,3) numpy로 카메라 생성 시 cfg에서 초기화되므로
        첫 render 이전에도 안전하다(`_atten_coeff_t`는 None일 수 있음).
        """
        ids = env_ids.tolist()
        mu_np = self._camera._atten_coeff_np[ids].mean(axis=1)
        mu = torch.from_numpy(mu_np).to(self.device).float()
        self._quality_mu[env_ids] = mu

    def _update_q_star(self, env_ids: torch.Tensor) -> None:
        """voxel별 달성 가능 최대 품질 q*(v)를 갱신한다 — (A) 정규화의 분모.

            q*(v) = exp(−μ · max(psi_min − r_v, d_near)),   r_v = |v − 물체중심|

        왜 전역 Q_sat이 아니라 voxel별인가
        ----------------------------------
        기존 정규화는 `Q_sat = exp(−μ·psi_min)`이라는 **전역 상수**였다. psi는
        물체 *중심*까지의 거리라, 카메라 쪽을 향한 표면 voxel은 psi_min보다
        가깝고 따라서 품질비가 1을 넘는다. 합산이 clamp보다 먼저 일어나므로
        그 초과분이 **한 번도 못 본 voxel을 상쇄**한다 — 2026-09-02 Stage 2
        평가에서 실측으로 확인됐다(정책의 cov_q/cov_bin = 1.083, psi가 하한
        1.02에 고착, 관측 표면량은 고정 orbit과 동일). 즉 "가까이 붙기"가
        "돌아보기"를 대체할 수 있는 경로가 열려 있었다.

        voxel마다 **그 voxel을 가장 잘 볼 수 있는 자세에서의 품질**로 나누면
        모든 voxel의 상한이 1로 같아진다. 상한 1을 받으려면 각 voxel을 각각
        가까이서 봐야 하고, 그러려면 물체 주위를 돌아야 한다 — 근접이
        시점 선택을 대체할 수 없게 된다.

        d_near는 하한 가드다. r_v가 psi_min에 가까운 voxel(큰 물체의 근측면)은
        `psi_min − r_v → 0`이 되어 "거리 0에서 봐야 만점"이라는 달성 불가능한
        분모가 되기 때문이다. DP 추종오차가 0.18 m 수준이라 그보다 가까운
        거리는 실제로 만들 수 없다.

        **수질 불변**: 정규화값이 `exp(−μ(d − d_best))`이므로 최댓값 1.0은 어떤
        μ에서도 달성 가능하다. 임계값·커리큘럼·성공률의 눈금이 수질과 무관해진다
        (난이도가 같아진다는 뜻은 아니다 — μ가 크면 거리 초과의 벌이 지수적으로
        커지는데 그건 의도된 물리다).
        """
        cfg = self.cfg
        centers = (
            self._vol_origin[env_ids][:, None, None, None, :]   # (n,1,1,1,3)
            + self._voxel_offset[None]                          # (1,Nx,Ny,Nz,3)
        )
        # 볼륨은 `_voxelize_gt_mesh()`가 물체 bbox 중심에 맞춰 놓으므로
        # 물체중심 = origin + (vol_dim·voxel)/2 이다.
        half = torch.tensor(
            [d * cfg.tsdf.voxel_size / 2.0 for d in cfg.tsdf.vol_dim],
            device=self.device,
        )
        obj_center = self._vol_origin[env_ids] + half            # (n,3)
        r_v = torch.norm(centers - obj_center[:, None, None, None, :], dim=-1)

        d_best = (cfg.psi_min - r_v).clamp(min=cfg.quality_d_near)
        mu = self._quality_mu[env_ids].view(-1, 1, 1, 1)
        self._q_star[env_ids] = torch.exp(-mu * d_best)

    def _update_quality_diagnostics(self, env_ids: Sequence[int]) -> None:
        """GT surface voxel의 품질 분포를 never/partial/full로 집계한다.

        binary coverage로는 "봤다"로 뭉뚱그려지는 것을 "얼마나 잘 봤나"로 쪼갠다.
        step_1에서 binary 0.857 vs quality 0.483의 괴리를 드러낸 지표가 이것이다.

        step_1은 env마다 도는 Python 루프였는데 여기서는 벡터화했다 — 리셋 경로의
        per-env 루프가 처리량을 5.7배 깎아먹은 전례(`_triangulate`)가 있어서
        같은 실수를 반복하지 않는다.
        """
        idx = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        gt = self._surf_vol[idx]                                    # (n,Nx,Ny,Nz)
        q_soft = (self._quality_vol[idx] / self._q_star[idx]).clamp(0.0, 1.0)
        n_gt = gt.sum(dim=(1, 2, 3)).float().clamp(min=1.0)
        dims = (1, 2, 3)
        self._diag_gt_never[idx] = ((q_soft == 0.0) & gt).sum(dim=dims).float() / n_gt
        self._diag_gt_partial[idx] = (
            (q_soft > 0.0) & (q_soft < 1.0) & gt
        ).sum(dim=dims).float() / n_gt
        self._diag_gt_full[idx] = ((q_soft >= 1.0) & gt).sum(dim=dims).float() / n_gt

    # ── 보상 ─────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        # 수중 렌더를 결정당 **여기서 한 번** 돌린다. 센서의 `update()`는
        # 서브스텝마다 불리므로 거기서 돌리면 같은 프레임을 499번 더 계산한다
        # (2026-09-03 프로파일: 전체의 75.1%). 이 지점은 decimation 루프가
        # 끝난 직후, 즉 그 결정의 렌더가 끝난 뒤라 프레임이 최신이다.
        self._camera.refresh_uw()

        # pose 드리프트는 결정 단위 random walk이므로 융합 **직전**에 한 칸
        # 전진시킨다. `_get_rewards()`는 env-step당 정확히 한 번 불린다.
        self._corruptor.step()
        self._integrate_depth()

        # binary는 quality 모드에서도 항상 계산한다 — 로그/비교용이고, 2026-08-26
        # 9.3시간 baseline과 같은 축에서 볼 수 있어야 한다.
        self.curr_coverage = self._compute_curr_coverage()

        if self.cfg.use_quality_coverage:
            self._compute_quality()                       # _integrate_depth 이후
            self.curr_coverage_q = self._compute_coverage_q()

        # 보상·종료·커리큘럼이 공통으로 쓰는 정규화 coverage
        coverage = self._coverage_for_reward()
        delta_coverage = coverage - self._prev_coverage_norm
        self._prev_coverage_norm = coverage.clone()

        # 이동 비용은 **로봇 몸체**가 얼마나 움직였는지로 잰다.
        #
        # 주의(2026-08-26 버그 수정): 이전 구현은 `self._camera.data.pos_w`를
        # 썼는데, 로봇이 PhysX 외력으로 움직이는 구조에서는 이 값이 초기 스폰
        # pose에 고정되므로(= `_build_cam_pose()` 주석 참조) `dist_moved`가 항상
        # 0이 되어 k_x 항이 완전히 무력했다. `root_pos_w`는 articulation의 물리
        # 상태라 이런 문제가 없다.
        #
        # step_1은 `camera.data.pos_w`를 썼지만 그쪽 카메라 오프셋이 (0,0,0)이라
        # 사실상 rig(몸체) 위치와 동일했다 — 로봇 위치를 쓰는 게 step_1의 충실한
        # 대응이며, 의미상으로도 이동 비용은 몸체 이동량이 맞다(카메라 오프셋
        # 0.16m의 회전 성분까지 비용에 넣을 이유가 없음).
        # k_x=0.0(기본)이면 보상에는 기여하지 않지만, 정책이 실제로 얼마나
        # 움직이는지 보는 **진단 지표**로 계속 계산해 노출한다 — step_1의 최종
        # 정책이 "안 움직이는" 상태로 굳었던 전례(analysis/comparison_table.csv의
        # ep_len std=0.00)가 있어 이 값을 관측할 수 있어야 한다.
        robot_pos = self._robot.data.root_pos_w
        dist_moved = torch.norm(robot_pos - self._prev_robot_pos, dim=-1)
        self._prev_robot_pos = robot_pos.clone()
        self.last_dist_moved = dist_moved

        cfg = self.cfg
        goal_reached = (coverage >= self._current_coverage_terminal()).float()
        success_reward = goal_reached * cfg.coverage_bonus

        reward_coverage = cfg.k_c * delta_coverage
        reward_penalty = (cfg.k_x * dist_moved) + cfg.c_step
        stall_mask = (delta_coverage < cfg.stall_thr).float()
        reward_stall = cfg.k_still * stall_mask

        # 보상 분해도 진단용으로 보관 — step_1에서 "성공 보너스가 전체 보상의
        # 91%"라는 불균형이 로그 분해로만 드러났던 선례(step_1 CLAUDE.md §9)를
        # 따라, step_3에서도 같은 분해를 볼 수 있게 한다.
        self.last_reward_terms = {
            "coverage": reward_coverage,
            "penalty": reward_penalty,
            "stall": reward_stall,
            "success": success_reward,
        }

        return reward_coverage - reward_penalty - reward_stall + success_reward

    # ── 종료 ─────────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # `_get_rewards()`가 먼저 돌아 coverage/quality를 갱신해 둔 상태다
        # (DirectRLEnv.step: _get_rewards → _get_dones → _reset_idx).
        coverage_reached = self._coverage_for_reward() >= self._current_coverage_terminal()
        pos_env = self._robot.data.root_pos_w - self.scene.env_origins
        out_of_bounds = (pos_env.abs() > self.cfg.max_bound).any(dim=-1)
        # 커리큘럼은 **커버리지 달성만** 성공으로 봐야 한다 — `terminated`에는
        # 경계 이탈이 섞여 있어서 그대로 쓰면 밖으로 나가버린 에피소드가 성공으로
        # 집계되고, 난이도가 잘못 올라간다.
        self._last_coverage_reached = coverage_reached
        terminated = coverage_reached | out_of_bounds
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── 리셋 ─────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        # GT surface 품질 분포 진단은 **_voxelize_gt_mesh()가 _surf_vol을 새
        # 에피소드 것으로 갈아끼우기 전에** 해야 한다. step_1은 이 순서를 틀려
        # never=0.791인데 coverage_q=0.464인 모순된 로그를 냈다(§11 버그 이력).
        if self.cfg.use_quality_coverage:
            self._update_quality_diagnostics(env_ids)

        # 커리큘럼도 super() 이전에 — `_last_coverage_reached`는 방금 끝난
        # 에피소드의 결과이고, 초기 reset(아직 에피소드가 없음)에서는 건너뛴다.
        if (
            self.cfg.curriculum_enabled
            and self.cfg.curriculum_adaptive
            and self.common_step_counter > 0
        ):
            self._update_curriculum(
                torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            )

        super()._reset_idx(env_ids)

        cfg = self.cfg
        n = len(env_ids)
        env_ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if cfg.eval_mode:
            self._sph_theta[env_ids_t] = cfg.eval_theta
            self._sph_phi[env_ids_t] = cfg.eval_phi
            self._sph_psi[env_ids_t] = cfg.eval_psi
        else:
            self._randomize_rock_pose(env_ids)
            self._sph_theta[env_ids_t] = torch.rand(n, device=self.device) * 2.0 * math.pi
            self._sph_phi[env_ids_t] = torch.rand(n, device=self.device) * (cfg.phi_max - cfg.phi_min) + cfg.phi_min
            self._sph_psi[env_ids_t] = torch.rand(n, device=self.device) * (cfg.psi_max - cfg.psi_min) + cfg.psi_min

        # 차량 조명 켜기 (step_1과 동일: 매 리셋마다 light_level_init로 설정).
        # scene_cfg의 SphereLight는 intensity=0.0으로 선언되므로 이 호출이 없으면
        # 조명이 꺼진 채로 돌아간다 — 초기 구현에서 빠져 있던 부분.
        self._light_level[env_ids_t] = cfg.light_level_init
        self._update_light_intensity(self._light_level)

        offset = torch.stack([
            self._sph_psi[env_ids_t] * torch.sin(self._sph_phi[env_ids_t]) * torch.cos(self._sph_theta[env_ids_t]),
            self._sph_psi[env_ids_t] * torch.sin(self._sph_phi[env_ids_t]) * torch.sin(self._sph_theta[env_ids_t]),
            self._sph_psi[env_ids_t] * torch.cos(self._sph_phi[env_ids_t]),
        ], dim=-1)
        spawn_pos = self.rock_pos[env_ids_t] + offset
        spawn_quat = self._look_at_quat(spawn_pos, self.rock_pos[env_ids_t])

        default_state = self._robot.data.default_root_state[env_ids].clone()
        default_state[:, :3] = spawn_pos
        default_state[:, 3:7] = spawn_quat
        self._robot.write_root_pose_to_sim(default_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_state[:, 7:], env_ids)

        # NBV 목표를 스폰 자세로 초기화 — 첫 액션이 오기 전까지 "목표 미부여
        # 시 정지 호버" 상태를 유지한다 (Stage 1 검증 순서 1번째 항목).
        # 순서 주의: reset()이 목표를 원점/항등쿼터니언으로 되돌리므로 반드시
        # reset() 먼저, set_target() 나중에 호출해야 한다(반대로 하면 목표가
        # 스폰 pose가 아니라 월드 원점이 됨).
        self._guidance.reset(env_ids_t)
        self._guidance.set_target(env_ids_t, spawn_pos, spawn_quat)
        self._dp.reset(env_ids_t)
        self._thruster.reset(env_ids_t)
        self._hydro.reset(env_ids_t)

        # 리셋 직전 coverage를 캐싱한다 — `DirectRLEnv.step()`이 `_get_dones()` →
        # `_reset_idx()` → `_get_observations()` 순으로 돌기 때문에, 학습 스크립트가
        # step() 반환 후 `curr_coverage`를 읽으면 **이미 0으로 리셋된 값**을 본다.
        # step_1도 같은 이유로 `_terminal_coverage_q`를 따로 뒀다(그 부분을 초기
        # 이식에서 빠뜨려 학습 로그에 cov=0.000이 찍히던 것을 2026-08-26 수정).
        self.terminal_coverage[env_ids_t] = self.curr_coverage[env_ids_t]
        self.terminal_coverage_q[env_ids_t] = self._coverage_for_reward()[env_ids_t]
        self.curr_coverage[env_ids_t] = 0.0
        self._quality_vol[env_ids_t] = 0.0
        self.curr_coverage_q[env_ids_t] = 0.0
        self._prev_coverage_norm[env_ids_t] = 0.0
        self._prev_robot_pos[env_ids_t] = spawn_pos
        self._actions[env_ids_t] = 0.0

        # ── 도메인 랜덤화 (step_2_BROV/envs/vel_env.py와 동일 1단계 범위) ──
        if cfg.dr_enable_mass:
            mass_result = randomize_articulation_mass(
                self._robot, env_ids_t, relative_range=cfg.dr_mass_scale_range,
            )
        else:
            mass_result = randomize_articulation_mass(
                self._robot, env_ids_t, relative_range=(1.0, 1.0),
            )
        self._mass_scale[env_ids_t] = mass_result.scale.to(self.device)

        vol_lo, vol_hi = cfg.dr_volume_range
        volume = math_utils.sample_uniform(vol_lo, vol_hi, (n,), self.device)
        cob_offset = _sample_from_sphere(n, cfg.dr_cob_radius, self.device)
        am_lo, am_hi = cfg.dr_added_mass_rot_range
        am_scale = math_utils.sample_uniform(am_lo, am_hi, (n,), self.device)
        added_mass_rot = self._nominal_added_mass_rot.unsqueeze(0) * am_scale.unsqueeze(-1)
        self._hydro.randomize(env_ids_t, volume=volume, cob_offset=cob_offset, added_mass_rot=added_mass_rot)

        # ── 카메라 프레임 채우기(초기 관측 버퍼) + GT 메쉬 복셀화 ──
        sim = sim_utils.SimulationContext.instance()
        for _ in range(5):
            sim.render()

        # 최초 `env.reset()`에는 앞선 `_get_rewards()`가 없으므로 여기서 보장한다.
        self._camera.refresh_uw()
        raw_rgb = self._camera.data.output["uw_rgb"][env_ids_t, :, :, :3]
        current_obs = torch.mean(raw_rgb.float(), dim=-1) / 255.0
        current_depth = self._camera.data.output["distance_to_camera"][env_ids_t].squeeze(-1)

        current_obs = F.interpolate(
            current_obs.unsqueeze(1), size=(cfg.visual.h, cfg.visual.w),
            mode="bilinear", align_corners=False,
        ).squeeze(1)
        current_depth = F.interpolate(
            current_depth.unsqueeze(1), size=(cfg.visual.h, cfg.visual.w),
            mode="nearest",
        ).squeeze(1)

        self._image_buffer[env_ids_t] = current_obs.unsqueeze(1).expand(-1, cfg.visual.num_seq_actor, -1, -1)
        self._depth_buffer[env_ids_t] = current_depth.unsqueeze(1).expand(-1, cfg.visual.num_seq_critic, -1, -1)

        if cfg.jerlov_dr_enabled:
            self._randomize_water_params(env_ids_t)

        # μ 동기화는 수질 랜덤화 **이후**여야 새 값이 반영된다. DR 여부와
        # 무관하게 항상 호출한다 — step_1은 DR이 꺼지면 μ가 초기값 0.1에 머물러
        # 정규화 상수와 어긋났다(§13 알려진 불일치).
        if cfg.use_quality_coverage:
            self._sync_quality_water(env_ids_t)

        self._voxelize_gt_mesh(env_ids)

        # 오염 상태(스케일·μ̂ 배율·드리프트)를 에피소드 단위로 새로 뽑는다.
        self._corruptor.reset(env_ids_t)
        if cfg.corruption.enabled:
            # 로봇이 **믿는** 감쇠계수. 실제 렌더 감쇠는 그대로이므로
            # "수질을 잘못 알고 있는 상태"가 된다.
            self._quality_mu[env_ids_t] = (
                self._quality_mu[env_ids_t] * self._corruptor.mu_factor[env_ids_t]
            )

        # q*는 μ(수질)와 `_vol_origin`(물체 위치)에 모두 의존하므로 **복셀화
        # 이후**에 갱신해야 한다. 순서가 바뀌면 직전 에피소드의 물체 중심으로
        # 정규화돼 보상이 조용히 틀어진다. μ̂ 교란도 이 앞에 와야 반영된다.
        if cfg.use_quality_coverage:
            self._update_q_star(env_ids_t)

    def _randomize_water_params(self, env_ids) -> None:
        import numpy as np

        # `scene_cfg.py`가 같은 표를 import하며 경로를 이미 넣지만, 이 함수만
        # 따로 호출되는 경우(진단 스크립트 등)에도 동작하도록 여기서도 보장한다.
        # 이전에는 경로 없이 import해서, `jerlov_dr_enabled=True`로 켜는 순간
        # ImportError로 죽는 잠복 결함이었다(기본값이 False라 드러나지 않았다).
        from envs.scene_cfg import JERLOV_PRESETS

        for eid in env_ids:
            eid_int = int(eid)
            chosen = str(np.random.choice(self.cfg.jerlov_types))
            self._camera.set_water_params([eid_int], **JERLOV_PRESETS[chosen])
