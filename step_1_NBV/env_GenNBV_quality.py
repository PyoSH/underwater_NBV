"""
env_GenNBV_quality.py — Quality-Aware Voxel for Underwater NBV
==============================================================

OceanEnvGenNBV 서브클래스. ch2를 binary "occupied" → continuous "quality"로 교체.

  ch0: unknown  (weight == 0)
  ch1: free     (weight > 0, tsdf > 0)
  ch2: quality  (continuous, normalized by Q_sat)

quality 정의 (Beer-Lambert, OceanSim UWrenderer 단방향 감쇠):
  q(voxel) = exp(-μ * d)
    μ : 수중 감쇠계수 (에피소드별 camera._atten_coeff 동기화)
    d : cam_pos ↔ voxel 중심 거리
    단방향: OceanSim renderer가 exp(-μd) post-process 1회만 적용

누적 방식: sum (Fisher 정보 합산, NBUV CVPR 2016 Eq.20 대응)
  Q_vol(v) += q(v)  when surface_mask

포화 임계값: Q_sat = exp(-μ×psi_min) ≈ 0.80
  → psi_min(1.0m)에서 단일 방문 시 포화
  soft_coverage(v) = clip(Q_vol(v) / Q_sat, 0, 1)
  coverage_q       = mean over GT surface voxels of soft_coverage(v)

보상: k_c_q * Δcoverage_q  (k_c=0으로 설정, binary reward 비활성)
종료: coverage_q >= coverage_terminal
"""
from __future__ import annotations
import math
import torch
from env_GenNBV import OceanEnvGenNBV
from envCfg import OceanEnvCfg


class OceanEnvGenNBVQuality(OceanEnvGenNBV):
    """Beer-Lambert quality-aware voxel + quality-weighted coverage reward."""

    cfg: OceanEnvCfg

    # ── 초기화 ────────────────────────────────────────────────────────────────

    def __init__(self, cfg: OceanEnvCfg, render_mode: str | None = None):
        super().__init__(cfg, render_mode)

        Nx, Ny, Nz = cfg.tsdf.vol_dim
        self._quality_vol = torch.zeros(
            self.num_envs, Nx, Ny, Nz, device=self.device
        )
        self._quality_Q_sat = cfg.q_sat  # exp(-μ×psi_min) ≈ 0.805: psi_min 단일 방문 포화 기준

        # μ 초기값: atten_coeff_max 채널 평균 (리셋 시 실제 값으로 동기화)
        self._quality_mu = float(
            sum(cfg.water_dr.atten_coeff_max) / len(cfg.water_dr.atten_coeff_max)
        )

        # quality-weighted coverage 추적
        self.curr_coverage_q = torch.zeros(self.num_envs, device=self.device)
        self._prev_coverage_q = torch.zeros(self.num_envs, device=self.device)
        self._terminal_coverage_q = torch.zeros(self.num_envs, device=self.device)

        # 진단: GT surface voxel quality 분포 (에피소드 종료 시 기록)
        self._diag_gt_never   = torch.zeros(self.num_envs, device=self.device)  # quality=0 비율
        self._diag_gt_partial = torch.zeros(self.num_envs, device=self.device)  # 0<q<1 비율
        self._diag_gt_full    = torch.zeros(self.num_envs, device=self.device)  # q≥1 비율

        self._precompute_voxel_offsets()

    def _precompute_voxel_offsets(self) -> None:
        Nx, Ny, Nz = self.cfg.tsdf.vol_dim
        vs = self.cfg.tsdf.voxel_size
        gx, gy, gz = torch.meshgrid(
            torch.arange(Nx, device=self.device),
            torch.arange(Ny, device=self.device),
            torch.arange(Nz, device=self.device),
            indexing="ij",
        )
        self._voxel_offset = torch.stack([gx, gy, gz], dim=-1).float() * vs

    # ── Quality 계산 ──────────────────────────────────────────────────────────

    def _compute_quality(self) -> None:
        """
        현재 관측된 표면 voxel의 quality를 Beer-Lambert로 업데이트 (max 갱신).
        _integrate_depth() 완료 후 호출되어야 함 (TSDF 최신 상태 전제).
        """
        centers = (
            self._vol_origin[:, None, None, None, :]
            + self._voxel_offset[None]
        )
        cam = self.cam_pos[:, None, None, None, :]
        dist = torch.norm(centers - cam, dim=-1)

        # 단방향 Beer-Lambert (OceanSim UWrenderer: exp(-μd) post-process 단회 적용)
        quality_new = torch.exp(-self._quality_mu * dist)

        # 관측된 voxel이면 갱신 (TSDF 분류 무관 — 해석 B: 관측 품질 기준)
        surface_mask = self._weight_vol > 0

        # 최근접 접근 품질 기록 (max): 반복 방문 시 누적 없음
        self._quality_vol = torch.maximum(
            self._quality_vol,
            quality_new * surface_mask.float()
        )

    def _compute_coverage_q(self) -> torch.Tensor:
        """quality-weighted coverage 계산. _compute_quality() 이후 호출."""
        count = (self._quality_vol * self._surf_vol.float()).sum(dim=(1, 2, 3))
        return count / self._total_surf_voxels

    # ── 보상 override ─────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """
        k_c_q * Δcoverage_q 기반 보상.
        _integrate_depth() → _compute_quality() → coverage_q 순서 보장.
        """
        self._integrate_depth()

        # quality 갱신 (TSDF 최신 상태)
        self._compute_quality()

        # binary coverage (로그 전용, 보상에 사용 안 함)
        self.curr_coverage = self._compute_curr_coverage()

        # quality-weighted coverage
        self.curr_coverage_q = self._compute_coverage_q()
        delta_coverage_q = self.curr_coverage_q - self._prev_coverage_q
        self._prev_coverage_q = self.curr_coverage_q.clone()

        # binary coverage prev 동기화 (부모 클래스 호환)
        delta_coverage = self.curr_coverage - self._prev_coverage
        self._prev_coverage = self.curr_coverage.clone()

        cfg = self.cfg

        goal_reached   = ((self.curr_coverage_q / self._quality_Q_sat) >= cfg.coverage_terminal).float()
        success_reward = goal_reached * cfg.coverage_bonus

        reward_coverage = cfg.k_c_q * delta_coverage_q
        dist_moved      = torch.norm(self.cam_pos - self._prev_cam_pos, dim=-1)
        reward_penalty  = cfg.k_x * dist_moved + cfg.c_step
        self._prev_cam_pos = self.cam_pos.clone()

        # stall: quality-weighted coverage 기준
        stall_mask   = (delta_coverage_q < cfg.stall_thr).float()
        reward_stall = cfg.k_still * stall_mask

        if cfg.use_visit_map and cfg.k_explore > 0.0:
            reward_explore = cfg.k_explore * self._last_visit_new
        else:
            reward_explore = torch.zeros(self.num_envs, device=self.device)

        # 부모 로그 속성 호환 유지
        self._last_rew_coverage   = reward_coverage
        self._last_rew_penalty    = reward_penalty
        self._last_rew_stall      = reward_stall
        self._last_rew_explore    = reward_explore
        self._last_success_reward = success_reward
        self._last_rew_contrast   = torch.zeros(self.num_envs, device=self.device)

        # delta_coverage는 binary 기준 (stall 로그용)
        self._last_delta_coverage_q = delta_coverage_q
        self._last_delta_coverage   = delta_coverage

        return reward_coverage - reward_penalty - reward_stall + success_reward + reward_explore

    # ── 종료 조건 override ────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = (self.curr_coverage_q / self._quality_Q_sat) >= self.cfg.coverage_terminal
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ── Voxel 생성 override ───────────────────────────────────────────────────

    def _get_vox_actor(self) -> torch.Tensor:
        """
        ch0: unknown  (weight == 0)
        ch1: observed as free space by TSDF  (weight > 0, tsdf > 0)
        ch2: observation quality  (weight > 0, tsdf 무관 — 해석 B)
             ch1=1이면서 ch2>0인 voxel 존재 가능 (free space이지만 가까이서 관측됨)
        """
        observed = self._weight_vol > 0
        return torch.stack([
            (~observed).float(),
            (observed & (self._tsdf_vol > 0)).float(),
            (self._quality_vol / self._quality_Q_sat).clamp(0.0, 1.0),
        ], dim=1)

    # ── 관측 override ────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        # _compute_quality()는 _get_rewards() → _compute_coverage_q() 내에서 이미 호출됨
        # super()._get_observations() → _get_vox_actor() (quality 버전)
        obs = super()._get_observations()
        obs["coverage_q"] = self.curr_coverage_q.clone()
        return obs

    # ── 리셋 override ────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids) -> None:
        # 진단: super() 호출 전에 현재 episode의 surf_vol로 quality 분포 계산
        # (super()._reset_idx가 _surf_vol을 새 episode 것으로 교체하기 때문)
        for eid in env_ids:
            gt_mask = self._surf_vol[eid]                              # [Nx,Ny,Nz] bool
            if gt_mask.any():
                q_soft = (self._quality_vol[eid] / self._quality_Q_sat).clamp(0.0, 1.0)
                gt_q   = q_soft[gt_mask]                               # GT voxels만
                self._diag_gt_never[eid]   = (gt_q == 0.0).float().mean()
                self._diag_gt_partial[eid] = ((gt_q > 0.0) & (gt_q < 1.0)).float().mean()
                self._diag_gt_full[eid]    = (gt_q >= 1.0).float().mean()

        self._terminal_coverage_q[env_ids] = self.curr_coverage_q[env_ids]

        super()._reset_idx(env_ids)

        self._quality_vol[env_ids]         = 0.0
        self.curr_coverage_q[env_ids]      = 0.0
        self._prev_coverage_q[env_ids]     = 0.0

        # μ 동기화: _randomize_water_params() 이후 실제 값 반영
        if self.cfg.water_dr_enabled:
            c = self._camera._atten_coeff   # wp.vec3f
            self._quality_mu = (c.x + c.y + c.z) / 3.0
            self._quality_Q_sat = math.exp(-self._quality_mu * self.cfg.psi_min)
