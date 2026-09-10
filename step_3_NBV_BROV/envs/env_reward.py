"""
step_1_NBV/env/env_reward.py를 무수정 이식 — TSDF 적분/coverage 계산은 카메라가
자유 부유(sensor_rig)든 로봇에 고정 부착이든 동일 로직(`self._camera.data.*`,
`self._build_cam_pose()`만 사용, 로봇 종류 무관). `_build_cam_pose()` 자체의
변경(`envs/env_utils.py` 참조)만으로 충분해 여기는 손댈 곳이 없다.
"""

from __future__ import annotations
import math
import torch

from . import tsdf_fusion


class EnvRewardMixin:
    # ②a 이중 스트림 ────────────────────────────────────────────────────
    # 믿음(belief) 스트림: 오염된 depth + 믿는 pose + μ̂ → 정책 관측·보상·종료.
    # 진실(truth)  스트림: 깨끗한 depth + 실제 pose + 실제 μ → **채점 전용**.
    # 오염이 꺼져 있으면 진실 스트림은 계산하지 않고 채점도 믿음 스트림을 쓴다.
    # 채점이 믿음 스트림을 자기 자신과 비교하면 잡음이 관측 표시 voxel을 늘려
    # coverage가 **올라가는** 누수가 생긴다(2026-09-09 ②a 1차 스윕: noise0.20
    # 에서 +26%). 그래서 "로봇이 무엇을 봤다고 믿는가"와 "실제로 무엇이 보였나"를
    # 분리한다.
    def _integrate_depth(self) -> None:
        """믿음 스트림 융합 — 로봇이 **믿는** depth·pose."""
        depth_img = self._corruptor.depth(
            self._camera.data.output["distance_to_camera"]
        )
        self._tsdf_vol, self._weight_vol = self._fuse_depth(
            depth_img, self._build_cam_pose(), self._tsdf_vol, self._weight_vol)

    def _integrate_depth_true(self) -> None:
        """진실 스트림 융합 — 렌더된 depth 그대로 + 실제 pose. 오염기를 거치지 않는다."""
        depth_img = self._camera.data.output["distance_to_camera"]
        self._tsdf_vol_true, self._weight_vol_true = self._fuse_depth(
            depth_img, self._build_cam_pose(corrupt=False),
            self._tsdf_vol_true, self._weight_vol_true)

    def _fuse_depth(self, depth_img: torch.Tensor, cam_pose: torch.Tensor,
                    tsdf_vol: torch.Tensor, weight_vol: torch.Tensor):
        """TSDF 융합 — 구현은 `envs/tsdf_fusion.py` 에 있다.

        2026-09-10 분리: 배포 ROS 노드가 **같은 코드**를 import 하게 하기 위함이다
        (DEPLOY_3WEEK_PLAN §6: "복사하지 말고 import"). 여기서는 self 에서 인자를
        모아 넘기기만 한다. 수치 동일성은 `deploy/test_tsdf_fusion.py` 가 원본
        구현을 참조로 들고 비교해 고정한다.
        """
        cfg = self.cfg.tsdf
        k = self._camera.data.intrinsic_matrices
        if not hasattr(self, "_vox_local"):
            self._vox_local = tsdf_fusion.build_voxel_grid(
                cfg.vol_dim, cfg.voxel_size, device=self.device)

        # 유클리드 -> z-depth (2026-09-10 수정).
        # 카메라가 내는 `distance_to_camera` 는 **광학중심까지의 슬랜트 거리**인데
        # 융합식 `sdf = depth - vox_z` 는 **광축 방향 거리**를 요구한다. 그대로 넣으면
        # 광축에서 theta 벗어난 화소가 1/cos(theta) 배 과대 측정된다 — psi=1.7 m 에서
        # 수평 가장자리 15.5 cm(1.55 voxel), 모서리 23.7 cm(2.37 voxel)로 trunc_margin
        # 0.10 m 를 넘는다. 물체 반폭 0.7 m 도 22.4 도를 차지해 1.38 voxel 이 틀어진다.
        #
        # annotator 를 추가하지 않고 화소별로 변환한다 — 렌더 비용 0.
        # `distance_to_camera` 자체는 **UW 렌더에서 계속 옳다**(감쇠의 광로 길이는
        # 유클리드다). 즉 키를 바꾸는 게 아니라 융합 직전에만 변환한다.
        if not hasattr(self, "_z_depth_scale"):
            hw = depth_img.shape[1], depth_img.shape[2]
            self._z_depth_scale = tsdf_fusion.z_depth_scale(hw[0], hw[1], k)
        depth_z = tsdf_fusion.euclidean_to_z_depth(depth_img, k, self._z_depth_scale)

        return tsdf_fusion.fuse_depth(
            depth_z, cam_pose, tsdf_vol, weight_vol,
            intrinsics=k, vol_origin=self._vol_origin,
            voxel_size=cfg.voxel_size, trunc_margin=cfg.trunc_margin,
            vol_dim=cfg.vol_dim, vox_local=self._vox_local)

    def _compute_patch_contrast(self, img: torch.Tensor) -> torch.Tensor:
        patches = img.unfold(1, 14, 14).unfold(2, 14, 14)
        patch_std = torch.std(patches, dim=(-1, -2))

        return torch.mean(patch_std, dim=(1, 2))

    def _compute_curr_coverage(self, weight_vol: torch.Tensor | None = None) -> torch.Tensor:
        w = self._weight_vol if weight_vol is None else weight_vol
        observed = (w > 0) & self._surf_vol   # GT surface만 카운트
        count = observed.sum(dim=(1, 2, 3)).float()
        return (count / self._total_surf_voxels).clamp(0.0, 1.0)

    # ── Quality-weighted coverage (step_1 env_GenNBV_quality.py 이식) ─────────

    def _compute_quality(self) -> None:
        """관측된 voxel의 품질을 Beer-Lambert 감쇠로 갱신한다.

        `_integrate_depth()` 이후에 호출해야 한다(TSDF/weight가 최신이어야 함).

        `surface_mask = weight > 0` — TSDF 분류(`tsdf <= 0`)를 **조건에 넣지
        않는다**. step_1에서 이 조건을 넣었더니 GT surface voxel의 37%가
        "관측됐지만 TSDF는 free space로 분류"돼 품질 누적이 차단됐고,
        binary 0.857 vs quality 0.483이라는 괴리가 생겼다(step_1 CLAUDE.md §10,
        "해석 B"로 수정 완료). 여기서 재는 것은 재구성 확정도가 아니라
        **관측 품질**이므로 weight>0이면 누적하는 것이 맞다.

        누적은 합이 아니라 **max**다 — 같은 voxel을 반복 방문해도 품질이 무한히
        쌓이지 않고 "가장 가까이서 본 순간"만 남는다(step_1 2026-05-26 변경).
        (nbuv 모델은 합 누적 — `_accumulate_quality` 참조.)
        """
        self._quality_vol, self._last_info_gain = self._accumulate_quality(
            cam=self._camera_position_w(), mu=self._quality_mu,
            tsdf_vol=self._tsdf_vol, weight_vol=self._weight_vol,
            quality_vol=self._quality_vol, q_star=self._q_star)

    def _compute_quality_true(self) -> None:
        """②a 진실 스트림 품질 — 실제 pose·실제 μ·깨끗한 TSDF. 보상(정보 이득)에는 기여하지 않는다."""
        self._quality_vol_true, _ = self._accumulate_quality(
            cam=self._camera_position_w(corrupt=False), mu=self._quality_mu_true,
            tsdf_vol=self._tsdf_vol_true, weight_vol=self._weight_vol_true,
            quality_vol=self._quality_vol_true, q_star=self._q_star_true)

    def _accumulate_quality(self, cam: torch.Tensor, mu: torch.Tensor,
                            tsdf_vol: torch.Tensor, weight_vol: torch.Tensor,
                            quality_vol: torch.Tensor, q_star: torch.Tensor):
        """한 스트림의 품질 누적. 반환 (새 quality_vol, 정보 이득(E,)) — 입력은 바꾸지 않는다."""
        centers = (
            self._vol_origin[:, None, None, None, :]      # (E,1,1,1,3)
            + self._voxel_offset[None]                    # (1,Nx,Ny,Nz,3)
        )
        cam = cam[:, None, None, None, :]
        dist = torch.norm(centers - cam, dim=-1)          # (E,Nx,Ny,Nz)

        mu = mu.view(-1, 1, 1, 1)
        quality_new = torch.exp(-mu * dist)

        observed = weight_vol > 0
        E = self.num_envs
        info_gain = torch.zeros(E, device=self.device)

        if self.cfg.quality_model in ("pixel", "nbuv"):
            view = cam - centers                       # voxel -> 카메라
            view = view / view.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            # |cos| — 면의 어느 쪽인지는 입사각과 무관하다(env_utils 법선 주석).
            if self.cfg.quality_normal_source == "tsdf":
                # 배포에서 계산 가능한 법선. 이번 스텝의 융합이 끝난 TSDF에서
                # 한쪽차분으로 구한다(`tools/measure_tsdf_normals.py` 실측 근거).
                n_est, n_valid = self._tsdf_normals(tsdf_vol, weight_vol)
                cos_inc = torch.where(
                    n_valid,
                    (n_est * view).sum(dim=-1).abs(),
                    torch.full_like(dist, self.cfg.nbuv_cos_fallback),
                )
            else:
                cos_inc = (self._surf_normal * view).sum(dim=-1).abs()

        if self.cfg.quality_model == "pixel":
            # (브랜치 1차안, 비교용으로 유지) 해상도 항을 SNR에 곱한 형태.
            # 무한 상승 항이라 psi_min 고착·눈금 붕괴를 만든다 — "nbuv" 참조.
            quality_new = quality_new * cos_inc / dist.clamp(min=1e-3) ** 2

        if self.cfg.quality_model == "nbuv":
            cfg = self.cfg
            d = dist.clamp(min=1e-3)
            # ── 해상도 요구 (NBUV Sec.6) ──
            # R = f^2 cos / d^2 [px/m^2]: voxel 표면이 화면에서 차지하는 픽셀 밀도.
            f_px = self._camera.data.intrinsic_matrices[:, 0, 0].view(-1, 1, 1, 1)
            R = f_px ** 2 * cos_inc / d ** 2
            R_min = (cfg.nbuv_px_per_voxel_edge / cfg.tsdf.voxel_size) ** 2
            gamma = R / R_min
            # gamma<1: 요구 해상도 미달 -> exp 벌 (Eq.29). gamma>1: 여분 픽셀의
            # 평균화 이득, 상한 (Eq.30). 임계 밖에서만 거리 압력이 걸린다.
            res_factor = (torch.exp(-cfg.nbuv_res_penalty_eta * (1.0 / gamma.clamp(min=1e-3) - 1.0).clamp(min=0.0))
                          * gamma.clamp(max=cfg.nbuv_res_gain_max).clamp(min=1e-6))
            # ── SNR 품질 (NBUV Eq.2,4,16; co-located 조명이라 l_LS = l_SC = d) ──
            # E: 조명 역제곱 + 왕복 감쇠 + Lambertian cos. B: 시선 backscatter.
            E_irr = cfg.nbuv_light_power * torch.exp(-2.0 * mu * d) / d ** 2 * cos_inc
            binf = self._quality_binf.view(-1, 1, 1, 1)
            bcoef = self._quality_bcoef.view(-1, 1, 1, 1)
            B = binf * (1.0 - torch.exp(-bcoef * d))
            q_snr = E_irr ** 2 / (cfg.nbuv_albedo * E_irr + B + cfg.nbuv_noise_floor)
            quality_new = q_snr * res_factor

            # ── 정보 이득 (Eq.25): 1/2 ln(1 + Q_t / Q_acc) — 재방문은 체감 ──
            # Q_acc에 사전 정밀도 Q_prior를 더해 첫 관측의 이득을 유한하게 한다.
            # 정규화: 요구조건을 정확히 만족하는 첫 관측(Q=Q_target)의 이득 = 1.
            q_new_obs = quality_new * observed.float()
            q_prior = cfg.nbuv_prior_ratio * q_star
            gain = 0.5 * torch.log1p(q_new_obs / (quality_vol + q_prior))
            gain_norm = 0.5 * math.log1p(1.0 / cfg.nbuv_prior_ratio)
            surf = self._surf_vol.float()
            info_gain = (gain * surf).sum(dim=(1, 2, 3)) / (
                self._total_surf_voxels * gain_norm)
            # 누적은 **합** (Eq.21) — Fisher 정보는 더해진다.
            return quality_vol + q_new_obs, info_gain

        return torch.maximum(quality_vol, quality_new * observed.float()), info_gain

    def _tsdf_normals(self, tsdf_vol: torch.Tensor | None = None,
                      weight_vol: torch.Tensor | None = None):
        """∇TSDF 법선 (한쪽차분 허용) — 관측 이웃이 있는 축만 쓴다.

        6-이웃 중앙차분은 voxel 10 cm(물체 6~11칸)에서 관측 표면의 5~21%만
        가용해 관측 채널에 쓸 수 없다. 축마다 관측된 이웃이 한쪽이라도 있으면
        그쪽 차분을 쓰면 가용이 62~84%로 오르고 정확도는 같다(2026-09-08 실측:
        각오차 중앙 11~25°, 무작위 60°). 반환: (E,Nx,Ny,Nz,3) 단위법선, 유효 mask.
        """
        tsdf = self._tsdf_vol if tsdf_vol is None else tsdf_vol
        obs = (self._weight_vol if weight_vol is None else weight_vol) > 0
        g = torch.zeros(*tsdf.shape, 3, device=tsdf.device)
        valid = obs.clone()
        for ax in range(3):
            c = [slice(None)] * 4; f = [slice(None)] * 4; b = [slice(None)] * 4
            c[ax + 1] = slice(1, -1); f[ax + 1] = slice(2, None); b[ax + 1] = slice(None, -2)
            c, f, b = tuple(c), tuple(f), tuple(b)
            fwd = torch.zeros_like(tsdf); bwd = torch.zeros_like(tsdf)
            of = torch.zeros_like(obs);   ob = torch.zeros_like(obs)
            fwd[c] = tsdf[f] - tsdf[c];   bwd[c] = tsdf[c] - tsdf[b]
            of[c] = obs[f];               ob[c] = obs[b]
            g[..., ax] = torch.where(of & ob, 0.5 * (fwd + bwd), torch.where(of, fwd, bwd))
            valid &= (of | ob)
        n = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return n, valid

    def _compute_coverage_q(self, quality_vol: torch.Tensor | None = None,
                            q_star: torch.Tensor | None = None) -> torch.Tensor:
        """GT surface voxel의 **voxel별 정규화** 품질 평균 — (A), 상한 1.0.

            coverage_q = (1/N) Σ_v  min(Q_vol(v) / q*(v), 1)      v ∈ GT surface

        정규화를 **합산 전에** voxel마다 적용하는 것이 핵심이다. 전역 상수로
        나누고 나중에 clamp하면 1을 넘는 voxel의 초과분이 못 본 voxel을
        상쇄해 버린다(`env.py::_update_q_star()`의 실측 근거 참조).
        clamp를 voxel 단위로 두면 어떤 voxel도 다른 voxel의 결손을 대신
        갚아줄 수 없다.
        """
        qv = self._quality_vol if quality_vol is None else quality_vol
        qs = self._q_star if q_star is None else q_star
        q_norm = (qv / qs).clamp(0.0, 1.0)
        count = (q_norm * self._surf_vol.float()).sum(dim=(1, 2, 3))
        return count / self._total_surf_voxels

    def _coverage_for_reward(self) -> torch.Tensor:
        """보상·종료·커리큘럼이 공통으로 쓰는 coverage (항상 0~1 정규화).

        quality 모드의 `coverage_q`는 (A) 정규화로 이미 0~1이므로 그대로 쓴다.
        `coverage_terminal`은 "달성 가능 상한 대비 비율"이라는 의미를 유지하고,
        binary 기준으로 실측 보정해 둔 k_c/c_step/coverage_bonus도 스케일
        변경 없이 그대로 유효하다.
        """
        if not self.cfg.use_quality_coverage:
            return self.curr_coverage
        return self.curr_coverage_q

    # ── ②a 채점용 (진실 대비) ──────────────────────────────────────────
    def _coverage_for_scoring(self) -> torch.Tensor:
        """평가 채점용 coverage — 오염이 켜져 있으면 진실 스트림, 아니면 보상용과 동일."""
        if not self.cfg.corruption.enabled:
            return self._coverage_for_reward()
        if not self.cfg.use_quality_coverage:
            return self.curr_coverage_true
        return self.curr_coverage_q_true

    def _coverage_bin_for_scoring(self) -> torch.Tensor:
        return self.curr_coverage_true if self.cfg.corruption.enabled else self.curr_coverage

    def _terminal_coverage_for_scoring(self):
        """(cov_q, cov_bin) 종료 시점 값 — 리셋 직전에 캐싱된 것."""
        if not self.cfg.corruption.enabled:
            return self.terminal_coverage_q, self.terminal_coverage
        q = (self.terminal_coverage_q_true if self.cfg.use_quality_coverage
             else self.terminal_coverage_true)
        return q, self.terminal_coverage_true
