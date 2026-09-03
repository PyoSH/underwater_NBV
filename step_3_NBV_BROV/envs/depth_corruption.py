"""②a 파라메트릭 오염 — "depth/pose 오차 얼마까지 견디는가"를 재는 장치.

왜 파라메트릭인가 (센서 무관)
------------------------------
배포 depth를 어떤 모델로 뽑을지(TRIDENT든 다른 것이든)는 아직 확정이 아니고,
바뀔 수도 있다. 그런데 **"정책이 얼마만큼의 오차를 견디는가"는 센서와 무관한
정책의 성질**이다. 그래서 오염을 특정 모델의 출력이 아니라 파라미터로 주입해
허용선을 먼저 정하고(②a), 센서는 나중에 그 선 안에 들어오는지만 시험한다(②b).
이 순서 덕분에 센서가 탈락해도 계획 본체가 무사하다.

무엇을 주입하는가
-----------------
1. **에피소드별 depth 스케일 편향** — 단안 depth의 지배적 오차 형태.
   TRIDENT 실측에서 정렬 전 AbsRel 0.257이 장면별 스케일 보정만으로 0.130까지
   떨어졌다(= 오차의 2/3가 스케일). 장면(에피소드) 안에서는 상수이므로
   프레임마다 새로 뽑지 않고 리셋 때 한 번 뽑는다 — 이 상관 구조가 중요하다.
   프레임마다 독립이면 평균으로 상쇄돼 실제보다 훨씬 관대한 시험이 된다.
2. **거리 비례 잡음** — 화소마다 독립. 멀수록 커지는 실제 특성.
3. **pose 드리프트** — DVL dead-reckoning 오차의 random walk. 위치·yaw 모두.
   TSDF 융합 좌표가 틀어지면 voxel이 엉뚱한 자리에 기입되므로, depth 오차와는
   질적으로 다른 열화를 만든다(재구성이 흐려지는 게 아니라 **어긋난다**).
4. **μ̂ 교란** — 수질 미지. 로봇이 믿는 감쇠계수가 실제와 다른 상황.

무엇을 오염시키지 **않는가**
---------------------------
GT surface voxel(`_surf_vol`)과 그 개수는 건드리지 않는다. 그것은 채점 기준이지
로봇의 관측이 아니다. 오염되는 것은 **로봇이 무엇을 봤다고 믿는가**뿐이고,
점수는 언제나 진실 대비로 매긴다 — 그래야 열화가 측정된다.

realized AbsRel
---------------
스윕의 가로축은 파라미터가 아니라 **실제로 주입된 AbsRel**이어야 한다. 그래야
②b에서 TRIDENT+앵커의 AbsRel(0.13~0.16 전망)을 같은 축에 얹어 합격/불합격을
바로 읽을 수 있다. 그래서 오염을 적용할 때마다 |d̂−d|/d를 함께 측정해 둔다.
"""

from __future__ import annotations

import torch

try:
    from isaaclab.utils import configclass
except ModuleNotFoundError:      # Isaac 앱 없이 도는 단위 테스트용 대체
    # `isaaclab.utils`는 pxr을 끌어오므로 AppLauncher 없이는 import되지 않는다.
    # 이 모듈의 설정은 전부 불변 스칼라/튜플이라 표준 dataclass로 충분하고,
    # 실제 실행 경로에서는 언제나 진짜 configclass가 쓰인다
    # (`env_cfg.py`가 AppLauncher 이후에 import되기 때문).
    from dataclasses import dataclass as configclass


@configclass
class DepthCorruptionCfg:
    """비활성이 기본. 학습(teacher)은 GT로 돌고, ②a 평가에서만 켠다."""

    enabled: bool = False
    # 에피소드별 스케일: d̂ = exp(N(0, scale_sigma)) · d
    scale_sigma: float = 0.0
    # 화소별 잡음: d̂ += N(0, rel_noise · d)
    rel_noise: float = 0.0
    # pose random walk (결정당 표준편차)
    pos_drift_std: float = 0.0      # [m/decision]
    yaw_drift_std: float = 0.0      # [rad/decision]
    # 로봇이 믿는 감쇠계수 배율 — 리셋마다 U(lo, hi)
    mu_factor_range: tuple = (1.0, 1.0)
    seed: int = 1234


class DepthCorruptor:
    """env별 오염 상태를 들고 있는 객체. 환경이 소유하고 리셋/스텝에서 호출한다."""

    def __init__(self, cfg: DepthCorruptionCfg, num_envs: int, device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        # 환경 난수와 **분리된** 생성기. 공유하면 오염 세기를 바꾸는 것만으로
        # 물체·초기 시점 추첨까지 달라져 조건 비교가 무너진다.
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed)

        self.scale = torch.ones(num_envs, device=device)
        self.mu_factor = torch.ones(num_envs, device=device)
        self.pos_offset = torch.zeros(num_envs, 3, device=device)
        self.yaw_offset = torch.zeros(num_envs, device=device)

        self._absrel_sum = 0.0
        self._absrel_n = 0

    # ── 상태 갱신 ────────────────────────────────────────────────────────
    def reset(self, env_ids: torch.Tensor) -> None:
        """에피소드 시작: 스케일·μ̂를 새로 뽑고 드리프트를 0으로 되돌린다."""
        if not self.cfg.enabled:
            return
        n = len(env_ids)
        if self.cfg.scale_sigma > 0.0:
            z = torch.randn(n, generator=self.gen, device=self.device)
            self.scale[env_ids] = torch.exp(z * self.cfg.scale_sigma)
        else:
            self.scale[env_ids] = 1.0
        lo, hi = self.cfg.mu_factor_range
        u = torch.rand(n, generator=self.gen, device=self.device)
        self.mu_factor[env_ids] = lo + u * (hi - lo)
        self.pos_offset[env_ids] = 0.0
        self.yaw_offset[env_ids] = 0.0

    def step(self) -> None:
        """결정 1회분 드리프트 누적. 융합 **이전**에 호출해야 한다."""
        if not self.cfg.enabled:
            return
        if self.cfg.pos_drift_std > 0.0:
            self.pos_offset += torch.randn(
                self.num_envs, 3, generator=self.gen, device=self.device
            ) * self.cfg.pos_drift_std
        if self.cfg.yaw_drift_std > 0.0:
            self.yaw_offset += torch.randn(
                self.num_envs, generator=self.gen, device=self.device
            ) * self.cfg.yaw_drift_std

    # ── 적용 ─────────────────────────────────────────────────────────────
    def depth(self, d: torch.Tensor) -> torch.Tensor:
        """(E,H,W[,1]) depth를 오염시키고 realized AbsRel을 누적한다."""
        if not self.cfg.enabled:
            return d
        squeeze = d.dim() == 4
        x = d.squeeze(-1) if squeeze else d
        s = self.scale.view(-1, *([1] * (x.dim() - 1)))
        out = x * s
        if self.cfg.rel_noise > 0.0:
            noise = torch.randn(x.shape, generator=self.gen, device=self.device)
            out = out + noise * (self.cfg.rel_noise * x)
        # 음수/0 depth는 융합에서 의미가 없다 — 물리적으로 가능한 하한으로 자른다.
        out = out.clamp(min=1e-3)

        valid = torch.isfinite(x) & (x > 1e-3) & torch.isfinite(out)
        if valid.any():
            self._absrel_sum += ((out - x).abs() / x)[valid].mean().item()
            self._absrel_n += 1
        return out.unsqueeze(-1) if squeeze else out

    def apply_pose(self, cam_pos_w: torch.Tensor, cam_quat_w: torch.Tensor):
        """로봇이 **믿는** 카메라 pose. 융합·품질·스칼라 관측이 공통으로 쓴다."""
        if not self.cfg.enabled:
            return cam_pos_w, cam_quat_w

        pos = cam_pos_w + self.pos_offset
        if self.cfg.yaw_drift_std > 0.0:
            # z축 회전 dq=[c,0,0,s]와의 Hamilton 곱을 직접 전개한다.
            # `isaaclab.utils.math.quat_mul`을 쓰지 않는 이유: 그 패키지가
            # pxr을 끌어와 AppLauncher 없이는 import되지 않아, 단위 테스트
            # (`tools/test_depth_corruption.py`)에서 이 경로만 못 돌게 된다.
            h = self.yaw_offset * 0.5
            c, sn = torch.cos(h), torch.sin(h)
            w, x, y, z = cam_quat_w.unbind(-1)
            quat = torch.stack([c * w - sn * z,
                                c * x - sn * y,
                                c * y + sn * x,
                                c * z + sn * w], dim=-1)
        else:
            quat = cam_quat_w
        return pos, quat

    # ── 보고 ─────────────────────────────────────────────────────────────
    @property
    def realized_absrel(self) -> float:
        """실제로 주입된 |d̂−d|/d 평균 — 스윕 표의 가로축."""
        return self._absrel_sum / self._absrel_n if self._absrel_n else 0.0

    def clear_stats(self) -> None:
        self._absrel_sum = 0.0
        self._absrel_n = 0

    def describe(self) -> str:
        c = self.cfg
        if not c.enabled:
            return "clean"
        return (f"scale σ{c.scale_sigma:.3f} noise {c.rel_noise:.3f} "
                f"pos {c.pos_drift_std*100:.1f}cm/dec yaw "
                f"{c.yaw_drift_std*57.3:.2f}°/dec μ̂ {c.mu_factor_range}")
