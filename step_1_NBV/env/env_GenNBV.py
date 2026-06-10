"""
env_GenNBV.py — OceanEnv 서브클래스 (GenNBV 전용 관측 추가)
=============================================================

OceanEnv를 상속하여 algo_GenNBV.py가 요구하는 두 가지 관측 키를 추가한다.

  "vox_actor"    : (E, 3, Nx, Ny, Nz)  uint8-compatible float
                   ch0 unknown  : _weight_vol == 0
                   ch1 free     : _weight_vol > 0 & _tsdf_vol > 0
                   ch2 occupied : _weight_vol > 0 & _tsdf_vol <= 0

  "img_semantic" : (E, 2, H, W)  — _image_buffer 의 직전 2 프레임
                   논문 M=2 (ablation에서 2 프레임으로 충분)

train_GenNBV.py 는 아래 키만 사용한다:
  obs["vox_actor"]    → Actor / Critic GeometricEncoder 입력
  obs["img_semantic"] → Actor / Critic SemanticEncoder 입력
  obs["extra_info"]   → Actor scalar  (θ,φ,ψ normalized, dim=3)
  obs["critic_scalar"]→ Critic scalar (θ,φ,ψ,coverage,   dim=4)
  obs["policy"] / obs["critic"] 는 사용하지 않음 (PPO 호환용)

호출 순서 보장:
  _get_rewards() 내 _integrate_depth() → TSDF 갱신
  _get_observations() 호출 → vox 구성 시 최신 TSDF 보장
"""

from __future__ import annotations

import torch
from .env import OceanEnv
from .envCfg import OceanEnvCfg


class OceanEnvGenNBV(OceanEnv):
    """GenNBV 전용 관측 키를 추가한 환경."""

    cfg: OceanEnvCfg

    # ── 3-state voxel 생성 ───────────────────────────────────────────────────

    def _get_vox_actor(self) -> torch.Tensor:
        """
        Returns: (E, 3, Nx, Ny, Nz) float32
          ch0 unknown  : weight == 0          (아직 관측 안 됨)
          ch1 free     : weight > 0, tsdf > 0 (카메라 앞 빈 공간)
          ch2 occupied : weight > 0, tsdf ≤ 0 (표면/물체)
        """
        observed = self._weight_vol > 0                         # (E, Nx, Ny, Nz)
        return torch.stack([
            (~observed).float(),                                # ch0: unknown
            (observed & (self._tsdf_vol  > 0)).float(),         # ch1: free
            (observed & (self._tsdf_vol <= 0)).float(),         # ch2: occupied
        ], dim=1)                                               # (E, 3, Nx, Ny, Nz)

    # ── 관측 override ────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        obs = super()._get_observations()

        obs["vox_actor"]    = self._get_vox_actor()
        obs["img_semantic"] = self._image_buffer[:, -2:, :, :].clone()  # (E, 2, H, W)

        # Critic scalar 항상 dim=4: (θ,φ,ψ,coverage)
        # use_visit_map 여부와 무관하게 통일 (extra_info는 항상 dim=3)
        obs["critic_scalar"] = torch.cat(
            [obs["extra_info"], self.curr_coverage.unsqueeze(-1)], dim=-1
        )  # (E, 4)

        return obs
