"""
TorchScript 정책 로드 + 추론 — topside PC용, IsaacLab/rsl_rl 의존성 없음.
`deploy/export_policy.py`가 컨테이너 안에서 만든 policy.pt를 로드한다.
"""

from __future__ import annotations

import torch


class PolicyRunner:
    def __init__(self, jit_path: str, device: str = "cpu"):
        self.device = device
        self._model = torch.jit.load(jit_path, map_location=device)
        self._model.eval()

    @torch.inference_mode()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (16,) 또는 (1,16) → action: (6,), [-1,1] 클램프까지 여기서 수행
        (sim의 `_pre_physics_step`이 `actions.clamp(-1.0, 1.0)`을 하는 것과 동일하게
        맞춤 — 정책 출력은 Gaussian 평균이라 하드 클램프 없이 [-1,1]을 벗어날 수 있음)."""
        x = obs if obs.dim() == 2 else obs.unsqueeze(0)
        action = self._model(x.to(self.device))
        return action.squeeze(0).clamp(-1.0, 1.0)
