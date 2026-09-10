"""voxel 격자 좌표 변환 — torch만 쓰고 Isaac을 import하지 않는다.

분리해 둔 이유: 회전 방향과 `grid_sample`의 축 순서는 부호 하나만 틀려도
학습이 조용히 망가진다(관측이 그럴듯한 쓰레기가 될 뿐 오류가 안 난다).
Isaac 없이 도는 모듈이라야 `tools/test_egocentric_vox.py`가 합성 표식으로
직접 검증할 수 있다.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def to_egocentric(vox: torch.Tensor, theta: torch.Tensor,
                  base: tuple | None = None) -> torch.Tensor:
    """물체중심 월드 격자 (E,C,Nx,Ny,Nz)를 **에이전트 방위각 기준**으로 돌린다.

    회전은 월드 z축(수직) 둘레로만 한다. 방위각 theta는 주기적이고 행동
    (dtheta)의 축이라 좌표변환 부담이 가장 크지만, 고도각 phi는 유계 스칼라라
    네트워크가 그대로 다룰 수 있고, z를 축으로 두면 '물체의 위/아래'라는 의미도
    그대로 남는다.

    출력 격자의 방위각 b는 입력의 b+theta에서 읽는다 — 그래야 에이전트가 있는
    월드 방위각 theta가 출력의 +X(b=0)로 온다.

    `grid_sample`의 grid 마지막 축은 (W,H,D) **역순**이다. 입력이
    (E,C,Nx,Ny,Nz)이므로 D=Nx(월드X), H=Ny(월드Y), W=Nz(월드Z)이고 따라서
    grid는 [z, y, x] 순으로 쌓는다.

    범위 밖(회전으로 비는 모서리)은 **ch0=1(미관측)**으로 채운다. 0으로 두면
    세 채널이 모두 0인, 실제로는 생기지 않는 상태가 되어 인코더에 학습 데이터에
    없는 패턴을 준다.

    `base`는 `(gx, gy, gz)` 정규화 좌표 격자. 호출자가 캐시해 재사용한다.
    """
    E, C = vox.shape[0], vox.shape[1]
    Nx, Ny, Nz = vox.shape[2], vox.shape[3], vox.shape[4]
    if base is None:
        base = make_base(Nx, Ny, Nz, vox.device)
    gx, gy, gz = base

    c = torch.cos(theta).view(-1, 1, 1, 1)
    s = torch.sin(theta).view(-1, 1, 1, 1)
    sx = c * gx - s * gy
    sy = s * gx + c * gy
    sz = gz.expand_as(sx)
    grid = torch.stack([sz, sy, sx], dim=-1)                 # (E,Nx,Ny,Nz,3)

    ones = torch.ones(E, 1, Nx, Ny, Nz, device=vox.device, dtype=vox.dtype)
    out = F.grid_sample(torch.cat([vox, ones], dim=1), grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    valid = out[:, C:C + 1]
    res = out[:, :C].clone()
    res[:, 0:1] = (res[:, 0:1] + (1.0 - valid)).clamp(0.0, 1.0)
    return res


def make_base(Nx: int, Ny: int, Nz: int, device) -> tuple:
    """정규화 좌표 격자 (gx,gy,gz). `align_corners=True`와 짝이다."""
    def lin(n):
        return torch.linspace(-1.0, 1.0, n, device=device)
    return torch.meshgrid(lin(Nx), lin(Ny), lin(Nz), indexing="ij")
