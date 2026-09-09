"""정책이 voxel 지도를 실제로 쓰는지 측정한다 — Isaac 불필요, CPU 수초.

왜 필요한가
-----------
2026-09-09에 run06 최종 정책이 지도를 **통째로 무시**하고 있음이 드러났다.
평가에서 voxel 관측을 다른 env 것으로 뒤섞거나(shuffle) 0으로 지워도(zero)
holdout coverage가 0.750 → 0.747 / 0.748로 그대로였다. 즉 행동이 자기
구면좌표만의 함수였고, "이미 본 곳을 피해 못 본 곳을 고른다"는 NBV의 정의가
성립하지 않았다.

그 절제 실험은 Isaac 씬이 필요해 6분씩 걸린다. 이 도구는 같은 질문을
네트워크 수준에서 수초에 답한다 — 체크포인트마다 돌려 **학습 경과에 따른
지도 사용도 곡선**을 그릴 수 있는 것이 절제 실험에는 없는 이점이다.
실제로 그 곡선이 원인을 특정했다: run06은 처음엔 지도를 쓰다가(롤아웃 8~64에서
지도 임베딩 노름 4.8, 민감도비 10~85) 점점 버렸다(롤아웃 128에서 0.13, 0.02).

무엇을 재는가
-------------
같은 물체를 좌반구에서만 본 지도와 우반구에서만 본 지도를 주고 행동 평균
mu의 변화량을 잰다. 비교군은 자기좌표를 방위각 36도만큼 옮겼을 때의 변화량이다.
둘의 비가 1보다 크게 크면 지도를 읽는 것이고, 0에 가까우면 무시하는 것이다.

    /isaac-sim/python.sh -u tools/measure_map_sensitivity.py checkpoints/nbuv_run08
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from algorithm.algo_nbv_continuous import Actor   # noqa: E402

DEV = "cpu"
N = 20


def make_map(seen_side: str, quality: float = 0.8) -> torch.Tensor:
    """구각 표면 물체를 한쪽에서만 관측한 지도 (1,3,N,N,N).

    채널은 `envs/env.py::_get_vox_actor()`와 같다 — ch0 미관측 / ch1 빈공간 /
    ch2 관측 품질(연속). ch2를 연속으로 두는 것이 중요하다: 이 값이 0/1로
    잘리던 것이 지도 사용 붕괴의 원인이었다(RolloutBuffer docstring 참조).
    """
    g = torch.arange(N).float() - (N - 1) / 2
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    r = (X ** 2 + Y ** 2 + Z ** 2).sqrt()
    surf = (r > 4.0) & (r < 5.5)
    inside = r <= 4.0
    if seen_side == "none":
        seen = torch.zeros_like(surf)
    else:
        seen = (X > 0) if seen_side == "left" else (X < 0)
    obs_surf = surf & seen
    free = (~surf) & (~inside) & seen
    return torch.stack([(~(obs_surf | free)).float(),
                        free.float(),
                        obs_surf.float() * quality]).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser(description="지도 사용도 측정")
    ap.add_argument("paths", nargs="+",
                    help="체크포인트 파일 또는 디렉터리(안의 *.pt를 순서대로)")
    args = ap.parse_args()

    files = []
    for p in args.paths:
        files += sorted(glob.glob(os.path.join(p, "*.pt"))) if os.path.isdir(p) else [p]

    torch.manual_seed(0)
    img = torch.rand(1, 2, 84, 84)
    sc = torch.tensor([[0.5, 0.5, 0.4]])
    sc_az = torch.tensor([[0.6, 0.5, 0.4]])          # 방위각 +36도

    @torch.no_grad()
    def mu(actor, vox, scalar=sc):
        return actor._dist(vox, img, scalar).mean[0]

    print(f"{'체크포인트':>28}{'지도노름':>10}{'좌↔우':>9}{'미관측대비':>11}"
          f"{'ch2품질':>9}{'자기좌표':>9}{'비':>8}")
    for f in files:
        actor = Actor().to(DEV)
        actor.load_state_dict(torch.load(f, map_location=DEV)["actor"])
        actor.eval()
        with torch.no_grad():
            gn = actor.embed.geo(make_map("left")).norm().item()
        base = mu(actor, make_map("left"))
        d_side = (mu(actor, make_map("right")) - base).norm().item()
        d_none = (mu(actor, make_map("none")) - base).norm().item()
        d_ch2 = (mu(actor, make_map("left", 1.0))
                 - mu(actor, make_map("left", 0.1))).norm().item()
        d_sc = (mu(actor, make_map("left"), sc_az) - base).norm().item()
        name = os.path.basename(f)
        print(f"{name:>28}{gn:>10.2f}{d_side:>9.4f}{d_none:>11.4f}"
              f"{d_ch2:>9.4f}{d_sc:>9.4f}{d_side / max(d_sc, 1e-9):>8.2f}")
    print("\n  비 = 지도 좌우반전 민감도 / 자기좌표 방위각 민감도.")
    print("  1보다 크게 크면 지도를 읽는 정책, 0에 가까우면 자기좌표만 쓰는 정책이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
