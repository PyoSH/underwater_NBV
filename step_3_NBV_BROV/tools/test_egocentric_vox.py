"""`envs/vox_transform.to_egocentric()` 검증 — Isaac 없이 CPU에서 돈다.

왜 별도 테스트인가: 회전 부호와 `grid_sample`의 축 순서(W,H,D 역순)는 틀려도
예외가 나지 않는다. 관측이 그럴듯한 쓰레기가 될 뿐이라 학습 결과로만 드러나고,
그때는 7시간을 버린 뒤다. 그래서 표식을 알려진 자리에 놓고 알려진 자리로
오는지 직접 확인한다.

    /isaac-sim/python.sh -u tools/test_egocentric_vox.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from envs.vox_transform import to_egocentric   # noqa: E402

N = 20
CEN = (N - 1) / 2.0
R = 6.0                       # 표식을 놓을 반경 [voxel]
fails = []


def put_marker(azim_deg: float, z_off: float = 0.0) -> torch.Tensor:
    """월드 방위각 `azim_deg`, 반경 R 자리에 ch2 표식을 놓은 격자 (1,3,N,N,N)."""
    v = torch.zeros(1, 3, N, N, N)
    v[0, 0] = 1.0                                     # 전부 미관측이 기본
    a = math.radians(azim_deg)
    ix = int(round(CEN + R * math.cos(a)))
    iy = int(round(CEN + R * math.sin(a)))
    iz = int(round(CEN + z_off))
    v[0, 0, ix, iy, iz] = 0.0
    v[0, 2, ix, iy, iz] = 1.0
    return v


def marker_azimuth(v: torch.Tensor) -> tuple:
    """ch2 최대 위치의 방위각[deg]과 z 인덱스."""
    ch2 = v[0, 2]
    idx = int(torch.argmax(ch2))
    ix, iy, iz = idx // (N * N), (idx // N) % N, idx % N
    return math.degrees(math.atan2(iy - CEN, ix - CEN)) % 360.0, iz


def check(name, got, want, tol):
    d = abs((got - want + 180.0) % 360.0 - 180.0)
    ok = d <= tol
    print(f"  {name:<44} {got:7.1f}° (기대 {want:5.1f}°, 오차 {d:4.1f}°) {'OK' if ok else '실패'}")
    if not ok:
        fails.append(name)


print("[1] 에이전트가 선 방위각이 출력의 +X(0°)로 오는가")
for azim in (0.0, 45.0, 90.0, 180.0, 270.0):
    out = to_egocentric(put_marker(azim), torch.tensor([math.radians(azim)]))
    got, _ = marker_azimuth(out)
    check(f"표식·에이전트 모두 {azim:5.1f}°", got, 0.0, 12.0)

print("\n[2] 상대 방위각이 보존되는가 (표식 − 에이전트)")
for m, a in ((90.0, 0.0), (180.0, 90.0), (0.0, 90.0), (270.0, 180.0)):
    out = to_egocentric(put_marker(m), torch.tensor([math.radians(a)]))
    got, _ = marker_azimuth(out)
    check(f"표식 {m:5.1f}° − 에이전트 {a:5.1f}°", got, (m - a) % 360.0, 12.0)

print("\n[3] z축(높이)은 회전에 불변인가")
v = put_marker(0.0, z_off=4.0)
out = to_egocentric(v, torch.tensor([math.radians(90.0)]))
_, iz_in = marker_azimuth(v)
_, iz_out = marker_azimuth(out)
ok = iz_in == iz_out
print(f"  입력 z={iz_in} → 출력 z={iz_out}  {'OK' if ok else '실패'}")
if not ok:
    fails.append("z 불변")

print("\n[4] 회전으로 비는 모서리가 '미관측'으로 채워지는가")
v = torch.zeros(1, 3, N, N, N); v[0, 1] = 1.0            # 전부 '빈 공간'
out = to_egocentric(v, torch.tensor([math.radians(45.0)]))
corner_unknown = out[0, 0, 0, 0, 0].item()
tot = out[0].sum(dim=0)
ok = corner_unknown > 0.9 and tot.min().item() > 0.9
print(f"  모서리 ch0={corner_unknown:.3f} (>0.9), 채널합 최소={tot.min().item():.3f} (>0.9)"
      f"  {'OK' if ok else '실패'}")
if not ok:
    fails.append("모서리 채움")

print("\n[5] theta=0이면 항등인가")
v = put_marker(30.0)
out = to_egocentric(v, torch.tensor([0.0]))
d = (out - v).abs().max().item()
ok = d < 1e-5
print(f"  최대 차이 {d:.2e} (<1e-5)  {'OK' if ok else '실패'}")
if not ok:
    fails.append("항등")

print("\n" + ("전부 통과" if not fails else f"실패 {len(fails)}건: {fails}"))
sys.exit(1 if fails else 0)
