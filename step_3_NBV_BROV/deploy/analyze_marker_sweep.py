"""marker_sweep.csv 요약 — 입사각/화각 중 무엇이 구속인지, 대역별 가용률. 2026-09-10."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    n = len(rows)
    det = [r for r in rows if r["detected"] == "1"]
    print(f"시점 {n}개, 검출 {len(det)} ({100*len(det)/n:.1f}%)")

    bad_frame = [r for r in rows if r.get("frame", "ok") != "ok"]
    if bad_frame:
        print(f"  ⚠ 프레임 정착 실패 {len(bad_frame)}개 — 측정 신뢰도 확인 필요")

    print("\n입사각 구간별 (마커 법선 대비):")
    bins = [(0, 45), (45, 60), (60, 65), (65, 70), (70, 75), (75, 80), (80, 90), (90, 180)]
    for lo, hi in bins:
        sub = [r for r in rows if lo <= float(r["incidence_deg"]) < hi]
        if not sub:
            continue
        d = sum(1 for r in sub if r["detected"] == "1")
        print(f"  [{lo:3d},{hi:3d})  n={len(sub):4d}  검출 {d:4d} ({100*d/len(sub):5.1f}%)")

    print("\n(psi, phi) 격자 검출률 [%]:")
    psis = sorted({float(r["psi"]) for r in rows})
    phis = sorted({float(r["phi_deg"]) for r in rows})
    print("  psi\\phi " + " ".join(f"{p:5.0f}" for p in phis))
    for ps in psis:
        cells = []
        for ph in phis:
            sub = [r for r in rows if float(r["psi"]) == ps and float(r["phi_deg"]) == ph]
            cells.append("    -" if not sub else
                         f"{100*sum(1 for r in sub if r['detected']=='1')/len(sub):5.0f}")
        print(f"  {ps:5.2f}  " + " ".join(cells))

    if det:
        pe = sorted(float(r["pos_err_m"]) for r in det if r["pos_err_m"])
        re_ = sorted(float(r["rot_err_deg"]) for r in det if r["rot_err_deg"])
        if pe:
            print(f"\npose 오차 (검출된 {len(pe)}개): 위치 중앙 {pe[len(pe)//2]*1000:.1f} mm "
                  f"/ 최대 {pe[-1]*1000:.1f} mm")
            print(f"                      자세 중앙 {re_[len(re_)//2]:.2f} deg "
                  f"/ 최대 {re_[-1]:.2f} deg")
        qs = [r["q_wm_xyzw"] for r in det if r.get("q_wm_xyzw")]
        if qs:
            import numpy as np
            Q = np.array([[float(v) for v in q.split()] for q in qs])
            Q *= np.sign(Q[:, 3:4] + 1e-12)          # 반구 정규화
            m = Q.mean(0); m /= np.linalg.norm(m)
            sd = np.abs(Q - m).max(0)
            print(f"\n측정된 world<-marker 쿼터니언 xyzw = "
                  f"[{m[0]:+.5f}, {m[1]:+.5f}, {m[2]:+.5f}, {m[3]:+.5f}]")
            print(f"  표본 편차(최대) {sd.max():.5f}  (n={len(Q)})")
            print("  -> pool 원점이 수조 바닥 중앙이고 축이 world 와 같으므로 "
                  "이 값이 aruco.yaml 의 pool_to_marker_quaternion_xyzw 다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
