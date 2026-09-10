"""Gazebo 베이스라인 CSV -> 결정 수별 cov_bin 표 + ceiling 대비. 2026-09-10.

질문(계획서 §15.7): 신 카메라(실기 화각)로는 과업이 너무 쉬워진 게 아닌가 —
orbit 이 ceiling 에 붙으면 RL 이 더할 것은 거리가 아니라 다양성/가려짐뿐이다.
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

CHECKPOINTS = (3, 5, 10, 15, 20, 25, 30, 39)


def load(path: str) -> np.ndarray:
    rows = list(csv.DictReader(open(path)))
    cov = np.array([float(r["cov_bin"]) if r["cov_bin"] not in ("", "nan") else np.nan
                    for r in rows])
    return cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    args = ap.parse_args()

    sweep = sorted(glob.glob(str(args.dir / "loop_sweep_s*.csv")))
    ceiling = None
    if sweep:
        c = load(sweep[0])
        ceiling = float(np.nanmax(c))
        print(f"ceiling (sweep {len(c)}결정 최대 cov_bin): {ceiling:.4f}")
        print(f"  sweep 경과: " + "  ".join(
            f"@{k} {c[k-1]:.3f}" for k in (12, 24, 48, 72, 100) if k <= len(c)))

    print(f"\n{'정책':8s}" + "".join(f"{'@'+str(k):>8}" for k in CHECKPOINTS) + "   시드")
    summary = {}
    for pol in ("orbit", "random"):
        files = sorted(glob.glob(str(args.dir / f"loop_{pol}_s*.csv")))
        if not files:
            continue
        curves = [load(f) for f in files]
        n = min(len(c) for c in curves)
        M = np.stack([c[:n] for c in curves])
        mean = np.nanmean(M, axis=0)
        row = f"{pol:8s}"
        for k in CHECKPOINTS:
            row += f"{mean[k-1]:8.3f}" if k <= n else f"{'-':>8}"
        print(row + f"   n={len(files)}")
        lo = np.nanmin(M, axis=0); hi = np.nanmax(M, axis=0)
        print(f"{'  범위':8s}" + "".join(
            f"{lo[k-1]:.2f}-{hi[k-1]:.2f}".rjust(8) if k <= n else f"{'-':>8}"
            for k in CHECKPOINTS))
        summary[pol] = mean

    if ceiling and summary:
        print(f"\nceiling 대비 비율:")
        for pol, mean in summary.items():
            print(f"  {pol:8s}" + "".join(
                f"{mean[k-1]/ceiling:8.2f}" if k <= len(mean) else f"{'-':>8}"
                for k in CHECKPOINTS))
        if "orbit" in summary and "random" in summary:
            o, r = summary["orbit"], summary["random"]
            n = min(len(o), len(r))
            print(f"\nrandom - orbit 간격: " + "  ".join(
                f"@{k} {r[k-1]-o[k-1]:+.3f}" for k in CHECKPOINTS if k <= n))
            print("  (양수면 시점 선택에 여지가 있다 = 과업이 성립한다. §2 GSO 기준 0.25)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
