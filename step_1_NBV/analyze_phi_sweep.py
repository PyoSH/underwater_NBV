"""
analyze_phi_sweep.py
=====================
phi_sweep 디렉토리 구조:
  sweep_dir/
    UW_NBV_2_phi20/  UW_NBV_2_phi35/  ...  UW_NBV_2_phi80/
    Manual_phi20/    Manual_phi35/    ...  Manual_phi80/

각 phi별 UW_NBV_2 vs Manual coverage_q 를 비교해
  - 테이블 (CSV + 터미널 출력)
  - phi vs coverage_q 꺾은선 그래프
  - UW_NBV_2 / Manual 비율 (ratio > 1 이면 UW_NBV_2 우세)

사용법
------
python analyze_phi_sweep.py \
    --sweep_dir ./recon_output/phi_sweep \
    --phi_vals 20 35 50 65 80 \
    --success_thr 0.82 \
    --out_dir ./analysis/phi_sweep
"""

from __future__ import annotations
import argparse, csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_cov_q(ep_dir: Path) -> float | None:
    log = ep_dir / "step_log.csv"
    if not log.exists():
        return None
    rows = list(csv.DictReader(open(log)))
    if not rows:
        return None
    last = rows[-1]
    if "coverage_q" in last:
        return float(last["coverage_q"])
    return float(last.get("coverage", 0))


def load_mean_cov_q(algo_dir: Path) -> tuple[float, float, int]:
    """(mean, std, n)"""
    vals = []
    for ep in sorted(algo_dir.glob("ep_*_env*")):
        v = load_cov_q(ep)
        if v is not None:
            vals.append(v)
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir",   type=str, required=True)
    parser.add_argument("--phi_vals",    type=int, nargs="+", default=[20, 35, 50, 65, 80])
    parser.add_argument("--success_thr", type=float, default=0.82)
    parser.add_argument("--out_dir",     type=str, default="./analysis/phi_sweep")
    args = parser.parse_args()

    sweep = Path(args.sweep_dir)
    out   = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for phi in args.phi_vals:
        uw_dir  = sweep / f"UW_NBV_2_phi{phi}"
        man_dir = sweep / f"Manual_phi{phi}"

        uw_mean, uw_std, uw_n   = load_mean_cov_q(uw_dir)
        man_mean, man_std, man_n = load_mean_cov_q(man_dir)
        ratio = uw_mean / man_mean if man_mean > 0 else float("nan")

        rows.append(dict(phi=phi,
                         uw_mean=uw_mean, uw_std=uw_std, uw_n=uw_n,
                         man_mean=man_mean, man_std=man_std, man_n=man_n,
                         ratio=ratio))
        print(f"phi={phi:2d}°  UW={uw_mean:.3f}±{uw_std:.3f}(n={uw_n})  "
              f"Manual={man_mean:.3f}±{man_std:.3f}(n={man_n})  "
              f"ratio={ratio:.3f}{'  ← UW우세' if ratio > 1 else ''}")

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    csv_path = out / "phi_sweep_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_deg", "UW_cov_q", "UW_std", "UW_n",
                    "Manual_cov_q", "Manual_std", "Manual_n", "ratio"])
        for r in rows:
            w.writerow([r["phi"], f"{r['uw_mean']:.4f}", f"{r['uw_std']:.4f}", r["uw_n"],
                        f"{r['man_mean']:.4f}", f"{r['man_std']:.4f}", r["man_n"],
                        f"{r['ratio']:.4f}"])
    print(f"\n[save] {csv_path}")

    # ── 그래프 ────────────────────────────────────────────────────────────────
    phis     = [r["phi"]     for r in rows]
    uw_mus   = [r["uw_mean"] for r in rows]
    uw_stds  = [r["uw_std"]  for r in rows]
    man_mus  = [r["man_mean"] for r in rows]
    man_stds = [r["man_std"]  for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 왼쪽: coverage_q vs phi
    ax = axes[0]
    ax.errorbar(phis, uw_mus,  yerr=uw_stds,  fmt="-o", label="UW_NBV_2",
                color="#1f77b4", capsize=4, linewidth=2)
    ax.errorbar(phis, man_mus, yerr=man_stds, fmt="--s", label="Manual Orbit",
                color="#d62728", capsize=4, linewidth=2)
    ax.axhline(y=args.success_thr, color="gray", linestyle=":", linewidth=1,
               label=f"success thr ({args.success_thr})")
    ax.set_xlabel("eval_phi (°)")
    ax.set_ylabel("coverage_q (mean±std)")
    ax.set_title("Coverage_q vs Starting Elevation")
    ax.set_xticks(phis)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 오른쪽: UW / Manual 비율
    ax = axes[1]
    ratios = [r["ratio"] for r in rows]
    colors = ["#1f77b4" if r >= 1 else "#d62728" for r in ratios]
    ax.bar(phis, ratios, width=8, color=colors, alpha=0.75)
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("eval_phi (°)")
    ax.set_ylabel("UW_NBV_2 / Manual  (>1 = UW 우세)")
    ax.set_title("UW_NBV_2 / Manual Ratio")
    ax.set_xticks(phis)
    ax.set_ylim(0, 1.3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig_path = out / "phi_sweep.png"
    plt.savefig(str(fig_path), dpi=150)
    plt.close()
    print(f"[save] {fig_path}")


if __name__ == "__main__":
    main()
