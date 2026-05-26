"""
analyze_results.py
==================
evaluate_recon.py / evaluate_basic.py 가 생성한 결과를 집계해
비교 테이블(CSV) + coverage 커브 플롯을 출력.

사용법
------
python analyze_results.py \
    --results \
        UW_NBV_2:./recon_output/UW_NBV_2_327k \
        GenNBV:./recon_output/genNBV \
        ScanRL:./recon_output/scanRL \
        Manual:./recon_output/basic_orbit \
    --success_thr 0.82 \
    --out_dir ./analysis
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_episodes(result_dir: Path) -> list[dict]:
    """ep_XXX_env* 디렉토리마다 step_log.csv + npy 파일을 읽어 반환."""
    episodes = []
    for ep_dir in sorted(result_dir.glob("ep_*_env*")):
        log_path = ep_dir / "step_log.csv"
        if not log_path.exists():
            continue

        rows = list(csv.DictReader(open(log_path)))
        if not rows:
            continue

        last = rows[-1]
        # 신형(evaluate_recon/evaluate_basic): coverage_bin + coverage_q
        # 구형: coverage (binary만)
        if "coverage_q" in last:
            cov_bin = float(last["coverage_bin"])
            cov_q   = float(last["coverage_q"])
        else:
            cov_bin = float(last.get("coverage", last.get("coverage_bin", 0)))
            cov_q   = cov_bin  # 구형에는 quality 정보 없음

        ep_len = int(last["step"])

        bin_hist = (np.load(str(ep_dir / "coverage_bin_hist.npy"))
                    if (ep_dir / "coverage_bin_hist.npy").exists()
                    else np.array([cov_bin]))
        q_hist   = (np.load(str(ep_dir / "coverage_q_hist.npy"))
                    if (ep_dir / "coverage_q_hist.npy").exists()
                    else np.array([cov_q]))

        episodes.append({
            "coverage_bin": cov_bin,
            "coverage_q":   cov_q,
            "ep_len":        ep_len,
            "bin_hist":      bin_hist,
            "q_hist":        q_hist,
        })
    return episodes


def compute_stats(episodes: list[dict], success_thr: float) -> dict | None:
    if not episodes:
        return None
    cov_qs   = [e["coverage_q"]   for e in episodes]
    cov_bins = [e["coverage_bin"] for e in episodes]
    ep_lens  = [e["ep_len"]       for e in episodes]
    return {
        "n":            len(episodes),
        "cov_q_mean":   float(np.mean(cov_qs)),
        "cov_q_std":    float(np.std(cov_qs)),
        "cov_bin_mean": float(np.mean(cov_bins)),
        "cov_bin_std":  float(np.std(cov_bins)),
        "ep_len_mean":  float(np.mean(ep_lens)),
        "ep_len_std":   float(np.std(ep_lens)),
        "success_rate": float(np.mean([q >= success_thr for q in cov_qs])),
    }


def print_table(stats: dict[str, dict | None], success_thr: float):
    cols = f"{'Algorithm':<16} {'N':>4}  {'coverage_q':^18}  {'coverage_bin':^18}  {'success%':>9}  {'ep_len':^14}"
    sep  = "─" * len(cols)
    print(f"\n{sep}\n{cols}\n{sep}")
    for name, s in stats.items():
        if s is None:
            print(f"{name:<16}  (결과 없음)")
            continue
        cq = f"{s['cov_q_mean']:.3f} ± {s['cov_q_std']:.3f}"
        cb = f"{s['cov_bin_mean']:.3f} ± {s['cov_bin_std']:.3f}"
        sr = f"{s['success_rate']*100:.1f}%"
        el = f"{s['ep_len_mean']:.1f} ± {s['ep_len_std']:.1f}"
        print(f"{name:<16} {s['n']:>4}  {cq:^18}  {cb:^18}  {sr:>9}  {el:^14}")
    print(sep)
    print(f"  success threshold = coverage_q ≥ {success_thr}\n")


def save_table_csv(stats: dict[str, dict | None], out_path: Path, success_thr: float):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "n",
                    "cov_q_mean", "cov_q_std",
                    "cov_bin_mean", "cov_bin_std",
                    "success_rate", "ep_len_mean", "ep_len_std",
                    "success_thr"])
        for name, s in stats.items():
            if s is None:
                continue
            w.writerow([name, s["n"],
                        f"{s['cov_q_mean']:.4f}",   f"{s['cov_q_std']:.4f}",
                        f"{s['cov_bin_mean']:.4f}",  f"{s['cov_bin_std']:.4f}",
                        f"{s['success_rate']:.4f}",
                        f"{s['ep_len_mean']:.2f}",   f"{s['ep_len_std']:.2f}",
                        success_thr])
    print(f"[save] {out_path}")


def plot_curves(algo_episodes: dict[str, list[dict]], hist_key: str,
                ylabel: str, title: str, out_path: Path, success_thr: float):
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, episodes) in enumerate(algo_episodes.items()):
        if not episodes:
            continue
        hists   = [e[hist_key] for e in episodes]
        max_len = max(len(h) for h in hists)
        padded  = np.array([
            np.pad(h, (0, max_len - len(h)), mode="edge") for h in hists
        ])
        steps = np.arange(1, max_len + 1)
        mu    = padded.mean(axis=0)
        sigma = padded.std(axis=0)
        c     = colors[i % len(colors)]
        ls    = linestyles[i % len(linestyles)]
        ax.plot(steps, mu, label=name, color=c, linewidth=2,
                linestyle=ls, zorder=len(algo_episodes) - i)
        ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.15, color=c,
                        zorder=len(algo_episodes) - i)

    ax.axhline(y=success_thr, color="red", linestyle="--", linewidth=1,
               label=f"success thr ({success_thr})")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"[save] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="name:path 형식. 예) UW_NBV_2:./recon_output/UW_NBV_2_327k")
    parser.add_argument("--success_thr", type=float, default=0.82)
    parser.add_argument("--out_dir", type=str, default="./analysis")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # name:path 파싱
    algo_dirs: dict[str, Path] = {}
    for item in args.results:
        if ":" in item:
            name, path = item.split(":", 1)
        else:
            name, path = Path(item).name, item
        algo_dirs[name] = Path(path)

    # 결과 로드
    algo_episodes: dict[str, list[dict]] = {}
    stats: dict[str, dict | None] = {}

    for name, d in algo_dirs.items():
        if not d.exists():
            print(f"[warn] {name}: {d} 없음")
            algo_episodes[name] = []
            stats[name] = None
            continue
        eps = load_episodes(d)
        print(f"[load] {name}: {len(eps)} episodes  ← {d}")
        algo_episodes[name] = eps
        stats[name] = compute_stats(eps, args.success_thr)

    # 테이블 출력 + 저장
    print_table(stats, args.success_thr)
    save_table_csv(stats, out_dir / "comparison_table.csv", args.success_thr)

    # 커브 플롯
    plot_curves(algo_episodes, "q_hist",   "coverage_q",   "Quality Coverage (mean±std)",
                out_dir / "coverage_q_curve.png",   args.success_thr)
    plot_curves(algo_episodes, "bin_hist", "coverage_bin", "Binary Coverage (mean±std)",
                out_dir / "coverage_bin_curve.png", args.success_thr)


if __name__ == "__main__":
    main()
