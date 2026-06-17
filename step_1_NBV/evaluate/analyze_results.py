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


JERLOV_TYPES = ["IB", "II", "III", "1C", "3C", "5C"]


def load_episodes(result_dir: Path, jerlov_eval: bool = False) -> list[dict]:
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

        ep_idx = int(ep_dir.name.split("_")[1])
        jerlov_type = JERLOV_TYPES[ep_idx % len(JERLOV_TYPES)] if jerlov_eval else None

        episodes.append({
            "coverage_bin": cov_bin,
            "coverage_q":   cov_q,
            "ep_len":        ep_len,
            "bin_hist":      bin_hist,
            "q_hist":        q_hist,
            "jerlov_type":   jerlov_type,
        })
    return episodes


MAX_STEPS = 50  # episode budget for AUC normalisation


def _auc(hist: np.ndarray) -> float:
    padded = np.pad(hist, (0, max(0, MAX_STEPS - len(hist))), mode="edge")
    return float(padded.mean())


def compute_stats(episodes: list[dict]) -> dict | None:
    if not episodes:
        return None
    cov_qs   = [e["coverage_q"]   for e in episodes]
    cov_bins = [e["coverage_bin"] for e in episodes]
    ep_lens  = [e["ep_len"]       for e in episodes]
    aucs_q   = [_auc(e["q_hist"])   for e in episodes]
    aucs_bin = [_auc(e["bin_hist"]) for e in episodes]
    return {
        "n":            len(episodes),
        "cov_q_mean":   float(np.mean(cov_qs)),
        "cov_q_std":    float(np.std(cov_qs)),
        "cov_q_max":    float(np.max(cov_qs)),
        "cov_bin_mean": float(np.mean(cov_bins)),
        "cov_bin_std":  float(np.std(cov_bins)),
        "cov_bin_max":  float(np.max(cov_bins)),
        "auc_q_mean":   float(np.mean(aucs_q)),
        "auc_q_std":    float(np.std(aucs_q)),
        "auc_bin_mean": float(np.mean(aucs_bin)),
        "auc_bin_std":  float(np.std(aucs_bin)),
        "ep_len_mean":  float(np.mean(ep_lens)),
        "ep_len_std":   float(np.std(ep_lens)),
    }


def print_table(stats: dict[str, dict | None]):
    cols = (f"{'Algorithm':<16} {'N':>4}  {'cov_q mean±std':^20}  {'cov_q max':^10}"
            f"  {'AUC_q mean±std':^20}  {'cov_bin mean±std':^20}  {'cov_bin max':^11}"
            f"  {'AUC_bin mean±std':^20}  {'ep_len':^14}")
    sep  = "─" * len(cols)
    print(f"\n{sep}\n{cols}\n{sep}")
    for name, s in stats.items():
        if s is None:
            print(f"{name:<16}  (결과 없음)")
            continue
        cq    = f"{s['cov_q_mean']:.3f} ± {s['cov_q_std']:.3f}"
        cqmax = f"{s['cov_q_max']:.3f}"
        aq    = f"{s['auc_q_mean']:.3f} ± {s['auc_q_std']:.3f}"
        cb    = f"{s['cov_bin_mean']:.3f} ± {s['cov_bin_std']:.3f}"
        cbmax = f"{s['cov_bin_max']:.3f}"
        ab    = f"{s['auc_bin_mean']:.3f} ± {s['auc_bin_std']:.3f}"
        el    = f"{s['ep_len_mean']:.1f} ± {s['ep_len_std']:.1f}"
        print(f"{name:<16} {s['n']:>4}  {cq:^20}  {cqmax:^10}"
              f"  {aq:^20}  {cb:^20}  {cbmax:^11}"
              f"  {ab:^20}  {el:^14}")
    print(sep)
    print(f"  AUC normalised by {MAX_STEPS} steps\n")


def save_table_csv(stats: dict[str, dict | None], out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "n",
                    "cov_q_mean", "cov_q_std", "cov_q_max",
                    "auc_q_mean", "auc_q_std",
                    "cov_bin_mean", "cov_bin_std", "cov_bin_max",
                    "auc_bin_mean", "auc_bin_std",
                    "ep_len_mean", "ep_len_std"])
        for name, s in stats.items():
            if s is None:
                continue
            w.writerow([name, s["n"],
                        f"{s['cov_q_mean']:.4f}",   f"{s['cov_q_std']:.4f}",  f"{s['cov_q_max']:.4f}",
                        f"{s['auc_q_mean']:.4f}",   f"{s['auc_q_std']:.4f}",
                        f"{s['cov_bin_mean']:.4f}",  f"{s['cov_bin_std']:.4f}", f"{s['cov_bin_max']:.4f}",
                        f"{s['auc_bin_mean']:.4f}",  f"{s['auc_bin_std']:.4f}",
                        f"{s['ep_len_mean']:.2f}",   f"{s['ep_len_std']:.2f}"])
    print(f"[save] {out_path}")


# colorblind-friendly palette (Wong 2011)
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--"]


def plot_curves(algo_episodes: dict[str, list[dict]], hist_key: str,
                ylabel: str, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

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
        c  = PALETTE[i % len(PALETTE)]
        ls = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(steps, mu, label=name, color=c, linewidth=2.5,
                linestyle=ls, zorder=len(algo_episodes) - i)
        ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.15, color=c,
                        zorder=len(algo_episodes) - i)

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(1, MAX_STEPS)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")


def plot_bar_chart(stats: dict[str, dict | None], out_path: Path):
    RENAME = {"UW_NBV_5": "Proposed"}
    COLOR_MAP = {
        "Manual":   "#888888",   # 회색
        "ScanRL":   "#D62728",   # 빨강
        "GenNBV":   "#AEC6E8",   # 옅은 파랑
        "Proposed": "#2CA02C",   # 밝은 초록
    }
    names  = [RENAME.get(n, n) for n, s in stats.items() if s is not None]
    s_list = [stats[n] for n in (k for k, s in stats.items() if s is not None)]

    metrics = [
        ("cov_q_mean",  "cov_q_std",  "Quality-aware Coverage\n(cov_q)"),
        ("auc_q_mean",  "auc_q_std",  f"AUC Quality-aware\n(AUC_q / {MAX_STEPS} steps)"),
        ("cov_bin_mean","cov_bin_std", "Binary Coverage\n(cov_bin)"),
        ("auc_bin_mean","auc_bin_std", f"AUC Binary\n(AUC_bin / {MAX_STEPS} steps)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5), sharey=False)

    x = np.arange(len(names))
    bar_w = 0.55

    for ax, (mean_key, std_key, label) in zip(axes, metrics):
        means = [s[mean_key] for s in s_list]
        stds  = [s[std_key]  for s in s_list]
        colors = [COLOR_MAP.get(n, PALETTE[i % len(PALETTE)]) for i, n in enumerate(names)]
        bars = ax.bar(x, means, width=bar_w, color=colors,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                    capsize=4, linewidth=1.5, zorder=4)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.1 + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylim(0, min(1.0, max(means) * 1.4 + 0.05))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)
        ax.tick_params(axis="y", labelsize=10)

    fig.suptitle("Algorithm Comparison", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")


def plot_jerlov_bar(algo_episodes: dict[str, list[dict]], out_path: Path):
    """수종별 grouped bar chart (cov_q / cov_bin 두 subplot)."""
    algos = [n for n, eps in algo_episodes.items() if eps]
    n_algos = len(algos)
    x = np.arange(len(JERLOV_TYPES))
    bar_w = 0.8 / n_algos
    offsets = np.linspace(-(n_algos - 1) / 2 * bar_w, (n_algos - 1) / 2 * bar_w, n_algos)

    # type → {algo → value} 빌드
    type_vals: dict[str, dict[str, dict]] = {t: {} for t in JERLOV_TYPES}
    for name, eps in algo_episodes.items():
        for e in eps:
            t = e.get("jerlov_type")
            if t in type_vals:
                type_vals[t][name] = {"cov_q": e["coverage_q"], "cov_bin": e["coverage_bin"]}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in zip(
        axes,
        ["cov_q", "cov_bin"],
        ["coverage_q", "coverage_bin"],
    ):
        for i, name in enumerate(algos):
            vals = [type_vals[t].get(name, {}).get(metric, 0.0) for t in JERLOV_TYPES]
            bars = ax.bar(x + offsets[i], vals, width=bar_w,
                          label=name, color=PALETTE[i % len(PALETTE)],
                          edgecolor="white", linewidth=0.8, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(JERLOV_TYPES, fontsize=11)
        ax.set_xlabel("Jerlov Water Type", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=10, framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--", zorder=0)

    fig.suptitle("Per-Jerlov-Type Coverage Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")


def plot_jerlov_curves(algo_episodes: dict[str, list[dict]], hist_key: str,
                       ylabel: str, title: str, out_path: Path):
    """수종별 6-subplot coverage curve (각 subplot에 알고리즘별 step-by-step 커브)."""
    type_hists: dict[str, dict[str, np.ndarray]] = {t: {} for t in JERLOV_TYPES}
    for name, eps in algo_episodes.items():
        for e in eps:
            t = e.get("jerlov_type")
            if t in type_hists:
                type_hists[t][name] = e[hist_key]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, t in zip(axes.flatten(), JERLOV_TYPES):
        for i, (name, eps) in enumerate(algo_episodes.items()):
            if not eps:
                continue
            hist = type_hists[t].get(name)
            if hist is None:
                continue
            steps = np.arange(1, len(hist) + 1)
            ax.plot(steps, hist, label=name,
                    color=PALETTE[i % len(PALETTE)],
                    linestyle=LINESTYLES[i % len(LINESTYLES)],
                    linewidth=2.0)
        ax.set_title(f"Jerlov  {t}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(1, MAX_STEPS)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, alpha=0.25, linestyle="--")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True,
                        help="name:path 형식. 예) UW_NBV_2:./recon_output/UW_NBV_2_327k")
    parser.add_argument("--out_dir", type=str, default="./analysis")
    parser.add_argument("--jerlov_eval", action="store_true",
                        help="에피소드 순서를 Jerlov 수종(IB→5C)으로 해석해 수종별 플롯 추가 생성")
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
        eps = load_episodes(d, jerlov_eval=args.jerlov_eval)
        print(f"[load] {name}: {len(eps)} episodes  ← {d}")
        algo_episodes[name] = eps
        stats[name] = compute_stats(eps)

    # 테이블 출력 + 저장
    print_table(stats)
    save_table_csv(stats, out_dir / "comparison_table.csv")

    # 커브 플롯 (전체 에피소드 평균)
    plot_curves(algo_episodes, "q_hist",   "coverage_q",   "Quality Coverage over Steps (mean±std)",
                out_dir / "coverage_q_curve.png")
    plot_curves(algo_episodes, "bin_hist", "coverage_bin", "Binary Coverage over Steps (mean±std)",
                out_dir / "coverage_bin_curve.png")

    # bar chart (전체 평균)
    plot_bar_chart(stats, out_dir / "bar_chart.png")

    # Jerlov 수종별 플롯
    if args.jerlov_eval:
        plot_jerlov_bar(algo_episodes, out_dir / "jerlov_bar.png")
        plot_jerlov_curves(algo_episodes, "q_hist",
                           "coverage_q",
                           "coverage_q per Jerlov Type",
                           out_dir / "jerlov_cov_q_curves.png")
        plot_jerlov_curves(algo_episodes, "bin_hist",
                           "coverage_bin",
                           "coverage_bin per Jerlov Type",
                           out_dir / "jerlov_cov_bin_curves.png")


if __name__ == "__main__":
    main()
