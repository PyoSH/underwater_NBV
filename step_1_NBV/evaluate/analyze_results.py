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
import math
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


# colorblind-friendly palette (Okabe-Ito) — 미지정 알고리즘 fallback용
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]
LINEWIDTH = 2.4  # 전부 동일 두께 — 색으로만 구분 (두께 차등은 인위적 위계로 보일 수 있어 배제)

# 알고리즘별 고정 색상 (Okabe-Ito 색맹 안전 팔레트에서 5개를 최대한 떨어뜨려 선택).
# 두께는 전부 LINEWIDTH로 동일.
RENAME = {"UW_NBV_5": "Proposed", "UW_NBV_DR_2": "Proposed+DR"}
ALGO_STYLE = {
    "Manual":      dict(color="#999999", linewidth=LINEWIDTH),  # 회색
    "ScanRL":      dict(color="#D55E00", linewidth=LINEWIDTH),  # 버밀리언
    "GenNBV":      dict(color="#0072B2", linewidth=LINEWIDTH),  # 파랑
    "Proposed":    dict(color="#009E73", linewidth=LINEWIDTH),  # 청록
    "Proposed+DR": dict(color="#CC79A7", linewidth=LINEWIDTH),  # 자홍
}


def get_style(name: str, idx: int) -> dict:
    """이름을 RENAME으로 정규화 후 ALGO_STYLE 조회, 없으면 PALETTE로 fallback (두께는 동일)."""
    canon = RENAME.get(name, name)
    if canon in ALGO_STYLE:
        return ALGO_STYLE[canon]
    return dict(color=PALETTE[idx % len(PALETTE)], linewidth=LINEWIDTH)


def thin_on_top_zorder(linewidth: float) -> float:
    """굵은 선이 얇은 선을 덮어버리지 않도록 — 얇을수록 zorder를 높여 위로 그림."""
    return 30.0 - linewidth


def plot_curves(algo_episodes: dict[str, list[dict]], hist_key: str,
                ylabel: str, title: str, out_path: Path, show_variance: bool = True):
    fig, ax = plt.subplots(figsize=(10, 5))

    label_targets = []  # (y_end, name, color) — 끝점 직접 라벨링용

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
        style = get_style(name, i)
        c, lw = style["color"], style["linewidth"]
        zo = thin_on_top_zorder(lw)
        ax.plot(steps, mu, label=name, color=c, linewidth=lw, zorder=zo)
        if show_variance:
            sigma = padded.std(axis=0)
            ax.fill_between(steps, mu - sigma, mu + sigma, alpha=0.15, color=c,
                            zorder=zo - 0.5)
        label_targets.append([mu[-1], name, c])

    # ── 끝점 직접 라벨링: 값이 가까우면 겹치지 않게 수직으로 살짝 벌림 ──────────
    label_targets.sort(key=lambda t: t[0])
    min_gap = 0.035
    for k in range(1, len(label_targets)):
        if label_targets[k][0] - label_targets[k - 1][0] < min_gap:
            label_targets[k][0] = label_targets[k - 1][0] + min_gap
    x_end = MAX_STEPS
    for y, name, c in label_targets:
        ax.annotate(name, xy=(x_end, y), xytext=(8, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=10, fontweight="bold", color=c)

    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(1, MAX_STEPS * 1.22)
    ax.set_xticks([t for t in ax.get_xticks() if t <= MAX_STEPS])
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[save] {out_path}")


def plot_bar_chart(stats: dict[str, dict | None], out_path: Path):
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
        colors = [get_style(n, i)["color"] for i, n in enumerate(names)]
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


def plot_jerlov_bar(algo_episodes: dict[str, list[dict]], out_path: Path,
                    types: list[str] | None = None):
    """수종별 grouped bar chart (cov_q / cov_bin 두 subplot)."""
    types = types or JERLOV_TYPES
    algos = [n for n, eps in algo_episodes.items() if eps]
    n_algos = len(algos)
    x = np.arange(len(types))
    bar_w = 0.8 / n_algos
    offsets = np.linspace(-(n_algos - 1) / 2 * bar_w, (n_algos - 1) / 2 * bar_w, n_algos)

    # type → {algo → value} 빌드
    type_vals: dict[str, dict[str, dict]] = {t: {} for t in types}
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
            vals = [type_vals[t].get(name, {}).get(metric, 0.0) for t in types]
            bars = ax.bar(x + offsets[i], vals, width=bar_w,
                          label=RENAME.get(name, name), color=get_style(name, i)["color"],
                          edgecolor="white", linewidth=0.8, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.012,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(types, fontsize=11)
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
                       ylabel: str, title: str, out_path: Path,
                       types: list[str] | None = None):
    """수종별 subplot coverage curve (각 subplot에 알고리즘별 step-by-step 커브).

    types: 표시할 수종 부분집합 (예: ["IB","II","III"]). None이면 6종 전체.
    """
    types = types or JERLOV_TYPES
    type_hists: dict[str, dict[str, np.ndarray]] = {t: {} for t in types}
    for name, eps in algo_episodes.items():
        for e in eps:
            t = e.get("jerlov_type")
            if t in type_hists:
                type_hists[t][name] = e[hist_key]

    n = len(types)
    rows = 2 if n > 3 else 1
    cols = math.ceil(n / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True, squeeze=False)
    for ax, t in zip(axes.flatten(), types):
        local_max_len = 1
        for i, (name, eps) in enumerate(algo_episodes.items()):
            if not eps:
                continue
            hist = type_hists[t].get(name)
            if hist is None:
                continue
            steps = np.arange(1, len(hist) + 1)
            local_max_len = max(local_max_len, len(hist))
            style = get_style(name, i)
            ax.plot(steps, hist, label=RENAME.get(name, name),
                    color=style["color"], linewidth=style["linewidth"],
                    zorder=thin_on_top_zorder(style["linewidth"]))
        ax.set_title(f"Jerlov  {t}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(1, local_max_len * 1.05)   # 실제 도달한 step까지만 — 패널을 꽉 채움
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
    parser.add_argument("--no_variance", action="store_true",
                        help="coverage curve에서 std 음영(fill_between)을 빼고 평균선만 그림")
    parser.add_argument("--jerlov_subset", nargs="+", default=None, choices=JERLOV_TYPES,
                        help="jerlov_bar/jerlov_curves에 표시할 수종 부분집합 (예: IB II III). "
                             "미지정시 6종 전체")
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
    show_var = not args.no_variance
    title_suffix = "(mean±std)" if show_var else "(mean)"
    plot_curves(algo_episodes, "q_hist",   "coverage_q",   f"Quality Coverage over Steps {title_suffix}",
                out_dir / "coverage_q_curve.png", show_variance=show_var)
    plot_curves(algo_episodes, "bin_hist", "coverage_bin", f"Binary Coverage over Steps {title_suffix}",
                out_dir / "coverage_bin_curve.png", show_variance=show_var)

    # bar chart (전체 평균)
    plot_bar_chart(stats, out_dir / "bar_chart.png")

    # Jerlov 수종별 플롯
    if args.jerlov_eval:
        plot_jerlov_bar(algo_episodes, out_dir / "jerlov_bar.png", types=args.jerlov_subset)
        plot_jerlov_curves(algo_episodes, "q_hist",
                           "coverage_q",
                           "coverage_q per Jerlov Type",
                           out_dir / "jerlov_cov_q_curves.png", types=args.jerlov_subset)
        plot_jerlov_curves(algo_episodes, "bin_hist",
                           "coverage_bin",
                           "coverage_bin per Jerlov Type",
                           out_dir / "jerlov_cov_bin_curves.png", types=args.jerlov_subset)


if __name__ == "__main__":
    main()
