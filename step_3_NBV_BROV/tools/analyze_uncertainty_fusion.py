"""TRIDENT uncertainty가 depth 오차를 예측하는가, 그리고 가중 융합이 이득인가.

왜 이 실험인가
--------------
Stage 2/3의 설계 전환 후보는 "voxel 품질을 `exp(-μd)`로 **모델링**하지 말고
**측정**하자"는 것이다. 측정값 후보가 TRIDENT의 3번째 출력인 uncertainty다.
배포 시 μ 추정이 불필요해지고(depth를 내는 모델이 함께 내놓으므로),
depth가 틀린 곳은 TSDF 융합에서 낮은 가중치를 받게 된다.

그런데 이 전환은 **uncertainty가 실제 오차를 예측할 때만** 의미가 있다.
예측력이 없으면 가중치가 잡음일 뿐이므로, 그것부터 확인한다.

무엇을 재는가
-------------
1. **예측력** — uncertainty 십분위별 실제 |pred-GT| 오차. 단조 증가해야 한다.
   순위상관(Spearman)도 함께 본다.

2. **가중 융합의 이론적 이득** — 같은 3D 점을 N번 관측해 평균낼 때,
   역분산 가중 추정량의 분산은 `1/Σ(1/σᵢ²)`, 균등 평균은 `(1/N²)Σσᵢ²`이다.
   비율은 관측된 σ 분포만으로 계산된다. 우리 표본에는 동일 3D 점의 다중 관측
   대응이 없으므로(장면·시점이 모두 다름) 실측 대신 이 해석해로 상한을
   추정한다 — 대응을 만들려면 실제 TSDF 적분이 필요하고, 그건 이 실험의
   결과에 따라 할지 말지가 정해진다.

   주의: 이 계산은 σ가 **참오차의 척도로 정확할 때**의 이득이다. 1번에서
   예측력이 약하면 실제 이득은 이보다 작다.

사용법
------
/isaac-sim/python.sh -u analyze_uncertainty_fusion.py --pairs /tmp/trident_pairs_fixed.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="TRIDENT uncertainty 예측력/융합 이득 분석")
parser.add_argument("--pairs", type=str, required=True)
parser.add_argument("--ckpt", type=str, default="ckpt/TRIDENT_three_task.pth")
parser.add_argument("--cfg", type=str, default="local_configs/cfg/TRIDENT_three_task.py")
parser.add_argument("--depth_out", type=int, default=0)
parser.add_argument("--unc_out", type=int, default=2)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--max_images", type=int, default=200)
args = parser.parse_args()

# 저장소 dataloader(joint_de_eh_dataloader.py:349)의 수중 통계.
# 정규화를 빼면 결과가 무의미해진다(2026-09-02에 실제로 그렇게 틀렸다).
_MEAN = [0.13553666, 0.41034216, 0.34636855]
_STD = [0.04927989, 0.10722694, 0.10722694]


def load_cfg(path):
    ns = {}
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), ns)
    return ns


def main() -> int:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    d = np.load(args.pairs)
    rgb, gt = d["rgb"][: args.max_images], d["depth"][: args.max_images]

    cfg = load_cfg(args.cfg)
    from core.models.network_builder import MODEL_BUILDER
    model = MODEL_BUILDER.build(dict(cfg["basic_cfg"]["model_cfg"])).to(device).eval()
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd.get("model", sd.get("state_dict", sd)), strict=False)
    H, W = cfg["image_size"]["input_height"], cfg["image_size"]["input_width"]

    mt = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    st = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    preds, uncs = [], []
    with torch.no_grad():
        for i in range(0, len(rgb), args.batch):
            x = torch.from_numpy(rgb[i : i + args.batch]).to(device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            y = model((x - mt) / st)
            outs = y if isinstance(y, (list, tuple)) else [y]
            for src, dst in ((args.depth_out, preds), (args.unc_out, uncs)):
                o = outs[src]
                if o.ndim == 4:
                    o = o[:, 0]
                o = F.interpolate(o.unsqueeze(1), size=gt.shape[1:],
                                  mode="bilinear", align_corners=False).squeeze(1)
                dst.append(o.cpu().numpy())
    pred, unc = np.concatenate(preds), np.concatenate(uncs)

    m = (np.isfinite(gt) & (gt > 0.1) & (gt < 20.0)
         & np.isfinite(pred) & (pred > 1e-6) & np.isfinite(unc))
    err = np.abs(pred - gt)
    e, u, g = err[m], unc[m], gt[m]
    print(f"[unc] 유효 픽셀 {len(e):,}")
    print(f"[unc] uncertainty 분포: min {u.min():.5f} 중앙 {np.median(u):.5f} "
          f"평균 {u.mean():.5f} max {u.max():.5f}")
    nz = (u > 1e-6).mean()
    print(f"[unc] 0이 아닌 비율 {nz*100:.1f}%")
    if nz < 0.05:
        print("[unc] ⚠ 거의 전부 0이다 — 가중치로 쓸 정보가 없다는 뜻이다.")

    # ── 1. 예측력 ───────────────────────────────────────────────────────────
    print(f"\n[1] uncertainty 십분위별 실제 오차 (예측력이 있으면 단조 증가)")
    qs = np.quantile(u, np.linspace(0, 1, 11))
    print(f"{'십분위':>7}{'unc 범위':>22}{'평균 |오차|':>12}{'AbsRel':>9}")
    print("-" * 52)
    prev = None
    monotone = True
    for k in range(10):
        sel = (u >= qs[k]) & (u <= qs[k + 1] if k == 9 else u < qs[k + 1])
        if sel.sum() < 100:
            continue
        me, ar = e[sel].mean(), (e[sel] / g[sel]).mean()
        print(f"{k+1:>7}{f'{qs[k]:.5f}~{qs[k+1]:.5f}':>22}{me:>12.3f}{ar:>9.3f}")
        if prev is not None and me < prev - 1e-9:
            monotone = False
        prev = me
    # 순위상관 (표본이 크므로 하위표집)
    # 전체 ρ는 97.6%가 0으로 동률이라 의미가 희석된다. 0이 아닌 부분에서
    # **등급이 매겨지는지**를 따로 본다 — 연속 가중치로 쓸 수 있는지가 여기서 갈린다.
    idx = np.random.default_rng(0).choice(len(e), size=min(200_000, len(e)), replace=False)
    ru = np.argsort(np.argsort(u[idx])).astype(np.float64)
    re = np.argsort(np.argsort(e[idx])).astype(np.float64)
    rho = float(np.corrcoef(ru, re)[0, 1])
    print(f"\n   Spearman 순위상관 ρ = {rho:+.3f}   (단조성 {'유지' if monotone else '깨짐'})")
    if rho < 0.1:
        print("   → 예측력이 없다. uncertainty 가중은 잡음을 곱하는 것과 같다.")
    elif rho < 0.3:
        print("   → 예측력이 약하다. 가중 이득이 이론치에 크게 못 미칠 것이다.")
    else:
        print("   → 예측력이 있다. 가중 융합이 의미를 가진다.")

    nzm = u > 1e-6
    if nzm.sum() > 1000:
        ii = np.random.default_rng(2).choice(int(nzm.sum()), size=min(200_000, int(nzm.sum())), replace=False)
        un, en = u[nzm][ii], e[nzm][ii]
        rho_nz = float(np.corrcoef(np.argsort(np.argsort(un)).astype(float),
                                   np.argsort(np.argsort(en)).astype(float))[0, 1])
        print(f"   0이 아닌 화소({nzm.sum():,}개)만: ρ = {rho_nz:+.3f}")
        print(f"     → {'등급이 있다 = 연속 가중치 가능' if rho_nz>0.3 else '등급이 약하다 = 이진 마스크로 쓰는 편이 낫다'}")

    # ── 2. 마스킹 효과 ──────────────────────────────────────────────────────
    # uncertainty가 97.6% 0인 **희소** 신호라 역분산 가중은 성립하지 않는다
    # (0을 어떤 값으로 클램프하느냐가 결과를 지배한다). 실제 쓰임새는
    # "이 화소는 믿지 마라"는 **배제 마스크**이므로 그 효과를 잰다.
    print(f"\n[2] 고-uncertainty 화소를 버리면 남은 화소의 정확도가 오르는가")
    print(f"{'임계값':>10}{'남는 비율':>10}{'AbsRel':>9}{'개선':>8}{'|오차| 평균':>12}")
    print("-" * 50)
    base_ar = float((e / g).mean()); base_e = float(e.mean())
    print(f"{'(전체)':>10}{'100.0%':>10}{base_ar:>9.3f}{'—':>8}{base_e:>12.3f}")
    for th in (0.25, 0.10, 0.05, 0.02, 0.01, 1e-6):
        keep = u <= th
        if keep.sum() < 1000:
            continue
        ar = float((e[keep] / g[keep]).mean())
        print(f"{th:>10.5f}{keep.mean()*100:>9.1f}%{ar:>9.3f}"
              f"{(base_ar-ar)/base_ar*100:>7.1f}%{e[keep].mean():>12.3f}")
    print("\n   버려서 얻는 정확도 개선이 작으면, uncertainty를 TSDF 가중에 써도")
    print("   coverage 품질이 크게 나아지지 않는다는 뜻이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
