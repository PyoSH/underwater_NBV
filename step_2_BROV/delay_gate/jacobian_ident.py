#!/usr/bin/env python3
"""정책 국소 선형화 — 순항 관측점에서 autograd Jacobian.

DELAY_TRAINING_PLAN.md 단계 2. scratchpad `policy_ident.py` 방식을 그대로
쓰되 28-D(설계 B) 이력 성분까지 다룬다.

- K_p  = ∂a_i/∂v_e_i        (obs idx 4+i)   — 병진 이득. 판정 참조표:
                                              τ=80 ms 문턱 3.52, 60 ms 4.40,
                                              50 ms 5.01
- K_i  = ∂a_i/∂z_v_i        (obs idx 10+i)  — 적분 이득(위상원)
- K_h1 = ∂a_j/∂a_{t-1,j}    (obs idx 16+j)  — 이력 1 자기축 Jacobian (28-D 전용)
- K_h2 = ∂a_j/∂a_{t-2,j}    (obs idx 22+j)  — 이력 2 자기축 Jacobian (28-D 전용)

재생 검증(replay)에는 runtime clip ±1 을 반드시 반영해야 한다 — 기록된
action 은 잘린 값이라 clip 없이 비교하면 실패한다(선례에서 실제로 겪음).

usage: jacobian_ident.py <policy.pt> <isaac_dump.json> [...]
"""
import json
import sys

import numpy as np
import torch

WRENCH_SCALE = np.array([85.0, 85.0, 120.0, 26.0, 14.0, 22.0])
M_EFF_AXIS = np.array([14.635 + 6.36, 14.635 + 7.12, 14.635 + 13.5])
AX6 = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
KP_THRESHOLD = {80: 3.52, 60: 4.40, 50: 5.01}


def load_isaac(path):
    d = json.load(open(path))

    def sq(x):
        x = np.asarray(x, float)
        return x[:, 0, :] if x.ndim == 3 else x

    O, A = sq(d["obs"]), sq(d["action"])
    n = min(len(O), len(A))
    s = n // 6                      # 초기 과도 제외
    return O[s:n], A[s:n]


def _jacobian_at(policy, x_np):
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    J = torch.autograd.functional.jacobian(
        lambda z: policy(z).squeeze(0), x, vectorize=True
    )
    return J.squeeze().detach().numpy()


def anchor_report(policy, dim):
    """모든 정책에서 **동일한** 관측점의 이득 — 순항 관측 분포가 정책마다
    다른 것이 이득 비교를 오염시키지 않게 하는 대조점.

    q_e = 항등, v_e = 0, ω = 0, z_v = z_q = 0, (28-D 면) 이력 = 0.
    """
    x = np.zeros(dim)
    x[0] = 1.0                    # q_e = [1,0,0,0]
    J = _jacobian_at(policy, x)
    print("\n--- 공통 anchor 관측점 (q_e=I, v_e=0, ω=0, z=0, hist=0)")
    print(f"    {'축':>6} {'K_p':>9} {'K_i(z_v)':>9} "
          + (f"{'∂a/∂a_(t-1)':>12} {'∂a/∂a_(t-2)':>12}" if dim >= 28 else ""))
    for a in range(6):
        kp = J[a, 4 + a] if a < 3 else float("nan")
        ki = J[a, 10 + a] if a < 3 else float("nan")
        extra = ""
        if dim >= 28:
            extra = f" {J[a, 16 + a]:12.3f} {J[a, 22 + a]:12.3f}"
        print(f"    {AX6[a]:>6} {kp:9.3f} {ki:9.3f}{extra}")
    if dim >= 28:
        print(f"    이력 블록 평균 |J| = {np.abs(J[:, 16:28]).mean():.3f}   "
              f"v_e 블록 평균 |J| = {np.abs(J[:, 4:7]).mean():.3f}")
    return J


def main():
    policy = torch.jit.load(sys.argv[1], map_location="cpu").eval()
    print("=" * 78)
    print(f"policy: {sys.argv[1]}")
    print("=" * 78)
    probe_dim = 16
    for path in sys.argv[2:]:
        probe_dim = len(json.load(open(path))["obs"][0])
        break
    anchor_report(policy, probe_dim)
    for path in sys.argv[2:]:
        O, A = load_isaac(path)
        dim = O.shape[1]
        with torch.no_grad():
            raw = policy(torch.tensor(O, dtype=torch.float32)).numpy()
        replayed = np.clip(raw, -1.0, 1.0)
        err = np.abs(replayed - A).max()
        clipped = 100 * np.mean(np.abs(raw) > 1.0)

        X = torch.tensor(O, dtype=torch.float32)
        step = max(1, len(X) // 400)
        Js = []
        for i in range(0, len(X), step):
            x = X[i:i + 1].clone().requires_grad_(True)
            J = torch.autograd.functional.jacobian(
                lambda z: policy(z).squeeze(0), x, vectorize=True
            )
            Js.append(J.squeeze().detach().numpy())
        J = np.stack(Js)                                  # (N, 6, dim)

        print(f"\n--- {path.split('/')[-1]}  obs={dim}-D  n={len(O)}  "
              f"Jacobian 표본 {len(J)}")
        print(f"    재생 max|dA|={err:.2e}   raw clip 비율 {clipped:.1f}%")
        print(f"    {'축':>6} {'K_p 중앙':>9} {'K_p IQR':>16} {'K_i 중앙':>9} "
              f"{'80ms 문턱대비':>13}")
        for a in range(3):
            kp = J[:, a, 4 + a]
            ki = J[:, a, 10 + a]
            kpm = float(np.median(kp))
            print(f"    {AX6[a]:>6} {kpm:9.3f} "
                  f"[{np.percentile(kp,25):6.2f},{np.percentile(kp,75):6.2f}] "
                  f"{np.median(ki):9.3f} {100*kpm/KP_THRESHOLD[80]:12.0f}%")
        if dim >= 28:
            print(f"    {'축':>6} {'∂a/∂a_(t-1)':>12} {'IQR':>18} "
                  f"{'∂a/∂a_(t-2)':>12} {'IQR':>18}")
            for a in range(6):
                h1 = J[:, a, 16 + a]
                h2 = J[:, a, 22 + a]
                print(f"    {AX6[a]:>6} {np.median(h1):12.3f} "
                      f"[{np.percentile(h1,25):7.3f},{np.percentile(h1,75):7.3f}] "
                      f"{np.median(h2):12.3f} "
                      f"[{np.percentile(h2,25):7.3f},{np.percentile(h2,75):7.3f}]")
            # 이력 블록 전체 민감도 (교차항 포함) — v_e 블록과 크기 비교
            hist_norm = np.abs(J[:, :, 16:28]).mean()
            ve_norm = np.abs(J[:, :, 4:7]).mean()
            print(f"    이력 블록 평균 |J| = {hist_norm:.3f}   "
                  f"v_e 블록 평균 |J| = {ve_norm:.3f}   비 = {hist_norm/ve_norm:.2f}")


if __name__ == "__main__":
    main()
