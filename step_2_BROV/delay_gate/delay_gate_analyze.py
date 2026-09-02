#!/usr/bin/env python3
"""velocity_hold 덤프 → 지연 gate 지표.

gate_analyze.py(선례)의 정상상태 요약에 진동 지표를 더한다:
 - action PSD 지배 주파수 (2 Hz 대역 limit cycle 판별)
 - |omega| 중앙/p90
 - action 포화율 |a|>=0.99
 - Fz(heave 축) 명령 부호변화율
usage: delay_gate_analyze.py <dump.json> [...]
"""
import json, sys
import numpy as np

AX = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def sq(x):
    x = np.asarray(x, float)
    return x[:, 0, :] if x.ndim == 3 else x


def dominant_freq(sig, dt):
    """DC/트렌드 제거 후 PSD 지배 주파수와 그 대역 파워 비중."""
    n = len(sig)
    x = sig - sig.mean()
    # 선형 트렌드 제거 (정상상태 드리프트가 저주파에 실려 지배하지 않게)
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    w = np.hanning(n)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(n, dt)
    P = np.abs(X) ** 2
    P[0] = 0.0
    if P.sum() <= 0:
        return 0.0, 0.0, 0.0
    k = int(P.argmax())
    return float(f[k]), float(P[k] / P.sum()), float(np.sqrt(np.mean(x ** 2)))


def analyze(path, tail=0.5):
    d = json.load(open(path))
    t = np.asarray(d["t"], float)
    dt = float(t[1] - t[0])
    va, vd = sq(d["v_actual"]), sq(d["v_desired"])
    a = sq(d["action"])
    w = sq(d["omega_b"])
    eu = sq(d["euler_deg"])
    zv = sq(d["z_v"])
    fr, fl = sq(d["force_requested"]), sq(d["force_limited"])
    s = int(len(t) * (1 - tail))

    out = {"path": path, "steps": len(t), "dt": dt}
    des = np.linalg.norm(vd[s:], axis=-1).mean()
    out["v_des"] = vd[-1].tolist()
    out["track_pct"] = 100 * va[s:, 0].mean() / max(vd[s:, 0].mean(), 1e-9)
    out["u_mean"] = va[s:, 0].mean()
    out["u_std"] = va[s:, 0].std()
    out["speed_rmse"] = float(np.sqrt(np.mean(np.sum((va[s:] - vd[s:]) ** 2, axis=1))))
    wn = np.linalg.norm(w[s:], axis=-1)
    out["w_med"] = float(np.median(wn))
    out["w_p90"] = float(np.percentile(wn, 90))
    out["w_max"] = float(wn.max())
    out["sat_pct"] = 100 * float((np.abs(a[s:]) >= 0.99).mean())
    out["sat_any_pct"] = 100 * float((np.abs(a[s:]) >= 0.99).any(axis=-1).mean())
    # 축별 지배 주파수
    doms = []
    for i in range(6):
        f0, share, rms = dominant_freq(a[s:, i], dt)
        doms.append((AX[i], f0, share, rms))
    out["dom"] = doms
    # 가장 진동 에너지가 큰 축
    k = int(np.argmax([x[3] for x in doms]))
    out["dom_axis"] = doms[k][0]
    out["dom_f"] = doms[k][1]
    out["dom_rms"] = doms[k][3]
    # omega 지배 주파수 (실기 자이로 관측과 대응)
    wf = [dominant_freq(w[s:, i], dt) for i in range(3)]
    kw = int(np.argmax([x[2] for x in wf]))
    out["omega_dom_f"] = wf[kw][0]
    out["omega_dom_axis"] = ["p", "q", "r"][kw]
    # Fz(heave) 명령 부호변화율
    fz = a[s:, 2]
    out["fz_signflip_hz"] = float((np.diff(np.sign(fz)) != 0).sum() / (len(fz) * dt))
    out["clamp_pct"] = 100 * float((np.abs(fr[s:] - fl[s:]).max(axis=-1) > 1e-6).mean())
    out["zv_final"] = zv[-1].tolist()
    out["pitch_mean"] = eu[s:, 1].mean()
    out["pitch_std"] = eu[s:, 1].std()
    return out


hdr = (f"{'file':>26} {'track%':>7} {'u_std':>7} {'RMSE':>7} {'|w|med':>7} {'|w|p90':>7} "
       f"{'sat%':>6} {'dom_ax':>7} {'dom_f':>6} {'a_rms':>6} {'wf':>6} {'Fzflip':>7} {'clamp%':>7}")
print(hdr)
print("-" * len(hdr))
rows = []
for p in sys.argv[1:]:
    r = analyze(p)
    rows.append(r)
    print(f"{p.split('/')[-1]:>26} {r['track_pct']:7.1f} {r['u_std']:7.4f} {r['speed_rmse']:7.4f} "
          f"{r['w_med']:7.4f} {r['w_p90']:7.4f} {r['sat_any_pct']:6.1f} {r['dom_axis']:>7} "
          f"{r['dom_f']:6.2f} {r['dom_rms']:6.3f} {r['omega_dom_f']:6.2f} {r['fz_signflip_hz']:7.2f} {r['clamp_pct']:7.1f}")

print()
for r in rows:
    print(f"--- {r['path'].split('/')[-1]}  (축별 action 지배주파수 / 대역비중 / rms)")
    print("    " + "  ".join(f"{ax}:{f0:.2f}Hz({100*sh:.0f}%,{rms:.3f})" for ax, f0, sh, rms in r["dom"]))
