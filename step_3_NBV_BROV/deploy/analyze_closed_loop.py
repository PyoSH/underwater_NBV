"""폐루프 CSV(nbv_policy_node) 요약 — 도착 정확도·홉 시간·coverage 곡선.

용례: python3 analyze_closed_loop.py <run_dir>/closed_loop/closed_loop_random.csv [...]
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


def load(path: Path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if k not in ("status", "mission", "projected") else v
            except ValueError:
                pass
    return rows


def q(vals, p):
    vals = sorted(v for v in vals if isinstance(v, float) and not math.isnan(v))
    if not vals:
        return float("nan")
    i = min(len(vals) - 1, max(0, int(round(p * (len(vals) - 1)))))
    return vals[i]


def main(paths):
    for path in paths:
        rows = load(Path(path))
        ok = [r for r in rows if r["status"] == "ok"]
        print(f"== {path}")
        print(f"  hops: {len(rows)}  ok {len(ok)}  "
              f"other: {sorted(set(r['status'] for r in rows if r['status'] != 'ok'))}")
        if ok:
            pe = [r["pos_err_m"] for r in ok]; ae = [r["att_err_deg"] for r in ok]
            fs = [r["fly_s"] for r in ok]; hl = [r["hop_len_m"] for r in ok]
            print(f"  pos_err [m]   median {q(pe, .5):.3f}  p90 {q(pe, .9):.3f}  max {max(pe):.3f}")
            print(f"  att_err [deg] median {q(ae, .5):.1f}  p90 {q(ae, .9):.1f}  max {max(ae):.1f}")
            print(f"  fly_s         median {q(fs, .5):.1f}  max {max(fs):.1f}   hop_len median {q(hl, .5):.2f} m")
            print(f"  waypoints/hop {sorted(set(int(r['waypoints']) for r in ok))}")
        cov = [(int(r["decision"]), r["cov_bin"]) for r in rows if not math.isnan(r["cov_bin"])]
        if cov:
            print("  cov_bin: " + "  ".join(f"@{d} {c:.3f}" for d, c in cov))
        print("  per hop: " + " | ".join(
            f"{int(r['decision'])}:{r['status'][:12]} {r['pos_err_m']:.2f}m/{r['att_err_deg']:.0f}° {r['fly_s']:.0f}s"
            for r in rows))


if __name__ == "__main__":
    main(sys.argv[1:])
