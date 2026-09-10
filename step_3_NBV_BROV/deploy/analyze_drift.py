"""DVL 정지 후 INS 표류 분석 — 미션 길이 상한을 정하는 측정. 2026-09-10.

입력: run_nbv_sitl.sh 가 남긴 rosbag.
  /brov/sim/gazebo_odometry_raw  진실 pose (gz world ENU)
  /brov/sim/ekf_local_ned        ArduSub EKF3 (NED, FRD)
  /brov/stage2/dvl_valid         DVL 주입 유효 여부

좌표: EKF 는 NED, Gazebo 는 ENU 다. 변환은 **여기서 한 번만** 한다
  ENU(x=E, y=N, z=U)  ->  NED(N=y, E=x, D=-z)
EKF 원점은 start 시각의 진실 pose 로 잡는다(SET_GPS_GLOBAL_ORIGIN 이 그 시점 기준).
"""
from __future__ import annotations

import argparse
import glob
import sqlite3
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_topic(db: str, topic: str):
    con = sqlite3.connect(db)
    tid, tname = None, None
    for i, n, t in con.execute("select id, name, type from topics"):
        if n == topic:
            tid, tname = i, t
    if tid is None:
        return [], []
    cls = get_message(tname)
    ts, msgs = [], []
    for t, d in con.execute(
            "select timestamp, data from messages where topic_id=? order by timestamp", (tid,)):
        ts.append(t * 1e-9)
        msgs.append(deserialize_message(d, cls))
    return ts, msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    db = glob.glob(str(args.run_dir / "bag" / "*.db3"))
    if not db:
        raise SystemExit(f"bag 없음: {args.run_dir}")
    db = db[0]

    # EKF 가 수평 해를 포기하는 순간이 이 측정의 본론이다 (2026-09-10 발견):
    # ArduSub EKF3 는 외부항법(VISO)이 끊기면 **추측항법을 하지 않고** 약 7 초 뒤
    # CONST_POS_MODE 로 들어가 위치를 얼리고 LOCAL_POSITION_NED 발행을 멈춘다.
    st_t, st = read_topic(db, "/brov/sim/ekf_status")
    gt_t, gt = read_topic(db, "/brov/sim/gazebo_odometry_raw")
    ek_t, ek = read_topic(db, "/brov/sim/ekf_local_ned")
    dv_t, dv = read_topic(db, "/brov/stage2/dvl_valid")
    if not gt_t or not ek_t:
        raise SystemExit("GT 또는 EKF 토픽이 비어 있다")

    gt_t = np.array(gt_t); ek_t = np.array(ek_t)
    gt_enu = np.array([[m.pose.pose.position.x, m.pose.pose.position.y,
                        m.pose.pose.position.z] for m in gt])
    ek_ned = np.array([[m.pose.pose.position.x, m.pose.pose.position.y,
                        m.pose.pose.position.z] for m in ek])

    # DVL 이 마지막으로 유효했던 시각
    valid_t = [t for t, m in zip(dv_t, dv) if m.data]
    t_drop = valid_t[-1] if valid_t else ek_t[0]

    # 진실 pose 를 EKF 시각에 보간하고 NED 로 변환
    gt_i = np.stack([np.interp(ek_t, gt_t, gt_enu[:, k]) for k in range(3)], axis=1)
    gt_ned = np.stack([gt_i[:, 1], gt_i[:, 0], -gt_i[:, 2]], axis=1)
    # 원점 정렬: EKF 시작 시점의 진실값을 0 으로
    gt_ned = gt_ned - gt_ned[0]
    ek_ned = ek_ned - ek_ned[0]

    err = ek_ned - gt_ned
    horiz = np.linalg.norm(err[:, :2], axis=1)
    vert = np.abs(err[:, 2])
    rel = ek_t - ek_t[0]
    drop_rel = t_drop - ek_t[0]

    import re as _re
    BITS = [(1, "ATTITUDE"), (2, "VEL_HORIZ"), (4, "VEL_VERT"), (8, "POS_HORIZ_REL"),
            (16, "POS_HORIZ_ABS"), (32, "POS_VERT_ABS"), (64, "POS_VERT_AGL"),
            (128, "CONST_POS_MODE"), (256, "PRED_POS_HORIZ_REL"),
            (512, "PRED_POS_HORIZ_ABS"), (1024, "UNINITIALIZED")]
    flag_events, prev_f = [], None
    for t, m in zip(st_t, st):
        f = int(_re.search(r"flags=(\d+)", m.data).group(1))
        if f != prev_f:
            flag_events.append((t, f))
            prev_f = f

    print(f"run: {args.run_dir.name}")
    print(f"  EKF 샘플 {len(ek_t)}개, {rel[-1]:.1f} s")
    print(f"  DVL 마지막 유효 t={drop_rel:.1f} s  (이후 {rel[-1]-drop_rel:.1f} s 가 INS 단독)")
    print("\n  EKF flags 전이:")
    for t, f in flag_events:
        names = "|".join(n for b, n in BITS if f & b)
        rel_t = t - ek_t[0]
        mark = ""
        if f & 128:
            mark = f"   <== CONST_POS_MODE (DVL 정지 +{t - t_drop:.1f} s)"
        print(f"    t={rel_t:7.1f}s  flags={f:5d}  {names}{mark}")

    print(f"\n  {'구간':>12} {'수평오차 m':>12} {'수직오차 m':>12}")
    for lo, hi, label in ((0, drop_rel, "DVL 유효"),
                          (drop_rel, drop_rel + 20, "정지 +0~20s"),
                          (drop_rel + 20, drop_rel + 60, "+20~60s"),
                          (drop_rel + 60, 1e9, "+60s~")):
        m = (rel >= lo) & (rel < hi)
        if m.sum() < 2:
            continue
        print(f"  {label:>12} {horiz[m].max():12.3f} {vert[m].max():12.3f}"
              f"   (중앙 {np.median(horiz[m]):.3f} / {np.median(vert[m]):.3f})")

    after = rel >= drop_rel
    if after.sum() > 10:
        dt = rel[after] - drop_rel
        h = horiz[after] - horiz[after][0]
        # 선형/2차 적합으로 표류 성격을 본다
        lin = np.polyfit(dt, h, 1)
        quad = np.polyfit(dt, h, 2)
        print(f"\n  INS 단독 수평 표류")
        print(f"    선형 적합 {lin[0]*100:+.2f} cm/s   (잔차 rms "
              f"{np.sqrt(np.mean((np.polyval(lin,dt)-h)**2))*100:.1f} cm)")
        print(f"    2차 적합 {quad[0]*100:+.3f} cm/s^2  (잔차 rms "
              f"{np.sqrt(np.mean((np.polyval(quad,dt)-h)**2))*100:.1f} cm)")
        for tol in (0.10, 0.20, 0.50):
            over = dt[h > tol]
            when = f"{over[0]:.0f} s" if len(over) else f">{dt[-1]:.0f} s (측정 구간 내 미도달)"
            print(f"    수평오차 {tol*100:.0f} cm 도달까지: {when}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
