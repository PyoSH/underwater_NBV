"""
스러스터 채널 매핑 진단 — 정책/관측 완전히 배제, action 파이프라인만 단독 검증
================================================================================
목적: RC_CHANNELS_OVERRIDE의 채널 1~8이 우리 `brov2_heavy.yaml`의 thruster_1~8과
같은 물리적 위치/방향을 가리키는지 확인한다. SITL에서 "정책 적용 시 마구
요동친다"는 증상이 (a) obs_builder의 NED→Z-up 변환 버그인지, (b) 이 채널
매핑 자체가 어긋나 있는지(우리 YAML의 T_N ≠ ArduSub SERVO_N 실제 물리 위치)를
가르기 위해, 정책/관측을 전부 빼고 **채널 하나씩 직접 PWM을 줘서 실측 반응**을
`build_allocation_matrix()`가 예측하는 방향과 비교한다.

이 스크립트는 obs_builder.py/guidance_standalone.py의 NED→Z-up 변환(Q_M)을
전혀 쓰지 않는다 — RC_CHANNELS_OVERRIDE도, 비교 대상 예측(B 행렬)도 전부
SNAME/NED로 일관되게 맞춰서, "관측 변환 버그"와 "채널 매핑 버그"를 서로
섞이지 않게 분리해서 검증한다.

사용법
------
python.sh deploy/diag_thruster_map.py --connection udpin:0.0.0.0:14550 --arm
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # deploy의 부모 — deploy 패키지 하나만 있으면 됨(vendoring)
from deploy.real_robot_interface import RealRobotInterface
from deploy.vendor.thruster import build_allocation_matrix
from deploy.vendor.params import load_brov2_yaml, thruster_pos_dir_ned

_PWM_TEST = 1700     # 중립(1500) 대비 뚜렷한 반응이 나올 크기
_BURST_S  = 1.2
_SETTLE_S = 2.0


def _dominant_axis(vec6: torch.Tensor) -> str:
    labels = ["Fx(surge)", "Fy(sway)", "Fz(heave)", "Tx(roll)", "Ty(pitch)", "Tz(yaw)"]
    i = int(vec6.abs().argmax())
    sign = "+" if vec6[i] > 0 else "-"
    return f"{sign}{labels[i]}"


def main() -> None:
    parser = argparse.ArgumentParser(description="스러스터 채널 매핑 실측 진단")
    parser.add_argument("--connection", type=str, default="udpin:0.0.0.0:14550")
    parser.add_argument("--arm", action="store_true", default=True)
    args = parser.parse_args()

    yaml_params = load_brov2_yaml()
    thr_pos, thr_dir = thruster_pos_dir_ned(yaml_params)
    thr_pos = torch.tensor(thr_pos)
    thr_dir = torch.tensor(thr_dir)
    B = build_allocation_matrix(thr_pos, thr_dir)   # (6,8) — 열 i = 채널(i+1)의 예측 [Fx,Fy,Fz,Tx,Ty,Tz] 방향

    interface = RealRobotInterface(args.connection)
    interface.connect()
    interface.enable_passthrough()
    if args.arm:
        print("[diag] arm 시도...")
        if not interface.arm():
            print("[diag] arm 실패 — 종료")
            interface.close()
            return
        print("[diag] armed")

    print("[diag] 첫 텔레메트리 대기...")
    snap = None
    while snap is None:
        time.sleep(0.05)
        snap = interface.snapshot()

    try:
        for ch in range(1, 9):
            time.sleep(_SETTLE_S)
            snap0 = interface.snapshot()
            vel0 = snap0["vel_ned"].clone()
            rate0 = snap0["body_rates_ned"].clone()

            pwm = torch.zeros(8)
            pwm[ch - 1] = (_PWM_TEST - 1500) / 400.0   # send_pwm은 [-1,1] 입력을 받음
            interface.send_pwm(pwm)
            time.sleep(_BURST_S)

            snap1 = interface.snapshot()
            vel1 = snap1["vel_ned"].clone()
            rate1 = snap1["body_rates_ned"].clone()

            interface.send_pwm(torch.zeros(8))   # neutral

            dvel = (vel1 - vel0) / _BURST_S     # 대략적 선가속도(월드 NED) — burst 동안 자세 변화가 작다고 가정
            drate = (rate1 - rate0) / _BURST_S  # 대략적 각가속도(body NED)
            observed6 = torch.cat([dvel, drate])
            predicted6 = B[:, ch - 1]

            print(f"\n[채널 {ch}]")
            print(f"  예측 (YAML thruster_{ch}, B 열)  : {predicted6.tolist()} → 지배축 {_dominant_axis(predicted6)}")
            print(f"  실측 (Δvel_ned, Δrate_ned)      : {observed6.tolist()} → 지배축 {_dominant_axis(observed6)}")

    except KeyboardInterrupt:
        print("\n[diag] 중단")
    finally:
        interface.disarm()
        interface.close()
        print("[diag] 종료")


if __name__ == "__main__":
    main()
