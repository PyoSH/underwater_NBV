#!/usr/bin/env python3
"""Set a deterministic no-GPS EKF origin and require its MAVLink echo."""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MAVLINK20", "1")

from pymavlink import mavutil


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", default="udpin:0.0.0.0:14554")
    parser.add_argument("--latitude-e7", type=int, default=559954153)
    parser.add_argument("--longitude-e7", type=int, default=-33010225)
    parser.add_argument("--altitude-mm", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    args = parser.parse_args()

    if not args.connection.startswith("udpin:"):
        raise SystemExit("origin helper only accepts a dedicated udpin endpoint")
    master = mavutil.mavlink_connection(
        args.connection, source_system=255, source_component=190
    )
    try:
        if master.wait_heartbeat(timeout=args.timeout_s) is None:
            raise RuntimeError("ArduSub heartbeat unavailable for origin setup")
        deadline = time.monotonic() + args.timeout_s
        next_send = 0.0
        origin = None
        while time.monotonic() < deadline and origin is None:
            now = time.monotonic()
            if now >= next_send:
                master.mav.set_gps_global_origin_send(
                    master.target_system,
                    args.latitude_e7,
                    args.longitude_e7,
                    args.altitude_mm,
                )
                next_send = now + 1.0
            origin = master.recv_match(
                type="GPS_GLOBAL_ORIGIN", blocking=True, timeout=0.25
            )
        if origin is None:
            raise RuntimeError("EKF origin was not acknowledged")
        print(origin)
    finally:
        master.close()


if __name__ == "__main__":
    main()
