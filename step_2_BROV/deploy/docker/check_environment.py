#!/usr/bin/env python3
"""Docker/ROS/PyTorch 배포 환경을 모터 출력 없이 점검한다."""

from __future__ import annotations

import importlib
import os
import platform
import socket
import sys
from pathlib import Path


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.machine()} / {platform.system()}")
    print(f"ROS_DISTRO: {os.environ.get('ROS_DISTRO', '<unset>')}")
    print(f"ROS_DOMAIN_ID: {os.environ.get('ROS_DOMAIN_ID', '<unset>')}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '<unset>')}")

    failed = False
    for name in (
        "torch", "yaml", "pymavlink", "rclpy", "std_msgs", "sensor_msgs",
        "geometry_msgs", "cv_bridge", "cv2", "gi",
    ):
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "available")
            print(f"[OK] {name}: {version}")
        except Exception as exc:
            failed = True
            print(f"[FAIL] {name}: {exc}")

    try:
        import cv2
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco is unavailable")
        Gst.init(None)
        print(f"[OK] camera stack: OpenCV {cv2.__version__} / {Gst.version_string()}")
    except Exception as exc:
        failed = True
        print(f"[FAIL] camera stack: {exc}")

    try:
        from deploy.vendor.params import load_brov2_yaml
        from deploy.vendor.thruster import build_allocation_matrix

        params = load_brov2_yaml()
        print(f"[OK] deploy import: {params['name']} / {params['thrusters']['num']} thrusters")
        _ = build_allocation_matrix
    except Exception as exc:
        failed = True
        print(f"[FAIL] deploy import: {exc}")

    try:
        from ament_index_python.packages import get_package_prefix

        for package in (
            "brov_base", "brov_control", "brov_perception", "brov_bringup"
        ):
            prefix = get_package_prefix(package)
            print(f"[OK] ROS package {package}: {prefix}")
    except Exception as exc:
        failed = True
        print(
            "[FAIL] BROV ROS package overlay: "
            f"{exc} (run 'make ros-build' and enter with 'make shell')"
        )

    policy_path = Path("/workspace/deploy/exported/policy.pt")
    if policy_path.is_file():
        try:
            import torch

            model = torch.jit.load(str(policy_path), map_location="cpu")
            model.eval()
            output = model(torch.zeros(1, 16))
            if tuple(output.shape) != (1, 6):
                raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")
            print(f"[OK] policy.pt: input (1,16) -> output {tuple(output.shape)}")
        except Exception as exc:
            failed = True
            print(f"[FAIL] policy.pt: {exc}")
    else:
        failed = True
        print(f"[FAIL] policy.pt not found: {policy_path}")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 14550))
        sock.close()
        print("[OK] UDP 14550 can be bound (no packet was transmitted)")
    except OSError as exc:
        failed = True
        print(f"[FAIL] UDP 14550 bind: {exc}")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 5600))
        sock.close()
        print("[OK] UDP 5600 can be bound (no packet was transmitted)")
    except OSError as exc:
        failed = True
        print(f"[FAIL] UDP 5600 bind: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
