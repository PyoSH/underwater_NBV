"""스윕용 world 생성: 운영 world 에서 ROV 를 빼고 정적 프로브 카메라를 넣는다.

ROV 를 빼는 이유: 스윕이 재는 것은 마커의 **광학·기하 한계**이지 제어 성능이 아니다.
물리/제어가 끼면 pose 재현이 흐트러져 오차의 출처가 섞이고, 차체가 마커를 가릴 수도 있다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PROBE = """
    <!-- 스윕용 정적 프로브. 시작 위치는 수조 밖(스윕 스크립트가 set_pose 로 옮긴다). -->
    <include>
      <name>nbv_probe_camera</name>
      <pose>0 0 5 0 0 0</pose>
      <uri>model://nbv_probe_camera</uri>
    </include>
  </world>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("pool_6x10x3.sdf"))
    ap.add_argument("--out", type=Path, default=Path("pool_sweep.sdf"))
    ap.add_argument("--world-name", default="pool_sweep")
    args = ap.parse_args()

    s = args.src.read_text()
    s = s.replace('<world name="pool_6x10x3">', f'<world name="{args.world_name}">')
    # ROV include 제거 (주석 포함)
    s, n = re.subn(r"\n[ \t]*<!--[^<]*?스폰 깊이.*?-->\s*<include>\s*<pose[^>]*>[^<]*</pose>\s*"
                   r"<uri>model://bluerov2_heavy</uri>\s*</include>\n",
                   "\n", s, flags=re.S)
    if n != 1:
        raise SystemExit(f"ROV include 제거 실패 (matched {n}) — world 구조가 바뀌었다")
    # observer 카메라도 제거: 스윕에 불필요한 렌더 비용
    s, n = re.subn(r"\n[ \t]*<model name=\"observer_camera\">.*?</model>\n", "\n", s, flags=re.S)
    if n != 1:
        raise SystemExit("observer_camera 제거 실패")
    s = s.replace("  </world>", PROBE, 1)
    args.out.write_text(s)
    import xml.etree.ElementTree as ET
    ET.parse(args.out)
    print(f"[sweep-world] {args.out} (world name={args.world_name}) XML ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
