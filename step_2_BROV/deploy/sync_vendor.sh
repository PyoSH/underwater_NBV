#!/usr/bin/env bash
# deploy/vendor/의 원본(robots/dynamics/brov2/*, robots/data/BROV2/brov2_heavy.yaml)이
# 바뀌었을 때 재동기화. deploy/ 하나만 복사해도 동작하게 하려고 vendoring했지만,
# 정본은 여전히 robots/ 쪽이라 이 스크립트로 수동 재동기화가 필요하다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"   # OceanRL_test
VENDOR_DIR="${SCRIPT_DIR}/vendor"

cp "${REPO_ROOT}/robots/dynamics/brov2/thruster.py" "${VENDOR_DIR}/thruster.py"
cp "${REPO_ROOT}/robots/dynamics/brov2/params.py" "${VENDOR_DIR}/params.py"
cp "${REPO_ROOT}/robots/data/BROV2/brov2_heavy.yaml" "${VENDOR_DIR}/brov2_heavy.yaml"
# thruster.py가 상대 import하는 제조사 실측 추력 테이블과 그 로더.
# .npz는 build_t200_table.py가 만든 산출물이라 여기서는 복사만 한다.
cp "${REPO_ROOT}/robots/dynamics/brov2/t200_table.py" "${VENDOR_DIR}/t200_table.py"
cp "${REPO_ROOT}/robots/dynamics/brov2/t200_table.npz" "${VENDOR_DIR}/t200_table.npz"
# 액추에이터 동특성 (von Benzon Eq.19 3차) — thruster.py가 상대 import한다.
cp "${REPO_ROOT}/robots/dynamics/brov2/thruster_dynamics.py" "${VENDOR_DIR}/thruster_dynamics.py"

# 원본 params.py의 _BROV2_YAML_PATH는 robots/ 기준 상대경로라 vendor 사본에서는
# deploy/vendor/brov2_heavy.yaml을 직접 가리키도록 한 줄만 고쳐야 한다.
python3 - "${VENDOR_DIR}/params.py" "${VENDOR_DIR}/thruster.py" "${VENDOR_DIR}/t200_table.py" "${VENDOR_DIR}/thruster_dynamics.py" <<'EOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
# [^)]* stops at the FIRST ')', but the source nests os.path.dirname(__file__)
# inside os.path.join(...) -- matching non-greedily up to a ')' alone on its
# own line correctly spans the whole (possibly multi-line) join() call instead.
new_src, n = re.subn(
    r'_BROV2_YAML_PATH = os\.path\.join\(.*?^\)[ \t]*$',
    '_BROV2_YAML_PATH = os.path.join(os.path.dirname(__file__), "brov2_heavy.yaml")',
    src, count=1, flags=re.DOTALL | re.MULTILINE,
)
if n != 1:
    sys.exit(f"[sync_vendor] _BROV2_YAML_PATH pattern not found/matched exactly once in {path} (n={n}) -- source format changed, fix the regex")
open(path, "w").write(new_src)

# Python 3.8/3.9에서도 ``tuple[...]``과 ``X | Y`` 타입 표기가 import 시
# 평가되지 않도록 모든 vendor 모듈에서 annotation 평가를 지연한다.
for path in sys.argv[1:]:
    src = open(path).read()
    if "from __future__ import annotations" not in src:
        end = src.find('"""', 3) + 3
        src = src[:end] + "\n\nfrom __future__ import annotations" + src[end:]
    open(path, "w").write(src)
EOF

echo "[sync_vendor] 완료 — deploy/vendor/{thruster.py,thruster_dynamics.py,params.py,t200_table.py,t200_table.npz,brov2_heavy.yaml} 재동기화됨"
