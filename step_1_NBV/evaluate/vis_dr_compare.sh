#!/usr/bin/env bash
# vis_dr_compare.sh
# Poster Fig.2 (정성) — 5개 방법론(Manual/ScanRL/GenNBV/Proposed/Proposed+DR) ×
# 3개 Jerlov 수종(IB/II/III)의 경로 + 복원 메시를 grid(행=수종, 열=방법)로 비교.
# run_eval_jerlov_mesh.sh로 생성된 recon_output/jerlov3_mesh_* 데이터를 사용
# (mesh 파일이 있어야 함 — scikit-image 설치 필요, run_eval_jerlov_mesh.sh 참고).
# Isaac Sim 불필요 (Open3D 단독 실행).
#
# 사용법:
#   bash evaluate/vis_dr_compare.sh

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="${SCRIPT_DIR}/.."                          # step_1_NBV/

PY=/isaac-sim/python.sh
echo "[info] Python: ${PY}"

VIS_TRAJ="${PROJ}/utils_NBV/visualize_trajectory_o3d.py"
OUT_DIR="${PROJ}/analysis/vis_dr_compare"
RECON="${PROJ}/recon_output"

# ── 5개 방법론 디렉토리 (run_eval_jerlov_mesh.sh 출력) ──────────────────────
declare -A METHOD_DIR=(
    ["Manual"]="${RECON}/jerlov3_mesh_Manual"
    ["ScanRL"]="${RECON}/jerlov3_mesh_ScanRL"
    ["GenNBV"]="${RECON}/jerlov3_mesh_GenNBV"
    ["Proposed"]="${RECON}/jerlov3_mesh_Proposed"
    ["Proposed+DR"]="${RECON}/jerlov3_mesh_Proposed_DR"
)
METHOD_ORDER=("Manual" "ScanRL" "GenNBV" "Proposed" "Proposed+DR")

# ── 행 = 수종 (evaluate_recon.py TYPES 순서: IB II III 1C 3C 5C 중 앞 3개) ──
# IB: 전반적으로 다들 무난 — Proposed/Proposed+DR 차이가 크지 않음 (대조군)
# II/III: 학습 분포 경계 — Proposed(no DR)는 정체, Proposed+DR은 끝까지 탐색·성공
#         → DR ablation 핵심 증거. 1C/3C/5C는 전원 조기 정체라 grid에서 제외.
TYPE_ORDER=("IB" "II" "III")
declare -A TYPE_EP=( ["IB"]="ep_000_env0" ["II"]="ep_001_env0" ["III"]="ep_002_env0" )

# theta/dist_factor는 모드별로 분리 — 궤적 포함 그림은 전체 경로(Manual의 넓은
# 궤도 등)가 잘리지 않는 각도/거리 유지, mesh_only는 메시를 가깝게+다른 각도로 봄.
BASE_OPTS="--width 480 --height 440 --sphere_step 2 --arrow_step 5"
TRAJ_THETA=190
TRAJ_DIST_FACTOR=0.8
MESH_THETA=55
MESH_DIST_FACTOR=0.25

mkdir -p "${OUT_DIR}"

build_grid() {
    local mode_flag="$1"     # "" 또는 "--mesh_only"
    local out_name="$2"      # 최종 파일명
    local dist_factor="$3"
    local theta="$4"

    local row_pngs=()
    for t in "${TYPE_ORDER[@]}"; do
        local ep="${TYPE_EP[$t]}"
        local episode_args=()
        for m in "${METHOD_ORDER[@]}"; do
            local ep_dir="${METHOD_DIR[$m]}/${ep}"
            if [[ -f "${ep_dir}/trajectory_xyz.npy" && -f "${ep_dir}/recon_mesh_colored.ply" ]]; then
                episode_args+=("${m}:${ep_dir}")
            else
                echo "  [skip] ${t} / ${m}: ${ep_dir} 에 trajectory 또는 mesh 없음"
            fi
        done

        if [[ ${#episode_args[@]} -eq 0 ]]; then
            echo "[skip row] ${t}: 유효 panel 없음"
            continue
        fi

        local row_png="${OUT_DIR}/_row_${t}_${out_name}"
        echo "[row] ${t} (${#episode_args[@]} panels) → ${row_png}"
        "${PY}" "${VIS_TRAJ}" \
            --episodes "${episode_args[@]}" \
            --out "${row_png}" \
            ${BASE_OPTS} --dist_factor "${dist_factor}" --theta "${theta}" ${mode_flag}
        row_pngs+=("${row_png}")
    done

    if [[ ${#row_pngs[@]} -eq 0 ]]; then
        echo "[error] ${out_name}: 생성된 row 없음"
        return 1
    fi

    local final_png="${OUT_DIR}/${out_name}"
    echo "[stitch] ${#row_pngs[@]} rows → ${final_png}"
    "${PY}" - "${final_png}" "${row_pngs[@]}" << 'PYEOF'
import sys
from PIL import Image

out_path, *row_paths = sys.argv[1:]
rows = [Image.open(p) for p in row_paths]
w = max(r.width for r in rows)
h = sum(r.height for r in rows)
canvas = Image.new("RGB", (w, h), (30, 30, 30))
y = 0
for r in rows:
    canvas.paste(r, (0, y))
    y += r.height
canvas.save(out_path)
print(f"[save] {out_path}  ({w}x{h})")
PYEOF
}

build_grid "" "traj_grid.png" "${TRAJ_DIST_FACTOR}" "${TRAJ_THETA}"
build_grid "--mesh_only" "mesh_grid.png" "${MESH_DIST_FACTOR}" "${MESH_THETA}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  완료 (poster Fig.2 — 5방법 × 3수종 grid, 정성)"
echo "  경로+메시 grid: ${OUT_DIR}/traj_grid.png"
echo "  메시만 grid   : ${OUT_DIR}/mesh_grid.png"
echo "════════════════════════════════════════════════════════"
