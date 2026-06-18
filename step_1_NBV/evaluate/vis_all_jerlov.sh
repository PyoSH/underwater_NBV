#!/usr/bin/env bash
# vis_all_jerlov.sh
# 6개 Jerlov 수종 전체에 대해 visualize_trajectory + visualize_voxel 실행.
# Isaac Sim 불필요 (Open3D 단독 실행).
#
# 사용법:
#   bash evaluate/vis_all_jerlov.sh

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="${SCRIPT_DIR}/.."                          # step_1_NBV/

PY=/isaac-sim/python.sh
echo "[info] Python: ${PY}"

VIS_TRAJ="${PROJ}/utils_NBV/visualize_trajectory_o3d.py"
VIS_VOX="${PROJ}/utils_NBV/visualize_voxel.py"
OUT_DIR="${PROJ}/analysis/vis_jerlov"

# ── 시각화 대상 run 디렉토리 ──────────────────────────────────────────────────
# jerlov_eval=True, num_episodes=6 으로 생성된 recon_output 디렉토리를 지정
RUN_DIRS=(
    "${PROJ}/recon_output/UW_NBV_5_jerlov_DR"
    "${PROJ}/recon_output/UW_NBV_DR_1"
    "${PROJ}/recon_output/UW_NBV_DR_2"
)

# ── Jerlov 수종 순서 (evaluate_recon.py TYPES 배열과 동일) ───────────────────
JERLOV_LABELS=("IB" "II" "III" "1C" "3C" "5C")

# ── 공통 옵션 ─────────────────────────────────────────────────────────────────
TRAJ_OPTS="--width 700 --height 650 --dist_factor 0.9 --sphere_step 2 --arrow_step 5"
VOX_HIRES=""    # 고해상도 원할 경우 "--hires" 로 교체

# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${OUT_DIR}"

log_header() { echo -e "\n──────────── $1 ────────────"; }

for run_dir in "${RUN_DIRS[@]}"; do
    [[ -d "${run_dir}" ]] || { echo "[skip] 없음: ${run_dir}"; continue; }
    run_name="$(basename "${run_dir}")"
    log_header "${run_name}"

    # ── visualize_trajectory: 6 수종 가로 나열 1장 ───────────────────────────
    EPISODE_ARGS=()
    for j in "${!JERLOV_LABELS[@]}"; do
        ep_dir="${run_dir}/ep_00${j}_env0"
        if   [[ -f "${ep_dir}/recon_mesh_colored.ply" ]]; then mesh_ok=1
        elif [[ -f "${ep_dir}/gt_surface.ply"         ]]; then mesh_ok=1
        else                                                    mesh_ok=0
        fi

        if [[ -f "${ep_dir}/trajectory_xyz.npy" && ${mesh_ok} -eq 1 ]]; then
            EPISODE_ARGS+=("${JERLOV_LABELS[$j]}:${ep_dir}")
        else
            echo "  [skip traj] ep_00${j}: trajectory 또는 mesh 없음"
        fi
    done

    if [[ ${#EPISODE_ARGS[@]} -gt 0 ]]; then
        out_png="${OUT_DIR}/traj_${run_name}.png"
        echo "[traj] 수종=${#EPISODE_ARGS[@]}개 → ${out_png}"
        "${PY}" "${VIS_TRAJ}" \
            --episodes "${EPISODE_ARGS[@]}" \
            --out "${out_png}" \
            ${TRAJ_OPTS}

        mesh_png="${OUT_DIR}/mesh_${run_name}.png"
        echo "[mesh] 수종=${#EPISODE_ARGS[@]}개 → ${mesh_png}"
        "${PY}" "${VIS_TRAJ}" \
            --episodes "${EPISODE_ARGS[@]}" \
            --out "${mesh_png}" \
            ${TRAJ_OPTS} --mesh_only
    else
        echo "[skip traj] ${run_name}: 유효 에피소드 없음"
    fi

    # ── visualize_voxel: 수종별 서브디렉토리 ─────────────────────────────────
    for j in "${!JERLOV_LABELS[@]}"; do
        ep_dir="${run_dir}/ep_00${j}_env0"
        jerlov="${JERLOV_LABELS[$j]}"

        if [[ ! -f "${ep_dir}/weight_vol.npy" ]]; then
            echo "  [skip voxel] ${jerlov}: weight_vol.npy 없음"
            continue
        fi

        vox_out="${OUT_DIR}/voxel_${run_name}/${jerlov}"
        mkdir -p "${vox_out}"
        echo "[voxel] ${jerlov} → ${vox_out}"
        "${PY}" "${VIS_VOX}" \
            --ep_dir "${ep_dir}" \
            --out_dir "${vox_out}" \
            ${VOX_HIRES}
    done
done

# ── 결과 요약 ─────────────────────────────────────────────────────────────────
log_header "완료"
echo "출력 디렉토리: ${OUT_DIR}"
echo ""
echo "trajectory 이미지:"
find "${OUT_DIR}" -maxdepth 1 -name "traj_*.png" | sort | sed 's/^/  /'
echo ""
echo "mesh-only 이미지:"
find "${OUT_DIR}" -maxdepth 1 -name "mesh_*.png" | sort | sed 's/^/  /'
echo ""
echo "voxel 이미지:"
find "${OUT_DIR}" -name "binary_voxel.png" -o -name "quality_voxel.png" | sort | sed 's/^/  /'
