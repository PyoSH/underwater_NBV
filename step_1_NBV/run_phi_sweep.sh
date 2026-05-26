#!/usr/bin/env bash
# run_phi_sweep.sh
# eval_phi 5단계(20,35,50,65,80°)에서 UW_NBV_2 vs Manual Orbit 비교
# 컨테이너 내부 실행:
#   docker exec -it isaac-lab-base bash /workspace/OceanRL_test/step_1_NBV/run_phi_sweep.sh

set -euo pipefail

PROJ=/workspace/OceanRL_test/step_1_NBV
PY=/isaac-sim/python.sh
LOG_DIR=/workspace/logs
OUT_BASE=${PROJ}/recon_output/phi_sweep

CKPT_UW_NBV2=${PROJ}/checkpoints/UW_NBV_2/genNBV_quality_step_0000993280.pt
NUM_EPISODES=15
DISPLAY_NUM=:99

mkdir -p "${LOG_DIR}" "${OUT_BASE}"
cd "${PROJ}"

log_header() {
    echo ""
    echo "════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════"
}

# metric이 sum→max로 변경되었으므로 기존 결과 재활용 불가. 5단계 전체 신규 실행.

for PHI in 20 35 50 65 80; do

    log_header "UW_NBV_2  eval_phi=${PHI}°"
    DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_recon.py" \
        --checkpoint "${CKPT_UW_NBV2}" \
        --num_envs 1 \
        --num_episodes ${NUM_EPISODES} \
        --eval_phi ${PHI} \
        --out_dir "${OUT_BASE}/UW_NBV_2_phi${PHI}" \
        2>&1 | tee "${LOG_DIR}/phi_sweep_UW_phi${PHI}.log"

    log_header "Manual Orbit  eval_phi=${PHI}°"
    DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_basic.py" \
        --num_envs 1 \
        --num_episodes ${NUM_EPISODES} \
        --eval_phi ${PHI} \
        --out_dir "${OUT_BASE}/Manual_phi${PHI}" \
        2>&1 | tee "${LOG_DIR}/phi_sweep_Manual_phi${PHI}.log"

done

log_header "결과 집계"

${PY} "${PROJ}/analyze_phi_sweep.py" \
    --sweep_dir "${OUT_BASE}" \
    --phi_vals 20 35 50 65 80 \
    --success_thr 0.65 \
    --out_dir "${PROJ}/analysis/phi_sweep" \
    2>&1 | tee "${LOG_DIR}/phi_sweep_analyze.log"

echo ""
echo "완료. 결과: ${PROJ}/analysis/phi_sweep/"
