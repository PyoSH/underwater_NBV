#!/usr/bin/env bash
# run_eval_all.sh
# 비교 실험 4개를 순차 평가 후 결과 집계.
# 컨테이너 안에서 실행:
#   docker exec -it isaac-lab-base bash /workspace/OceanRL_test/step_1_NBV/run_eval_all.sh

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJ=/workspace/OceanRL_test/step_1_NBV
PY=/isaac-sim/python.sh
LOG_DIR=/workspace/logs
OUT_BASE=${PROJ}/recon_output
CKPT_DIR=${PROJ}/checkpoints

NUM_EPISODES=10
NUM_ENVS=1
DISPLAY_NUM=:99

# ── 체크포인트 ─────────────────────────────────────────────────────────────────
# UW_NBV_5: 학습 완료 후 가장 최신 체크포인트 자동 선택
CKPT_UW_NBV5=$(ls -t "${CKPT_DIR}/UW_NBV_5"/genNBV_quality_step_*.pt 2>/dev/null | head -1)
if [[ -z "${CKPT_UW_NBV5}" ]]; then
    echo "[ERROR] UW_NBV_5 체크포인트가 없습니다: ${CKPT_DIR}/UW_NBV_5/"
    exit 1
fi

CKPT_GENNBV=${CKPT_DIR}/genNBV/genNBV_step_0000307200.pt
CKPT_SCANRL=${CKPT_DIR}/scanRL_paper/scanRL_step_0000300000.pt

# ── 출력 디렉토리 ──────────────────────────────────────────────────────────────
UW5_STEP=$(basename "${CKPT_UW_NBV5}" .pt | sed 's/.*_//')
OUT_UW_NBV5=${OUT_BASE}/UW_NBV_5_${UW5_STEP}
OUT_GENNBV=${OUT_BASE}/genNBV_eval
OUT_SCANRL=${OUT_BASE}/scanRL_eval
OUT_ORBIT=${OUT_BASE}/basic_orbit
OUT_ANALYSIS=${PROJ}/analysis

mkdir -p "${LOG_DIR}"

# 기존 결과 삭제 (metric 변경으로 인한 재평가)
rm -rf "${OUT_UW_NBV5}" "${OUT_GENNBV}" "${OUT_SCANRL}" "${OUT_ORBIT}" "${OUT_ANALYSIS}"
mkdir -p "${OUT_UW_NBV5}" "${OUT_GENNBV}" "${OUT_SCANRL}" "${OUT_ORBIT}" "${OUT_ANALYSIS}"

cd "${PROJ}"

log_header() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════"
}
# ── Exp 1: Manual Orbit ──────────────────────────────────────────────────────
log_header "Exp 1/4 — Manual Orbit (orbital policy)"
DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_basic.py" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --out_dir "${OUT_ORBIT}" \
    2>&1 | tee "${LOG_DIR}/eval_orbit.log"

# ── Exp 2: ScanRL ────────────────────────────────────────────────────────────
log_header "Exp 2/4 — ScanRL (step 400000)"
DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_recon.py" \
    --checkpoint "${CKPT_SCANRL}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --out_dir "${OUT_SCANRL}" \
    2>&1 | tee "${LOG_DIR}/eval_scanRL.log"

# ── Exp 3: GenNBV binary ─────────────────────────────────────────────────────
log_header "Exp 3/4 — GenNBV binary (step 491520)"
DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_recon.py" \
    --checkpoint "${CKPT_GENNBV}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --out_dir "${OUT_GENNBV}" \
    2>&1 | tee "${LOG_DIR}/eval_genNBV.log"

# ── Exp 4: UW_NBV_5 (primary, algo_UW_NBV, max metric) ──────────────────────
log_header "Exp 4/4 — UW_NBV_5 (algo_UW_NBV, max metric, step ${UW5_STEP})"
DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_recon.py" \
    --checkpoint "${CKPT_UW_NBV5}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --out_dir "${OUT_UW_NBV5}" \
    2>&1 | tee "${LOG_DIR}/eval_UW_NBV5.log"

# ── 결과 집계 (Isaac Sim 불필요) ─────────────────────────────────────────────
log_header "결과 집계 — analyze_results.py"
${PY} "${PROJ}/analyze_results.py" \
    --results \
        "Manual:${OUT_ORBIT}" \
        "ScanRL:${OUT_SCANRL}" \
        "GenNBV:${OUT_GENNBV}" \
        "UW_NBV_5:${OUT_UW_NBV5}" \
    --out_dir "${OUT_ANALYSIS}" \
    2>&1 | tee "${LOG_DIR}/analyze_results.log"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  모든 실험 완료"
echo "  비교 테이블 : ${OUT_ANALYSIS}/comparison_table.csv"
echo "  커버리지 커브: ${OUT_ANALYSIS}/coverage_q_curve.png"
echo "  UW_NBV_5 ckpt: ${CKPT_UW_NBV5}"
echo "════════════════════════════════════════════════════════"
