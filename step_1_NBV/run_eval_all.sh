#!/usr/bin/env bash
# run_eval_all.sh
# 4개 알고리즘을 순차적으로 평가 후 결과를 집계.
# 컨테이너 안에서 실행:
#   docker exec -it isaac-lab-base bash /workspace/OceanRL_test/step_1_NBV/run_eval_all.sh

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJ=/workspace/OceanRL_test/step_1_NBV
PY=/isaac-sim/python.sh
LOG_DIR=/workspace/logs
OUT_BASE=${PROJ}/recon_output

NUM_EPISODES=30
DISPLAY_NUM=:99

# ── 체크포인트 ─────────────────────────────────────────────────────────────────
CKPT_UW_NBV2=${PROJ}/checkpoints/UW_NBV_2/genNBV_quality_step_0000327680.pt
CKPT_GENNBV=${PROJ}/checkpoints/genNBV/genNBV_step_0000491520.pt
CKPT_SCANRL=${PROJ}/checkpoints/scanRL_paper/scanRL_step_0000400000.pt

# ── 출력 디렉토리 ──────────────────────────────────────────────────────────────
OUT_UW_NBV2=${OUT_BASE}/UW_NBV_2_327k
OUT_GENNBV=${OUT_BASE}/genNBV_eval
OUT_SCANRL=${OUT_BASE}/scanRL_eval
OUT_ORBIT=${OUT_BASE}/basic_orbit
OUT_ANALYSIS=${PROJ}/analysis

mkdir -p "${LOG_DIR}"
mkdir -p "${OUT_UW_NBV2}" "${OUT_GENNBV}" "${OUT_SCANRL}" "${OUT_ORBIT}" "${OUT_ANALYSIS}"

cd "${PROJ}"

log_header() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════"
}

# ── Exp 1: UW_NBV_2 (quality model, 327680) ───────────────────────────────────
log_header "Exp 1/4 — UW_NBV_2 (genNBV_quality, step 327680)"
DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_recon.py" \
    --checkpoint "${CKPT_UW_NBV2}" \
    --num_envs 1 \
    --num_episodes ${NUM_EPISODES} \
    --out_dir "${OUT_UW_NBV2}" \
    2>&1 | tee "${LOG_DIR}/eval_UW_NBV2_327k.log"

# ── Exp 2: GenNBV binary ───────────────────────────────────────────────────────
log_header "Exp 2/4 — GenNBV binary (step 491520)"
DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_recon.py" \
    --checkpoint "${CKPT_GENNBV}" \
    --num_envs 1 \
    --num_episodes ${NUM_EPISODES} \
    --out_dir "${OUT_GENNBV}" \
    2>&1 | tee "${LOG_DIR}/eval_genNBV.log"

# ── Exp 3: ScanRL ─────────────────────────────────────────────────────────────
log_header "Exp 3/4 — ScanRL (step 400000)"
DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_recon.py" \
    --checkpoint "${CKPT_SCANRL}" \
    --num_envs 1 \
    --num_episodes ${NUM_EPISODES} \
    --out_dir "${OUT_SCANRL}" \
    2>&1 | tee "${LOG_DIR}/eval_scanRL.log"

# ── Exp 4: Manual Orbit ───────────────────────────────────────────────────────
log_header "Exp 4/4 — Manual Orbit (orbital policy)"
DISPLAY=${DISPLAY_NUM} ${PY} "${PROJ}/evaluate_basic.py" \
    --num_envs 1 \
    --num_episodes ${NUM_EPISODES} \
    --out_dir "${OUT_ORBIT}" \
    2>&1 | tee "${LOG_DIR}/eval_orbit.log"

# ── 결과 집계 (Isaac Sim 불필요) ────────────────────────────────────────────────
log_header "결과 집계 — analyze_results.py"
${PY} "${PROJ}/analyze_results.py" \
    --results \
        "UW_NBV_2:${OUT_UW_NBV2}" \
        "GenNBV:${OUT_GENNBV}" \
        "ScanRL:${OUT_SCANRL}" \
        "Manual:${OUT_ORBIT}" \
    --success_thr 0.82 \
    --out_dir "${OUT_ANALYSIS}" \
    2>&1 | tee "${LOG_DIR}/analyze_results.log"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  모든 실험 완료"
echo "  비교 테이블 : ${OUT_ANALYSIS}/comparison_table.csv"
echo "  커브 플롯   : ${OUT_ANALYSIS}/coverage_q_curve.png"
echo "════════════════════════════════════════════════════════"
