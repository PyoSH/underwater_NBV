#!/usr/bin/env bash
# run_eval_jerlov.sh
# 5개 정책(Manual/ScanRL/GenNBV/UW_NBV_5/UW_NBV_DR_2) × 6개 Jerlov 수종(IB/II/III/1C/3C/5C)
# × 10 episode/수종을 순차 평가 후 결과 집계.
# poster Fig.1 (정량) — "Jerlov DR이 수종별 coverage AUC에 효과적이었는가"를 보기 위한 설계.
#
# 변인 통제: 암석 pose / 카메라 시작 pose는 evaluate_recon.py·evaluate_basic.py의
# eval_mode=True에서 항상 고정값(eval_theta/eval_phi/eval_psi, 암석 회전 미적용)으로
# 유지됨 — 수종(jerlov_types)만 의도적으로 변화시키는 변수. 10회 반복은 그 외
# 시뮬레이터 단의 잔차 노이즈(렌더링/센서)에 대한 분산을 확인하기 위함.
#
# 컨테이너 안에서 실행:
#   docker exec -it isaac-lab-base bash /workspace/OceanRL_test/step_1_NBV/evaluate/run_eval_jerlov.sh

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJ=/workspace/OceanRL_test/step_1_NBV
EVAL_DIR=${PROJ}/evaluate
PY=/isaac-sim/python.sh
LOG_DIR=/workspace/logs
OUT_BASE=${PROJ}/recon_output
CKPT_DIR=${PROJ}/checkpoints

NUM_EPISODES=60   # 6 수종 × 10 episode/수종 (ep_idx % 6 으로 수종 순환)
NUM_ENVS=1
DISPLAY_NUM=:99

# ── 체크포인트 ─────────────────────────────────────────────────────────────────
CKPT_UW_NBV5=$(ls -t "${CKPT_DIR}/UW_NBV_5"/genNBV_quality_step_*.pt 2>/dev/null | head -1)
if [[ -z "${CKPT_UW_NBV5}" ]]; then
    echo "[ERROR] UW_NBV_5 체크포인트가 없습니다: ${CKPT_DIR}/UW_NBV_5/"
    exit 1
fi

CKPT_GENNBV=${CKPT_DIR}/genNBV/genNBV_step_0000307200.pt
CKPT_SCANRL=${CKPT_DIR}/scanRL_paper/scanRL_step_0000300000.pt

CKPT_UW_NBV_DR2=$(ls -t "${CKPT_DIR}/UW_NBV_DR_2"/genNBV_quality_step_*.pt 2>/dev/null | head -1)
if [[ -z "${CKPT_UW_NBV_DR2}" ]]; then
    echo "[ERROR] UW_NBV_DR_2 체크포인트가 없습니다: ${CKPT_DIR}/UW_NBV_DR_2/"
    exit 1
fi

# ── 출력 디렉토리 ──────────────────────────────────────────────────────────────
UW5_STEP=$(basename "${CKPT_UW_NBV5}" .pt | sed 's/.*_//')
DR2_STEP=$(basename "${CKPT_UW_NBV_DR2}" .pt | sed 's/.*_//')
OUT_ORBIT=${OUT_BASE}/jerlov6_basic_orbit
OUT_SCANRL=${OUT_BASE}/jerlov6_scanRL
OUT_GENNBV=${OUT_BASE}/jerlov6_genNBV
OUT_UW_NBV5=${OUT_BASE}/jerlov6_UW_NBV_5_${UW5_STEP}
OUT_UW_NBV_DR2=${OUT_BASE}/jerlov6_UW_NBV_DR_2_${DR2_STEP}
OUT_ANALYSIS=${PROJ}/analysis/jerlov6_compare

mkdir -p "${LOG_DIR}"

rm -rf "${OUT_ORBIT}" "${OUT_SCANRL}" "${OUT_GENNBV}" "${OUT_UW_NBV5}" "${OUT_UW_NBV_DR2}" "${OUT_ANALYSIS}"
mkdir -p "${OUT_ORBIT}" "${OUT_SCANRL}" "${OUT_GENNBV}" "${OUT_UW_NBV5}" "${OUT_UW_NBV_DR2}" "${OUT_ANALYSIS}"

cd "${PROJ}"

log_header() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════"
}

# ── Exp 1: Manual Orbit (jerlov_dr_enabled은 evaluate_basic.py 내부에서 항상 True) ──
log_header "Exp 1/5 — Manual Orbit × 6 Jerlov × 10 episode"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_basic.py" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --max_steps 50 \
    --out_dir "${OUT_ORBIT}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov6_orbit.log"

# ── Exp 2: ScanRL ────────────────────────────────────────────────────────────
log_header "Exp 2/5 — ScanRL × 6 Jerlov × 10 episode"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_recon.py" \
    --checkpoint "${CKPT_SCANRL}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --jerlov_eval \
    --out_dir "${OUT_SCANRL}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov6_scanRL.log"

# ── Exp 3: GenNBV binary ─────────────────────────────────────────────────────
log_header "Exp 3/5 — GenNBV × 6 Jerlov × 10 episode"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_recon.py" \
    --checkpoint "${CKPT_GENNBV}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --jerlov_eval \
    --out_dir "${OUT_GENNBV}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov6_genNBV.log"

# ── Exp 4: UW_NBV_5 (Proposed, no DR) ───────────────────────────────────────
log_header "Exp 4/5 — UW_NBV_5 (Proposed, no DR) × 6 Jerlov × 10 episode"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_recon.py" \
    --checkpoint "${CKPT_UW_NBV5}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --jerlov_eval \
    --out_dir "${OUT_UW_NBV5}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov6_UW_NBV5.log"

# ── Exp 5: UW_NBV_DR_2 (Proposed + Jerlov DR) ───────────────────────────────
log_header "Exp 5/5 — UW_NBV_DR_2 (Proposed + Jerlov DR) × 6 Jerlov × 10 episode"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_recon.py" \
    --checkpoint "${CKPT_UW_NBV_DR2}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --jerlov_eval \
    --out_dir "${OUT_UW_NBV_DR2}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov6_UW_NBV_DR2.log"

# ── 결과 집계 (수종별 breakdown, Isaac Sim 불필요) ──────────────────────────
log_header "결과 집계 — analyze_results.py --jerlov_eval"
${PY} "${EVAL_DIR}/analyze_results.py" \
    --results \
        "Manual:${OUT_ORBIT}" \
        "ScanRL:${OUT_SCANRL}" \
        "GenNBV:${OUT_GENNBV}" \
        "UW_NBV_5:${OUT_UW_NBV5}" \
        "UW_NBV_DR_2:${OUT_UW_NBV_DR2}" \
    --jerlov_eval \
    --out_dir "${OUT_ANALYSIS}" \
    2>&1 | tee "${LOG_DIR}/analyze_results_jerlov6.log"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  모든 실험 완료 (5-way × 6 Jerlov × 10 episode)"
echo "  비교 테이블(전체 평균)     : ${OUT_ANALYSIS}/comparison_table.csv"
echo "  수종별 bar chart          : ${OUT_ANALYSIS}/jerlov_bar.png"
echo "  수종별 coverage_q 커브    : ${OUT_ANALYSIS}/jerlov_cov_q_curves.png"
echo "  수종별 coverage_bin 커브  : ${OUT_ANALYSIS}/jerlov_cov_bin_curves.png"
echo "════════════════════════════════════════════════════════"
