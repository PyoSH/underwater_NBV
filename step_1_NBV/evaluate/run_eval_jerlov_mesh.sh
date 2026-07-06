#!/usr/bin/env bash
# run_eval_jerlov_mesh.sh
# 5개 정책(Manual/ScanRL/GenNBV/UW_NBV_5/UW_NBV_DR_2) × IB/II/III 3개 수종 ×
# 1 episode/수종 (대표 1개) 평가 — poster Fig.2 (정성) 용 메시 생성 전용.
#
# run_eval_jerlov.sh(60episode/정책, 통계용)와 별개 — 이건 가볍게 mesh/trajectory만
# 다시 뽑기 위한 것. scikit-image가 설치되어 있어야 recon_mesh_colored.ply가
# 저장됨 (없으면 marching-cubes가 조용히 스킵되어 메시 파일이 안 생김 — 먼저
# `/isaac-sim/python.sh -m pip install scikit-image` 확인).
#
# 컨테이너 안에서 실행:
#   docker exec -it isaac-lab-base bash /workspace/OceanRL_test/step_1_NBV/evaluate/run_eval_jerlov_mesh.sh

set -euo pipefail

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PROJ=/workspace/OceanRL_test/step_1_NBV
EVAL_DIR=${PROJ}/evaluate
PY=/isaac-sim/python.sh
LOG_DIR=/workspace/logs
OUT_BASE=${PROJ}/recon_output
CKPT_DIR=${PROJ}/checkpoints

NUM_EPISODES=3   # IB, II, III (ep_idx % 6 → 0,1,2) — 1C/3C/5C는 poster에서 제외
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

# ── 출력 디렉토리 (jerlov6_*와 겹치지 않는 새 경로) ─────────────────────────────
UW5_STEP=$(basename "${CKPT_UW_NBV5}" .pt | sed 's/.*_//')
DR2_STEP=$(basename "${CKPT_UW_NBV_DR2}" .pt | sed 's/.*_//')
OUT_ORBIT=${OUT_BASE}/jerlov3_mesh_Manual
OUT_SCANRL=${OUT_BASE}/jerlov3_mesh_ScanRL
OUT_GENNBV=${OUT_BASE}/jerlov3_mesh_GenNBV
OUT_UW_NBV5=${OUT_BASE}/jerlov3_mesh_Proposed
OUT_UW_NBV_DR2=${OUT_BASE}/jerlov3_mesh_Proposed_DR

mkdir -p "${LOG_DIR}"
rm -rf "${OUT_ORBIT}" "${OUT_SCANRL}" "${OUT_GENNBV}" "${OUT_UW_NBV5}" "${OUT_UW_NBV_DR2}"
mkdir -p "${OUT_ORBIT}" "${OUT_SCANRL}" "${OUT_GENNBV}" "${OUT_UW_NBV5}" "${OUT_UW_NBV_DR2}"

cd "${PROJ}"

log_header() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════"
}

log_header "Exp 1/5 — Manual Orbit × IB/II/III"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_basic.py" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --max_steps 50 \
    --out_dir "${OUT_ORBIT}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov3mesh_orbit.log"

log_header "Exp 2/5 — ScanRL × IB/II/III"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_recon.py" \
    --checkpoint "${CKPT_SCANRL}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --jerlov_eval \
    --out_dir "${OUT_SCANRL}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov3mesh_scanRL.log"

log_header "Exp 3/5 — GenNBV × IB/II/III"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_recon.py" \
    --checkpoint "${CKPT_GENNBV}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --jerlov_eval \
    --out_dir "${OUT_GENNBV}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov3mesh_genNBV.log"

log_header "Exp 4/5 — UW_NBV_5 (Proposed, no DR) × IB/II/III"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_recon.py" \
    --checkpoint "${CKPT_UW_NBV5}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --jerlov_eval \
    --out_dir "${OUT_UW_NBV5}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov3mesh_UW_NBV5.log"

log_header "Exp 5/5 — UW_NBV_DR_2 (Proposed + Jerlov DR) × IB/II/III"
DISPLAY=${DISPLAY_NUM} ${PY} "${EVAL_DIR}/evaluate_recon.py" \
    --checkpoint "${CKPT_UW_NBV_DR2}" \
    --num_envs ${NUM_ENVS} \
    --num_episodes ${NUM_EPISODES} \
    --jerlov_eval \
    --out_dir "${OUT_UW_NBV_DR2}" \
    2>&1 | tee "${LOG_DIR}/eval_jerlov3mesh_UW_NBV_DR2.log"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  완료 — 5개 정책 × IB/II/III mesh 데이터 생성됨"
echo "  ${OUT_ORBIT}"
echo "  ${OUT_SCANRL}"
echo "  ${OUT_GENNBV}"
echo "  ${OUT_UW_NBV5}"
echo "  ${OUT_UW_NBV_DR2}"
echo "  각 디렉토리의 ep_000(IB)/ep_001(II)/ep_002(III)_env0/recon_mesh_colored.ply 확인 필요"
echo "════════════════════════════════════════════════════════"
