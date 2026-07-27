#!/usr/bin/env bash
# run_experiment.sh
# BROVVelEnv 학습(train.py) → 평가(test_policy.py, Fig.4 3-trial: straight_line/
# square_ballast/square_random_attitude) 자동화. 각 단계는 Kit 앱을 기동/종료하는
# 독립 프로세스라(AppLauncher는 프로세스당 한 번만 기동 가능) 별도 python.sh 호출로 분리.
# 컨테이너 내부 실행:
#   docker exec -it isaac-lab-base bash /workspace/OceanRL_test/step_2_BROV/run_experiment.sh \
#       <experiment_name> [num_envs=512] [max_iterations=300] [record_video=false]

set -euo pipefail

PROJ=/workspace/OceanRL_test/step_2_BROV
PY=/isaac-sim/python.sh
LOG_DIR=${PROJ}/logs

# BROVVelPPORunnerCfg.num_steps_per_env(agents/rsl_rl_ppo_cfg.py) — 총 스텝 수 표시용
NUM_STEPS_PER_ENV=64
TEST_DURATION=60

EXP_NAME=${1:?"사용법: run_experiment.sh <experiment_name> [num_envs=512] [max_iterations=300] [record_video=false]"}
NUM_ENVS=${2:-512}
MAX_ITERATIONS=${3:-300}
RECORD_VIDEO=${4:-false}

VIDEO_FLAG=""
if [[ "${RECORD_VIDEO}" == "true" ]]; then
    VIDEO_FLAG="--record_video"
fi

cd "${PROJ}"
mkdir -p "${LOG_DIR}"

log_header() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════"
}

TOTAL_STEPS=$(( NUM_ENVS * NUM_STEPS_PER_ENV * MAX_ITERATIONS ))
log_header "실험 시작 — ${EXP_NAME} (num_envs=${NUM_ENVS}, max_iterations=${MAX_ITERATIONS}, 총 환경 스텝 수=${TOTAL_STEPS})"

# ── 1) 학습 ──────────────────────────────────────────────────────────────────
${PY} train.py \
    --experiment_name "${EXP_NAME}" \
    --num_envs ${NUM_ENVS} \
    --max_iterations ${MAX_ITERATIONS} \
    --headless \
    2>&1 | tee "${LOG_DIR}/${EXP_NAME}_train.log"

# ── 2) 체크포인트 자동 탐색 (mtime 기준 최신 — train.py가 순서대로 저장하므로 안전) ──
CKPT=$(ls -t "${LOG_DIR}/${EXP_NAME}"/model_*.pt 2>/dev/null | head -1)
if [[ -z "${CKPT}" ]]; then
    echo "[ERROR] 체크포인트를 찾을 수 없음: ${LOG_DIR}/${EXP_NAME}/model_*.pt"
    exit 1
fi
echo "[INFO] 평가에 쓸 체크포인트: ${CKPT}"

# ── 3) 평가 — Fig.4 3-trial ───────────────────────────────────────────────────
for TEST in straight_line square_ballast square_random_attitude; do
    log_header "평가 — ${TEST}"
    ${PY} test_policy.py \
        --checkpoint "${CKPT}" \
        --test ${TEST} \
        --duration ${TEST_DURATION} \
        ${VIDEO_FLAG} \
        --headless \
        2>&1 | tee "${LOG_DIR}/${EXP_NAME}_test_${TEST}.log"
done

log_header "실험 완료 — ${EXP_NAME}"
echo "  체크포인트   : ${CKPT}"
echo "  총 환경 스텝 수: ${TOTAL_STEPS}"
echo "  플롯         : plots/policy_eval_{straight_line,square_ballast,square_random_attitude}.png"
if [[ -n "${VIDEO_FLAG}" ]]; then
    echo "  영상         : videos/policy_eval_*.mp4"
fi
echo "  로그         : ${LOG_DIR}/${EXP_NAME}_*.log"
