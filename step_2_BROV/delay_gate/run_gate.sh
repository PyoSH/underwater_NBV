#!/bin/bash
# 단계 1 gate — velocity_hold [0.5,0,0], 지연 0/40/60/80 ms.
# 신선도 jitter 는 재현성을 위해 평가에서 항상 끈다(학습에서는 A/B 양쪽에 0.15).
# usage: run_gate.sh <profile> <checkpoint> <tag> [duration]
set -u
PROFILE=$1; CKPT=$2; TAG=$3; DUR=${4:-20}
cd /workspace/OceanRL_test/step_2_BROV
for D in 0 40 60 80; do
  /isaac-sim/python.sh test_policy.py --test velocity_hold --hold_velocity 0.5 0 0 \
    --profile "$PROFILE" --checkpoint "$CKPT" --duration "$DUR" --headless \
    --action_delay_ms "$D" --obs_stale_prob 0 \
    --dump_log "delay_gate/${TAG}_d${D}.json" > "/tmp/gate_${TAG}_d${D}.log" 2>&1
  echo "  ${TAG} d=${D} rc=$?"
done
