set -u
cd /workspace/OceanRL_test/step_2_BROV
for S in 1 7 13 42 99; do
  E="paper_ref_v1_2048_s${S}"
  if [ ! -f "logs/${E}/model_299.pt" ]; then
    echo "=== train seed ${S} ==="
    /isaac-sim/python.sh train.py --profile paper_ref_v1 --experiment_name "${E}" \
      --num_envs 2048 --max_iterations 300 --seed ${S} --headless > "/tmp/tr_${S}.log" 2>&1
    grep -aE "Mean reward|Total time" "/tmp/tr_${S}.log" | tail -2
  fi
  echo "=== eval seed ${S} (pure surge 0.5, no LOS) ==="
  /isaac-sim/python.sh test_policy.py --checkpoint "logs/${E}/model_299.pt" --profile paper_ref_v1 \
    --test velocity_hold --hold_velocity 0.5 0 0 --duration 20 --headless \
    --dump_log "/workspace/OceanRL_test/step_2_BROV/_seed_${S}.json" > /dev/null 2>&1
  echo "seed ${S} done"
done
