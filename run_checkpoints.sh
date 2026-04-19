#!/usr/bin/env bash
set -euo pipefail

NUM_ROLLOUTS=5
RLHF_DIR="./rlhf_0"
SYN_DIR="./synthetic_0"
OUT_DIR="./eval_logs"

mkdir -p "$OUT_DIR/rlhf_0" "$OUT_DIR/synthetic_0"

# Baseline synthetic checkpoint 9
python rollout_policy.py \
  --checkpoint "$SYN_DIR/policy_checkpoint9.params" \
  --num_rollouts "$NUM_ROLLOUTS" \
  --return-dir "$OUT_DIR/synthetic_0/checkpoint9"

# RLHF checkpoints 0..99
for ckpt in $(seq 0 99); do
  echo "Running RLHF checkpoint $ckpt"
  python rollout_policy.py \
    --checkpoint "$RLHF_DIR/policy_checkpoint${ckpt}.params" \
    --num_rollouts "$NUM_ROLLOUTS" \
    --return-dir "$OUT_DIR/rlhf_0/checkpoint${ckpt}"
done

# Analyze results and make plot
python analyze_returns.py \
  --rlhf-root "$OUT_DIR/rlhf_0" \
  --synthetic-log "$OUT_DIR/synthetic_0/checkpoint9/log.txt" \
  --plot-out "$OUT_DIR/rlhf_vs_synthetic_baseline.png"
