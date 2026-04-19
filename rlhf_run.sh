for seed in 0 1 2 3 4
do
  python vpg.py \
    --epochs 100 \
    --checkpoint \
    --checkpoint_dir ./rlhf_${seed} \
    --reward reward.params \
    --log-dir ./log_rlhf_${seed}
done
wait