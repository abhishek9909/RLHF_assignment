for seed in 0 1 2 3 4
do
  python vpg.py \
    --epochs 200 \
    --checkpoint \
    --checkpoint_dir ./synthetic_${seed} \
    --render \
    --log-dir ./log_${seed}
done
wait