#!/bin/bash
python scripts/rsl_rl/train.py \
--task AME-XLL-Stage0-v0 \
--max_iterations 15000 \
--num_envs 1024 \
--headless
# --run_name  \
# --resume \
# --load_run  \
# --checkpoint  \
