#!/bin/bash
python scripts/rsl_rl/play.py \
--task AME-XLL-Rough-Play-v0 \
--checkpoint /home/mglf/rc/AME_Locomotion/logs/rsl_rl/xll_ame/2026-10-14_21-25-32/model_12800.pt \
--num_envs 1 \
--video \
--video_length 300 \
--save_attention_weights \
--vis_attention \
--keyboard \
--real-time
