#!/bin/bash

# Script to reproduce offline IL results

GPU_LIST=(0 1 2 3)

for T in 1 10 20 30; do
for seed in 0 10 100; do

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main_odice_il.py \
  --env_name "hopper-expert-v2" \
  --weight_decay 0.00001 \
  --T $T \
  --seed $seed &

sleep 2
let "task=$task+1"

GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main_odice_il.py \
  --env_name "walker2d-expert-v2" \
  --weight_decay 0.001 \
  --T $T \
  --seed $seed &

sleep 2
let "task=$task+1"

done
done
