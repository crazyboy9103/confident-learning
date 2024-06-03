#!/bin/bash

cd /workspace/src

lr_values=(0.001)
epoch_values=(12)
resize_sizes=(800 384 384)
task_names=("ramen" "fruit" "pill")
task_count=${#task_names[@]}

for lr in "${lr_values[@]}"; do
    for epoch in "${epoch_values[@]}"; do
        for i in $(seq 0 $((task_count - 1))); do
            task=${task_names[$i]}
            resize=${resize_sizes[$i]}
            echo "Running: LR=$lr, Epochs=$epoch, Task=$task"
            python main_cla.py model.optimizer.lr=$lr num_epochs=$epoch task_name=$task resize_size=$resize paths.data_root_dir="/datasets/conflearn/cla/$task"
        done
    done
done
