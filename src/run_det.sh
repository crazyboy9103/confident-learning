#!/bin/bash

lr_values=(0.0001)
epoch_values=(12)
noise_types=("overlook" "badloc" "swap")
task_names=("cosmetic" "chocoball")
data_paths=("cosmetic" "chocoball")
batch_sizes=(8 16)
task_count=${#task_names[@]}

for lr in "${lr_values[@]}"; do
    for epoch in "${epoch_values[@]}"; do
        for noise in "${noise_types[@]}"; do
            for i in $(seq 0 $((task_count - 1))); do
                task=${task_names[$i]}
                path=${data_paths[$i]}
                batch_size=${batch_sizes[$i]}
                echo "Running: LR=$lr, Epochs=$epoch, Task=$task, Path=$path"
                python main_det.py model.optimizer.lr=$lr num_epochs=$epoch task_name=$task paths.data_root_dir="/datasets/conflearn/det/$path" conflearn=$noise data.batch_size=$batch_size
            done
        done
    done
done
