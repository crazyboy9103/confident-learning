#!/bin/bash

lr_values=(0.0001)
eta_mins=(0.00001)
epoch_values=(12)
noise_types=("overlook" "badloc" "swap")
task_names=("battery")
data_paths=("battery")
task_count=${#task_names[@]}

for lr in "${lr_values[@]}"; do
    for eta_min in "${eta_mins[@]}"; do
        for epoch in "${epoch_values[@]}"; do
            for noise in "${noise_types[@]}"; do
                for i in $(seq 0 $((task_count - 1))); do
                    task=${task_names[$i]}
                    path=${data_paths[$i]}
                    echo "Running: LR=$lr, Epochs=$epoch, Task=$task, Path=$path, Noise=$noise"
                    python main_seg.py model.optimizer.lr=$lr model.scheduler.eta_min=$eta_min num_epochs=$epoch task_name=$task paths.data_root_dir="/datasets/conflearn/seg/$path" conflearn=$noise
                done
            done
        done
    done
done
