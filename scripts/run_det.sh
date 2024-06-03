#!/bin/bash

cd /workspace/src

lr_values=(0.0001)
epoch_values=(24)
noise_types=("overlook" "badloc" "swap")
task_names=("chocoball" "bone" "cosmetic")
batch_sizes=(16 8 64)
min_sizes=(800 600 500)
max_sizes=(1333 1000 500)
task_count=${#task_names[@]}

for lr in "${lr_values[@]}"; do
    for epoch in "${epoch_values[@]}"; do
        for noise in "${noise_types[@]}"; do
            for i in $(seq 0 $((task_count - 1))); do
                task=${task_names[$i]}
                batch_size=${batch_sizes[$i]}
                min_size=${min_sizes[$i]}
                max_size=${max_sizes[$i]}
                echo "Running: LR=$lr, Epochs=$epoch, Task=$task, Noise=$noise"
                python main_det.py model.optimizer.lr=$lr model.min_size=$min_size model.max_size=$max_size num_epochs=$epoch task_name=$task paths.data_root_dir="/datasets/conflearn/det/$task" conflearn=$noise data.batch_size=$batch_size
            done
        done
    done
done
