#!/bin/bash

# Define arrays for learning rates, epochs, task names, and data paths
lr_values=(0.001 0.0001)
epoch_values=(12 24)
# task_names=("pill" "fruit" "ramen")
# data_paths=("applied_materials_processed" "Fruit_processed" "ramen_processed_data")
task_names=("fruit")
data_paths=("Fruit_processed")
# Calculate the number of tasks
task_count=${#task_names[@]}

# Loop through all combinations of configurations
for lr in "${lr_values[@]}"; do
    for epoch in "${epoch_values[@]}"; do
        for i in $(seq 0 $((task_count - 1))); do
            task=${task_names[$i]}
            path=${data_paths[$i]}
            echo "Running: LR=$lr, Epochs=$epoch, Task=$task, Path=$path"
            python main_cla.py model.optimizer.lr=$lr num_epochs=$epoch task_name=$task paths.data_root_dir="/datasets/conflearn/cla/$path"
        done
    done
done
