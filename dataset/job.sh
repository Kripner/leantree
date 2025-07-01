#!/bin/bash


total_workers=128

common_params="--project_path=/home/kripner/troja/arcoss-lean-repo-v4.19.0 --repl_path=/home/kripner/repos/lean-repl-fork/.lake/build/bin/repl --output_dir=/home/kripner/troja/project_1/data/datasets/lean/7"
for i in $(seq 0 $((total_workers - 1))); do
    params="--worker_id $i --total_workers $total_workers"
    echo "Running with params: $params"
    sbatch \
        --job-name="data-${i}" \
        --output="dataset_logs/lean-trees-${i}-%j.out" \
        --error="dataset_logs/lean-trees-${i}-%j.err" \
        --partition=cpu-troja \
        --cpus-per-task=2 \
        run /home/kripner/troja/project_1/venv/bin/python3.12 /home/kripner/leantree/dataset/tree_dataset.py generate $common_params $params
done
