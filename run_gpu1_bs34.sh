#!/bin/bash
# GPU1 运行 batch_size 3,4
source .venv/bin/activate
DATE_SUFFIX=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_gpu1_bs34_${DATE_SUFFIX}"
mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

nohup python ../src/benchmark_per_turn.py \
    --gpu 1 \
    --turns 50 \
    --num_traj 50 \
    --batch_sizes "3,4" \
    > run_gpu1_bs34_${DATE_SUFFIX}.log 2>&1 &

echo "GPU1 任务已启动，PID: $!"
echo "日志文件: ${OUTPUT_DIR}/run_gpu1_bs34_${DATE_SUFFIX}.log"
echo "输出目录: ${OUTPUT_DIR}/"
