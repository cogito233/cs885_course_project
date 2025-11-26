#!/bin/bash
# GPU3 运行 batch_size 16,64
DATE_SUFFIX=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_gpu3_bs16_64_${DATE_SUFFIX}"
mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

nohup python ../src/benchmark_per_turn.py \
    --gpu 3 \
    --turns 50 \
    --num_traj 50 \
    --batch_sizes "16,64" \
    > run_gpu3_bs16_64_${DATE_SUFFIX}.log 2>&1 &

echo "GPU3 任务已启动，PID: $!"
echo "日志文件: ${OUTPUT_DIR}/run_gpu3_bs16_64_${DATE_SUFFIX}.log"
echo "输出目录: ${OUTPUT_DIR}/"
