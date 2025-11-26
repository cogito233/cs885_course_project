#!/bin/bash
# GPU2 运行 batch_size 6,8
source .venv/bin/activate
DATE_SUFFIX=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_gpu2_bs68_${DATE_SUFFIX}"
mkdir -p ${OUTPUT_DIR}
cd ${OUTPUT_DIR}

nohup python ../src/benchmark_per_turn.py \
    --gpu 2 \
    --turns 50 \
    --num_traj 50 \
    --batch_sizes "6,8" \
    > run_gpu2_bs68_${DATE_SUFFIX}.log 2>&1 &

echo "GPU2 任务已启动，PID: $!"
echo "日志文件: ${OUTPUT_DIR}/run_gpu2_bs68_${DATE_SUFFIX}.log"
echo "输出目录: ${OUTPUT_DIR}/"
