#!/bin/bash
# Benchmark script for LLM inference performance testing
# Stateful KV Cache version - per-turn metrics recording
# 50 trajectories × 50 turns × different batch sizes

cd /data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854
source .venv/bin/activate

# Default configuration
GPU=${1:-1}
TURNS=${2:-50}
NUM_TRAJ=${3:-50}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 LLM Inference Benchmark - Stateful KV Cache"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Configuration:"
echo "  GPU: $GPU"
echo "  Trajectories: $NUM_TRAJ"
echo "  Turns: $TURNS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Configure batch sizes based on GPU
case $GPU in
  1)
    BATCH_SIZES="3,4"
    LOG_FILE="logs/BENCHMARK_PER_TURN_GPU1_BS34.log"
    ;;
  2)
    BATCH_SIZES="6,8"
    LOG_FILE="logs/BENCHMARK_PER_TURN_GPU2_BS68.log"
    ;;
  3)
    BATCH_SIZES="12,16"
    LOG_FILE="logs/BENCHMARK_PER_TURN_GPU3_BS1216.log"
    ;;
  *)
    BATCH_SIZES="3,4"
    LOG_FILE="logs/BENCHMARK_PER_TURN_GPU${GPU}_DEFAULT.log"
    ;;
esac

echo "Batch Sizes: $BATCH_SIZES"
echo "Log File: $LOG_FILE"
echo ""

# Create directories if they don't exist
mkdir -p logs results plots

# Run benchmark in background
nohup python src/benchmark_per_turn.py \
  --gpu $GPU \
  --turns $TURNS \
  --num_traj $NUM_TRAJ \
  --batch_sizes "$BATCH_SIZES" \
  > $LOG_FILE 2>&1 &

PID=$!
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Benchmark started in background"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Process ID: $PID"
echo ""
echo "📊 Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "🔄 Auto-refresh (every 5 seconds):"
echo "  watch -n 5 'tail -20 $LOG_FILE'"
echo ""
echo "✅ Check process status:"
echo "  ps -p $PID"
echo ""
echo "🎯 GPU utilization:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "📈 After completion, generate plots:"
echo "  python src/plot_metrics.py results/per_turn_metrics_gpu${GPU}_*.jsonl --output_dir plots/gpu${GPU}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Save PID to file for easy management
echo $PID > .benchmark_pid_gpu${GPU}
echo ""
echo "💡 Tip: To stop the benchmark, run:"
echo "  kill $(cat .benchmark_pid_gpu${GPU})"
