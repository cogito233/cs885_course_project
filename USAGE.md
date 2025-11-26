# Usage Guide

## Quick Start Examples

### 1. Run Benchmark on GPU 1

```bash
# Default: 50 trajectories, 50 turns, batch sizes [3, 4]
bash RUN_COMMAND.sh 1

# Custom configuration
bash RUN_COMMAND.sh 1 50 50  # GPU, turns, num_trajectories
```

### 2. Run Benchmark on GPU 2

```bash
# Default: batch sizes [6, 8]
bash RUN_COMMAND.sh 2
```

### 3. Run Benchmark on GPU 3

```bash
# Default: batch sizes [12, 16]
bash RUN_COMMAND.sh 3
```

### 4. Monitor Progress

```bash
# View real-time logs
tail -f logs/BENCHMARK_PER_TURN_GPU1_BS34.log

# Auto-refresh every 5 seconds
watch -n 5 'tail -20 logs/BENCHMARK_PER_TURN_GPU1_BS34.log'

# Monitor GPU utilization
watch -n 1 nvidia-smi
```

### 5. Generate Visualizations

After the benchmark completes:

```bash
# GPU 1 results
python src/plot_metrics.py results/per_turn_metrics_gpu1_50traj_50turns_bs4_3.jsonl \
  --output_dir plots/gpu1

# GPU 2 results
python src/plot_metrics.py results/per_turn_metrics_gpu2_50traj_50turns_bs8_6.jsonl \
  --output_dir plots/gpu2

# Generate only comparison plots
python src/plot_metrics.py results/per_turn_metrics_gpu1_*.jsonl \
  --output_dir plots/gpu1 \
  --comparison
```

## Advanced Usage

### Custom Batch Sizes

```bash
# Run with custom batch sizes
CUDA_VISIBLE_DEVICES=1 python src/benchmark_per_turn.py \
  --gpu 1 \
  --turns 50 \
  --num_traj 50 \
  --batch_sizes "2,3,4,5"
```

### Test Different Configurations

```bash
# Short test: 10 trajectories, 10 turns
CUDA_VISIBLE_DEVICES=1 python src/benchmark_per_turn.py \
  --gpu 1 \
  --turns 10 \
  --num_traj 10 \
  --batch_sizes "4,8"

# Long test: 100 trajectories, 50 turns
CUDA_VISIBLE_DEVICES=2 python src/benchmark_per_turn.py \
  --gpu 2 \
  --turns 50 \
  --num_traj 100 \
  --batch_sizes "6,8"
```

### Stop Running Benchmark

```bash
# Find the process ID
cat .benchmark_pid_gpu1

# Stop the benchmark
kill $(cat .benchmark_pid_gpu1)

# Force stop if needed
kill -9 $(cat .benchmark_pid_gpu1)
```

## Output Files

After running a benchmark, you'll find:

### Results Directory
- `results/per_turn_metrics_gpu{N}_50traj_50turns_bs{X}_{Y}.jsonl`
  - Detailed per-turn metrics (one line per turn completion)
  - Fields: batch_size, timestamp, traj_id, turn_idx, turn_tokens, token_throughput, etc.

- `results/per_turn_summary_gpu{N}_50traj_50turns_bs{X}_{Y}.json`
  - Summary statistics for all batch sizes tested
  - Overall throughput, average times, total tokens, etc.

### Logs Directory
- `logs/BENCHMARK_PER_TURN_GPU{N}_BS{X}{Y}.log`
  - Full execution log with progress updates
  - Model loading info, GPU memory usage, error messages

### Plots Directory
- `plots/gpu{N}/batch_size_{X}.png`
  - Individual plots for each batch size
  - Shows token throughput, cumulative tokens, active trajectories

- `plots/gpu{N}/batch_size_comparison.png`
  - Comparison plot across all batch sizes
  - Easy visual comparison of performance

## Interpreting Results

### Key Metrics

1. **Token Throughput (tokens/s)**
   - Higher is better
   - Primary metric for generation speed
   - Typically: 50-150 tok/s for Qwen2.5-14B on A100

2. **Trajectory Throughput (traj/s)**
   - Complete trajectories per second
   - Indicates end-to-end performance
   - Typically: 0.01-0.02 traj/s for 50-turn dialogues

3. **Average Time per Trajectory (s)**
   - Lower is better
   - Total time to complete one full dialogue
   - Typically: 50-100s for 50-turn dialogues

### Performance Expectations

| Configuration | Expected Token Throughput | Notes |
|--------------|--------------------------|-------|
| BS=3-4 (small) | 90-110 tok/s | Best for latency |
| BS=6-8 (medium) | 70-80 tok/s | Balanced throughput |
| BS=12-16 (large) | 50-70 tok/s | May have overhead |

### Troubleshooting

**Low throughput (<50 tok/s)**
- Check GPU utilization: `nvidia-smi`
- Reduce batch size if memory constrained
- Check for CPU bottlenecks

**Out of Memory (OOM)**
- Reduce `mem_fraction_static` (default: 0.88 → 0.85)
- Reduce `max_total_tokens` (default: 131k → 100k)
- Use smaller batch size

**Process hangs**
- Check SGLang server logs
- Kill process and restart with smaller batch size
- Verify model path is correct

## Example Workflow

Complete workflow from start to finish:

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify data file exists
ls -lh 20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl

# 3. Run benchmark on GPU 1
bash RUN_COMMAND.sh 1

# 4. Monitor progress (in another terminal)
tail -f logs/BENCHMARK_PER_TURN_GPU1_BS34.log

# 5. Wait for completion (will take ~1-2 hours)
# Look for "✓ 完成!" or "✓ Complete!" in logs

# 6. Generate visualizations
python src/plot_metrics.py \
  results/per_turn_metrics_gpu1_50traj_50turns_bs4_3.jsonl \
  --output_dir plots/gpu1

# 7. View results
ls -lh results/
ls -lh plots/gpu1/

# 8. View summary
cat results/per_turn_summary_gpu1_50traj_50turns_bs4_3.json | python -m json.tool
```

## Tips & Best Practices

1. **Start Small**: Test with 10 trajectories first to verify setup
2. **Monitor Resources**: Keep `nvidia-smi` running to watch memory
3. **Use `nohup`**: For long-running tests, use the provided script
4. **Save Logs**: Always redirect output to log files for debugging
5. **Compare Configs**: Run multiple batch sizes to find optimal point
6. **Visualize Results**: Always generate plots for easy interpretation

## Support

For issues or questions:
- Check logs in `logs/` directory
- Review `archive/` for additional documentation
- Verify SGLang installation: `python -c "import sglang; print(sglang.__version__)"`

