"""
带详细指标追踪的基准测试
- 模型: Qwen2.5-14B (128k上下文)
- GPU 1
- 50条轨迹 × 50轮对话
- Batch Size: 2, 3, 4, 6, 8
- 追踪: 时间 vs token吞吐量, 时间 vs 前缀长度
- 保存metadata到jsonl供后续画图
"""

import sglang as sgl
import os, time, json, yaml, sys, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen2.5-14B"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

def load_config():
    with open(YAML_FILE) as f:
        return yaml.safe_load(f)

def load_trajectories(num=50):
    trajs = []
    with open(JSONL_FILE) as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            trajs.append(json.loads(line))
    return trajs

@sgl.function
def multiturn_generate(s, system_msg, first_user_msg, observations, num_turns):
    """多轮对话生成"""
    s += sgl.system(system_msg)
    s += sgl.user(first_user_msg)
    
    for i in range(num_turns):
        s += sgl.assistant(sgl.gen(
            f"turn_{i}",
            max_tokens=256,
            temperature=0.7,
            stop=["<|im_end|>", "\n\nUSER:"]
        ))
        
        if i < len(observations):
            s += sgl.user(observations[i])

def prepare_trajectory_data(traj, config, num_turns):
    """准备单条轨迹数据 - 完整内容"""
    system_prompt = config['system_prompt']
    instance_template = config['instance_prompt']
    
    # 完整的problem_statement
    problem_statement = traj.get('problem_statement', '')
    first_user_msg = instance_template.format(problem_statement=problem_statement)
    
    # 完整的observations
    steps = traj.get('trajectory_steps', [])[:num_turns]
    observations = []
    total_env_time = 0
    
    for step in steps:
        env_time = float(step.get('env_exec_time', 0))
        total_env_time += env_time
        
        # 完整的observation
        observation = step.get('observation', '')
        obs_msg = f"{observation}\n\n[Environment execution time: {env_time:.4f}s]"
        observations.append(obs_msg)
    
    return system_prompt, first_user_msg, observations, total_env_time

def process_single_trajectory(traj_id, data, num_turns):
    """处理单条轨迹，返回详细指标"""
    try:
        start_time = time.time()
        state = multiturn_generate.run(
            system_msg=data["system_msg"],
            first_user_msg=data["first_user_msg"],
            observations=data["observations"],
            num_turns=num_turns
        )
        elapsed = time.time() - start_time
        
        # 统计生成的tokens
        total_tokens = 0
        turn_tokens = []
        for i in range(num_turns):
            try:
                resp = state[f"turn_{i}"]
                if resp:
                    tokens = len(resp.split())
                    total_tokens += tokens
                    turn_tokens.append(tokens)
                else:
                    turn_tokens.append(0)
            except:
                turn_tokens.append(0)
        
        return {
            "success": True,
            "traj_id": traj_id,
            "time": elapsed,
            "tokens": total_tokens,
            "turn_tokens": turn_tokens,
            "env_time": data["env_time"]
        }
    except Exception as e:
        return {
            "success": False,
            "traj_id": traj_id,
            "error": str(e)[:200]
        }

def benchmark_concurrent_with_metrics(runtime, trajs, batch_size, config, num_turns, metadata_file):
    """
    并发测试，记录详细的时间和token指标
    """
    log(f"\n{'='*70}")
    log(f"测试 Batch Size (并发数): {batch_size}")
    log(f"{'='*70}")
    
    # 准备所有轨迹数据
    log("准备数据...")
    all_data = []
    total_env_time = 0
    
    for idx, traj in enumerate(trajs):
        system_msg, first_user_msg, observations, env_time = prepare_trajectory_data(
            traj, config, num_turns
        )
        all_data.append({
            "system_msg": system_msg,
            "first_user_msg": first_user_msg,
            "observations": observations,
            "env_time": env_time
        })
        total_env_time += env_time
        
        if (idx+1) % 10 == 0:
            log(f"  准备: {idx+1}/{len(trajs)}")
    
    avg_env_time = total_env_time / len(all_data)
    log(f"✓ 数据准备完成")
    log(f"  平均env时间: {avg_env_time:.2f}秒/轨迹")
    
    # 开始并发处理，记录详细指标
    log(f"开始并发推理（并发数={batch_size}）...")
    start_time = time.time()
    
    processed = 0
    failed = 0
    total_tokens = 0
    
    # 用于追踪实时指标
    metrics_lock = Lock()
    time_series = []  # 时间序列数据
    
    # 打开metadata文件准备写入
    with open(metadata_file, 'a') as f_meta:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            # 提交所有任务
            futures = {
                executor.submit(process_single_trajectory, idx, data, num_turns): idx
                for idx, data in enumerate(all_data)
            }
            
            # 按完成顺序处理结果
            for future in as_completed(futures):
                result = future.result()
                current_time = time.time()
                elapsed = current_time - start_time
                
                if result["success"]:
                    processed += 1
                    traj_tokens = result["tokens"]
                    total_tokens += traj_tokens
                    
                    # 计算当前指标
                    current_throughput = processed / elapsed if elapsed > 0 else 0
                    token_throughput = total_tokens / elapsed if elapsed > 0 else 0
                    
                    # 计算前缀长度（近似：已完成轨迹的累积token数）
                    prefix_length = total_tokens
                    
                    # 记录时间序列数据
                    with metrics_lock:
                        time_point = {
                            "batch_size": batch_size,
                            "timestamp": elapsed,
                            "completed_trajs": processed,
                            "total_tokens_generated": total_tokens,
                            "trajectory_throughput": current_throughput,
                            "token_throughput": token_throughput,
                            "prefix_length_approx": prefix_length,
                            "traj_id": result["traj_id"],
                            "traj_time": result["time"],
                            "traj_tokens": traj_tokens,
                            "traj_env_time": result["env_time"]
                        }
                        time_series.append(time_point)
                        
                        # 实时写入metadata
                        f_meta.write(json.dumps(time_point) + '\n')
                        f_meta.flush()
                    
                    # 输出进度
                    if processed % 10 == 0:
                        log(f"  进度: {processed}/{len(all_data)} "
                            f"({100*processed/len(all_data):.0f}%) "
                            f"速度: {current_throughput:.2f}条/秒 "
                            f"Token吞吐: {token_throughput:.2f}tok/秒 "
                            f"用时: {elapsed:.1f}秒")
                else:
                    failed += 1
                    if failed <= 3:
                        log(f"  失败 #{failed}: {result['error']}")
    
    total_time = time.time() - start_time
    
    result_summary = {
        "batch_size": batch_size,
        "num_turns": num_turns,
        "num_trajectories": len(trajs),
        "processed": processed,
        "failed": failed,
        "total_time": total_time,
        "throughput_traj_per_sec": processed/total_time if total_time > 0 else 0,
        "throughput_token_per_sec": total_tokens/total_time if total_time > 0 else 0,
        "avg_time_per_traj": total_time/max(processed,1),
        "total_tokens": total_tokens,
        "avg_tokens_per_traj": total_tokens/max(processed,1),
        "avg_tokens_per_turn": total_tokens/max(processed,1)/num_turns if num_turns > 0 else 0,
        "avg_env_time": avg_env_time,
        "total_env_time": total_env_time,
        "time_series_points": len(time_series)
    }
    
    log(f"\n结果:")
    log(f"  成功: {processed}/{len(trajs)}")
    log(f"  总耗时: {total_time:.2f}秒")
    log(f"  轨迹吞吐: {result_summary['throughput_traj_per_sec']:.2f}条/秒")
    log(f"  Token吞吐: {result_summary['throughput_token_per_sec']:.2f}tok/秒")
    log(f"  每条耗时: {result_summary['avg_time_per_traj']:.2f}秒")
    log(f"  总tokens: {total_tokens}")
    log(f"  平均: {result_summary['avg_tokens_per_traj']:.0f} tokens/轨迹")
    log(f"  平均: {result_summary['avg_tokens_per_turn']:.1f} tokens/轮")
    log(f"  Env时间: {avg_env_time:.2f}秒/轨迹")
    log(f"  时间序列点数: {len(time_series)}")
    log("="*70)
    
    return result_summary

def main():
    if __name__ != '__main__':
        return
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='带指标追踪的并发基准测试')
    parser.add_argument('--gpu', type=int, default=1, help='使用的GPU编号')
    parser.add_argument('--turns', type=int, default=50, help='对话轮数')
    parser.add_argument('--num_traj', type=int, default=50, help='轨迹数量')
    args = parser.parse_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    log("="*70)
    log("带指标追踪的并发基准测试")
    log("="*70)
    log(f"模型: Qwen2.5-14B (128k上下文)")
    log(f"配置: GPU {args.gpu}, {args.num_traj}条轨迹, {args.turns}轮对话")
    log(f"Batch Size: 2, 3, 4, 6, 8")
    log("="*70)
    
    # 加载
    log("\n[1/3] 加载配置和数据...")
    config = load_config()
    trajs = load_trajectories(args.num_traj)
    log(f"✓ 加载了 {len(trajs)} 条轨迹")
    log(f"  每条约 {len(trajs[0].get('trajectory_steps', []))} 步")
    
    # 加载模型
    log(f"\n[2/3] 加载模型到GPU {args.gpu}...")
    log("配置: 优化显存利用")
    start = time.time()
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.88,  # 提高到88%以充分利用显存
        max_total_tokens=196608    # 192k tokens (提高50%以利用更多显存)
    )
    sgl.set_default_backend(runtime)
    load_time = time.time() - start
    log(f"✓ 模型加载完成! {load_time:.1f}秒")
    
    log(f"\nGPU {args.gpu}显存使用:")
    os.system(f"nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | grep '^{args.gpu},'")
    
    # 优化说明
    log("\n💡 优化配置:")
    log("  • mem_fraction=0.88: 提高显存利用率 (vs 默认0.8)")
    log("  • max_total_tokens=192k: 50%更大的KV cache (vs 128k)")
    log("  • SGLang自动prefix caching: 多轮对话自动复用共享前缀")
    
    # 测试不同batch size
    log("\n[3/3] 测试不同Batch Size (并发数)...")
    
    num_turns = args.turns
    batch_sizes = [2, 3, 4, 6, 8]  # 只测这5个
    all_results = []
    
    # Metadata文件
    metadata_file = f"metrics_gpu{args.gpu}_{args.num_traj}traj_{num_turns}turns.jsonl"
    log(f"Metadata保存到: {metadata_file}")
    
    # 清空metadata文件
    with open(metadata_file, 'w') as f:
        pass
    
    for batch_size in batch_sizes:
        try:
            log(f"\n{'>'*70}")
            log(f"开始测试 Batch Size = {batch_size}")
            result = benchmark_concurrent_with_metrics(
                runtime, trajs, batch_size, config, num_turns, metadata_file
            )
            all_results.append(result)
            time.sleep(2)
        except Exception as e:
            log(f"✗ Batch Size {batch_size} 失败: {str(e)[:150]}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总
    log("\n\n" + "="*70)
    log("测试汇总 - 不同Batch Size的影响")
    log("="*70)
    print(f"{'并发':<6} {'总耗时':<12} {'轨迹吞吐':<14} {'Token吞吐':<14} {'总Tokens':<12} {'Tok/轮':<10} {'Env时间':<10}")
    print(f"{'数':<6} {'(秒)':<12} {'(条/秒)':<14} {'(tok/秒)':<14} {'':<12} {'':<10} {'(秒)':<10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['batch_size']:<6} "
              f"{r['total_time']:<12.2f} "
              f"{r['throughput_traj_per_sec']:<14.2f} "
              f"{r['throughput_token_per_sec']:<14.2f} "
              f"{r['total_tokens']:<12} "
              f"{r['avg_tokens_per_turn']:<10.1f} "
              f"{r['avg_env_time']:<10.2f}")
    log("="*70)
    
    # 找出最优
    if all_results:
        best_traj = max(all_results, key=lambda x: x['throughput_traj_per_sec'])
        best_token = max(all_results, key=lambda x: x['throughput_token_per_sec'])
        
        log(f"\n🎯 最优配置:")
        log(f"  轨迹吞吐最高: Batch Size={best_traj['batch_size']}, {best_traj['throughput_traj_per_sec']:.2f}条/秒")
        log(f"  Token吞吐最高: Batch Size={best_token['batch_size']}, {best_token['throughput_token_per_sec']:.2f}tok/秒")
    
    # 保存汇总结果
    output_file = f"summary_gpu{args.gpu}_{args.num_traj}traj_{num_turns}turns.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": "Qwen2.5-14B",
            "model_path": MODEL_PATH,
            "gpu": f"GPU {args.gpu}",
            "num_trajectories": args.num_traj,
            "num_turns": num_turns,
            "max_context": "128k tokens",
            "model_load_time": load_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "note": "Concurrent with detailed metrics tracking",
            "metadata_file": metadata_file,
            "results": all_results
        }, f, indent=2)
    log(f"\n汇总结果已保存: {output_file}")
    log(f"详细指标已保存: {metadata_file}")
    log(f"  包含 {sum(r['time_series_points'] for r in all_results)} 个时间点数据")
    
    runtime.shutdown()
    log("\n✓ 所有测试完成!")

if __name__ == '__main__':
    main()

