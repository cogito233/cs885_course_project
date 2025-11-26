"""
测试不同batch_size的影响
- GPU 2, Qwen3-14B-Base
- 50条轨迹 × 5轮对话
- 测试batch_size: 1, 2, 4, 8, 16, 32, 64, 128, 256
- 不缩短内容，保持完整的problem_statement和observation
- 每轮assistant强制generate
"""

import sglang as sgl
import os, time, json, yaml, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen3-14B-Base"
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
    """
    多轮对话生成
    system -> user(problem) -> assistant(gen) -> user(obs) -> assistant(gen) -> ...
    """
    s += sgl.system(system_msg)
    s += sgl.user(first_user_msg)
    
    for i in range(num_turns):
        # 强制generate assistant回复
        s += sgl.assistant(sgl.gen(
            f"turn_{i}",
            max_tokens=256,
            temperature=0.7,
            stop=["<|im_end|>", "\n\nUSER:"]  # 防止无限生成
        ))
        
        # 添加下一个user消息（observation）
        if i < len(observations):
            s += sgl.user(observations[i])

def prepare_trajectory_data(traj, config, num_turns):
    """
    准备单条轨迹数据 - 不缩短内容
    """
    system_prompt = config['system_prompt']
    instance_template = config['instance_prompt']
    
    # 完整的problem_statement（不缩短）
    problem_statement = traj.get('problem_statement', '')
    first_user_msg = instance_template.format(problem_statement=problem_statement)
    
    # 准备observations - 不缩短
    steps = traj.get('trajectory_steps', [])[:num_turns]
    observations = []
    total_env_time = 0
    
    for step in steps:
        env_time = float(step.get('env_exec_time', 0))
        total_env_time += env_time
        
        # 完整的observation（不缩短）
        observation = step.get('observation', '')
        obs_msg = f"{observation}\n\n[Environment execution time: {env_time:.4f}s]"
        observations.append(obs_msg)
    
    return system_prompt, first_user_msg, observations, total_env_time

def benchmark_batch_size(runtime, trajs, batch_size, config, num_turns):
    """测试单个batch_size"""
    log(f"\n{'='*70}")
    log(f"测试 Batch Size: {batch_size}")
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
    
    # 开始处理
    log("开始推理...")
    start_time = time.time()
    processed = 0
    failed = 0
    total_tokens = 0
    
    # 分批处理
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        
        for data in batch:
            try:
                state = multiturn_generate.run(
                    system_msg=data["system_msg"],
                    first_user_msg=data["first_user_msg"],
                    observations=data["observations"],
                    num_turns=num_turns
                )
                processed += 1
                
                # 统计tokens
                for j in range(num_turns):
                    try:
                        resp = state[f"turn_{j}"]
                        if resp:
                            total_tokens += len(resp.split())
                    except:
                        pass
                
                # 进度
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    log(f"  进度: {processed}/{len(all_data)} "
                        f"({100*processed/len(all_data):.0f}%) "
                        f"速度: {processed/elapsed:.2f}条/秒 "
                        f"用时: {elapsed:.1f}秒")
                        
            except Exception as e:
                failed += 1
                if failed <= 2:
                    log(f"  错误 #{failed}: {str(e)[:100]}")
                continue
    
    total_time = time.time() - start_time
    
    result = {
        "batch_size": batch_size,
        "num_turns": num_turns,
        "num_trajectories": len(trajs),
        "processed": processed,
        "failed": failed,
        "total_time": total_time,
        "throughput": processed/total_time if total_time > 0 else 0,
        "avg_time_per_traj": total_time/max(processed,1),
        "total_tokens": total_tokens,
        "avg_tokens_per_traj": total_tokens/max(processed,1),
        "avg_tokens_per_turn": total_tokens/max(processed,1)/num_turns if num_turns > 0 else 0,
        "avg_env_time": avg_env_time,
        "total_env_time": total_env_time
    }
    
    log(f"\n结果:")
    log(f"  成功: {processed}/{len(trajs)}")
    log(f"  总耗时: {total_time:.2f}秒")
    log(f"  吞吐量: {result['throughput']:.2f}条/秒")
    log(f"  每条耗时: {result['avg_time_per_traj']:.2f}秒")
    log(f"  总tokens: {total_tokens}")
    log(f"  平均: {result['avg_tokens_per_traj']:.0f} tokens/轨迹")
    log(f"  平均: {result['avg_tokens_per_turn']:.1f} tokens/轮")
    log(f"  Env时间: {avg_env_time:.2f}秒/轨迹")
    log("="*70)
    
    return result

def main():
    if __name__ != '__main__':
        return
    
    log("="*70)
    log("测试不同Batch Size的影响")
    log("5轮对话 × 50条轨迹 × 不同Batch Size")
    log("="*70)
    
    # 加载
    log("[1/3] 加载配置和数据...")
    config = load_config()
    trajs = load_trajectories(50)
    log(f"✓ 加载了 {len(trajs)} 条轨迹")
    log(f"  每条约 {len(trajs[0].get('trajectory_steps', []))} 步")
    
    # 加载模型
    log("\n[2/3] 加载模型到GPU 2...")
    start = time.time()
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.8,
        max_total_tokens=16384  # 增大上下文窗口
    )
    sgl.set_default_backend(runtime)
    load_time = time.time() - start
    log(f"✓ 模型加载完成! {load_time:.1f}秒")
    
    log("\nGPU 2显存:")
    os.system("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^2,'")
    
    # 测试不同batch size
    log("\n[3/3] 测试不同Batch Size...")
    log("测试batch_size: 1, 2, 4, 8, 16, 32, 64, 128, 256")
    
    num_turns = 5
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    all_results = []
    
    for batch_size in batch_sizes:
        try:
            result = benchmark_batch_size(runtime, trajs, batch_size, config, num_turns)
            all_results.append(result)
            time.sleep(1)
        except Exception as e:
            log(f"✗ Batch Size {batch_size} 失败: {str(e)[:150]}")
            import traceback
            traceback.print_exc()
            continue
    
    # 汇总
    log("\n\n" + "="*70)
    log("测试汇总 - 不同Batch Size的影响")
    log("="*70)
    print(f"{'BS':<6} {'总耗时(秒)':<12} {'吞吐量':<14} {'每条(秒)':<12} {'总Tokens':<12} {'Tok/轮':<10} {'Env(秒)':<10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['batch_size']:<6} "
              f"{r['total_time']:<12.2f} "
              f"{r['throughput']:<14.2f} "
              f"{r['avg_time_per_traj']:<12.2f} "
              f"{r['total_tokens']:<12} "
              f"{r['avg_tokens_per_turn']:<10.1f} "
              f"{r['avg_env_time']:<10.2f}")
    log("="*70)
    
    # 找出最优配置
    if all_results:
        best = max(all_results, key=lambda x: x['throughput'])
        log(f"\n最优Batch Size: {best['batch_size']}")
        log(f"  吞吐量: {best['throughput']:.2f}条/秒")
        log(f"  完成50条需要: {best['total_time']:.2f}秒")
    
    # 保存
    output_file = f"batch_size_comparison_{num_turns}turns.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": "Qwen3-14B-Base",
            "gpu": "GPU 2",
            "num_trajectories": 50,
            "num_turns": num_turns,
            "model_load_time": load_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "note": "Testing different batch sizes with full content (no truncation)",
            "results": all_results
        }, f, indent=2)
    log(f"\n结果已保存: {output_file}")
    
    runtime.shutdown()
    log("\n✓ 所有测试完成!")

if __name__ == '__main__':
    main()

