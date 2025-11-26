"""
最终版本 - 详细进度输出
GPU 2, Qwen3-14B-Base
50条轨迹 × 50轮对话 × Batch Size 128
强制generate每轮assistant回复
"""

import sglang as sgl
import os
import time
import json
import yaml
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen3-14B-Base"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"


def log(msg):
    """带时间戳的日志"""
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()


def load_config():
    with open(YAML_FILE, 'r') as f:
        return yaml.safe_load(f)


def load_trajectories(num=50):
    trajs = []
    with open(JSONL_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            trajs.append(json.loads(line))
    return trajs


@sgl.function
def multiturn_gen(s, user_msgs, num_turns):
    """多轮对话生成"""
    for i in range(num_turns):
        if i < len(user_msgs):
            s += sgl.user(user_msgs[i])
        s += sgl.assistant(sgl.gen(f"turn_{i}", max_tokens=128, temperature=0.7))


def benchmark(runtime, trajs, batch_size, config, num_turns):
    """性能测试"""
    log(f"='*70")
    log(f"测试: {num_turns}轮 × {len(trajs)}条 × BS{batch_size}")
    log("="*70)
    
    # 准备数据
    log("准备数据...")
    all_data = []
    total_env_time = 0
    
    for idx, traj in enumerate(trajs):
        problem = traj.get('problem_statement', '')[:500]
        user_msgs = [f"Problem: {problem}"]
        traj_env_time = 0
        
        steps = traj.get('trajectory_steps', [])[:num_turns]
        for step in steps:
            env_time = float(step.get('env_exec_time', 0))
            traj_env_time += env_time
            obs = step.get('observation', '')[:200]
            user_msgs.append(f"{obs}...\n[Env: {env_time:.3f}s]")
        
        all_data.append({
            "user_msgs": user_msgs,
            "env_time": traj_env_time
        })
        total_env_time += traj_env_time
        
        # 准备数据的进度
        if (idx + 1) % 10 == 0:
            log(f"  准备数据: {idx+1}/{len(trajs)}")
    
    avg_env = total_env_time / len(all_data)
    log(f"✓ 数据准备完成, 平均env: {avg_env:.2f}秒/轨迹")
    
    # 开始处理
    log(f"开始推理...")
    start_time = time.time()
    processed = 0
    total_tokens = 0
    
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        batch_start = time.time()
        
        log(f"  处理批次 {i//batch_size + 1}/{(len(all_data)-1)//batch_size + 1}")
        
        for batch_idx, data in enumerate(batch):
            try:
                iter_start = time.time()
                state = multiturn_gen.run(
                    user_msgs=data["user_msgs"],
                    num_turns=num_turns
                )
                iter_time = time.time() - iter_start
                processed += 1
                
                # 统计tokens
                traj_tokens = 0
                for j in range(num_turns):
                    try:
                        resp = state[f"turn_{j}"]
                        if resp:
                            traj_tokens += len(resp.split())
                    except:
                        pass
                
                total_tokens += traj_tokens
                
                # 详细进度
                elapsed = time.time() - start_time
                if processed % 5 == 0:
                    log(f"    {processed}/{len(all_data)} "
                          f"({100*processed/len(all_data):.0f}%) "
                          f"速度:{processed/elapsed:.1f}条/秒 "
                          f"用时:{elapsed:.1f}秒 "
                          f"本条:{traj_tokens}tok/{iter_time:.2f}秒")
                        
            except Exception as e:
                log(f"    错误 #{processed}: {str(e)[:100]}")
                continue
    
    total_time = time.time() - start_time
    
    result = {
        "num_turns": num_turns,
        "batch_size": batch_size,
        "num_trajectories": len(trajs),
        "processed": processed,
        "total_time": total_time,
        "throughput": processed/total_time,
        "avg_time_per_traj": total_time/max(processed,1),
        "total_tokens": total_tokens,
        "avg_tokens_per_traj": total_tokens/max(processed,1),
        "avg_tokens_per_turn": total_tokens/max(processed,1)/num_turns,
        "avg_env_time": avg_env,
        "total_env_time": total_env_time
    }
    
    log("\n结果:")
    log(f"  总耗时: {total_time:.2f}秒")
    log(f"  吞吐量: {result['throughput']:.2f}条/秒")
    log(f"  每条耗时: {result['avg_time_per_traj']:.2f}秒")
    log(f"  总tokens: {total_tokens}")
    log(f"  平均: {result['avg_tokens_per_traj']:.0f} tokens/轨迹")
    log(f"  平均: {result['avg_tokens_per_turn']:.1f} tokens/轮")
    log(f"  Env时间: {avg_env:.2f}秒/轨迹")
    log("="*70)
    
    return result


def main():
    log("="*70)
    log("完整性能测试: 50轨迹 × 50轮 × BS128")
    log("GPU 2 | Qwen3-14B-Base")
    log("="*70)
    
    # 加载
    log("[1/3] 加载配置和数据...")
    config = load_config()
    trajs = load_trajectories(50)
    log(f"✓ {len(trajs)}条轨迹")
    
    # 加载模型
    log("\n[2/3] 加载模型到GPU 2...")
    start = time.time()
    
    try:
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.8,
            max_total_tokens=8192,
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start
        log(f"✓ 模型加载完成! {load_time:.1f}秒")
    except Exception as e:
        log(f"✗ 模型加载失败: {e}")
        return
    
    log("\nGPU 2显存:")
    os.system("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^2,'")
    
    # 测试
    log("\n[3/3] 开始性能测试...")
    log("测试顺序: 5轮 -> 10轮 -> 20轮 -> 50轮")
    
    all_results = []
    
    for num_turns in [5, 10, 20, 50]:
        try:
            log(f"\n▶ 开始测试 {num_turns} 轮...")
            result = benchmark(runtime, trajs, 128, config, num_turns)
            all_results.append(result)
            log(f"✓ {num_turns}轮完成!")
            time.sleep(2)
        except Exception as e:
            log(f"✗ {num_turns}轮失败: {str(e)[:150]}")
            import traceback
            traceback.print_exc()
            break
    
    # 汇总
    log("\n\n" + "="*70)
    log("测试汇总")
    log("="*70)
    print(f"{'轮数':<8} {'总耗时(秒)':<14} {'吞吐量':<12} {'总Tokens':<12} {'Tok/轮':<10} {'Env(秒)':<10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['num_turns']:<8} "
              f"{r['total_time']:<14.2f} "
              f"{r['throughput']:<12.2f} "
              f"{r['total_tokens']:<12} "
              f"{r['avg_tokens_per_turn']:<10.1f} "
              f"{r['avg_env_time']:<10.2f}")
    log("="*70)
    
    # 保存
    output_file = "final_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": "Qwen3-14B-Base",
            "gpu": "GPU 2",
            "batch_size": 128,
            "num_trajectories": 50,
            "model_load_time": load_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": all_results
        }, f, indent=2)
    log(f"\n结果已保存: {output_file}")
    
    # 关闭
    log("\n关闭runtime...")
    runtime.shutdown()
    log("✓ 所有测试完成!")


if __name__ == "__main__":
    main()

