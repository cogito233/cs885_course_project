"""
完整性能测试
- GPU 2
- Qwen3-14B-Base
- 50条轨迹 × 50轮对话
- Batch Size 128
- 强制generate每轮assistant回复
"""

import sglang as sgl
import os
import time
import json
import yaml


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen3-14B-Base"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"


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
    print(f"\n{'='*70}")
    print(f"测试: {num_turns}轮 × {len(trajs)}条 × Batch Size {batch_size}")
    print(f"{'='*70}")
    
    system_prompt = config['system_prompt']
    instance_template = config['instance_prompt']
    
    # 准备数据
    all_data = []
    total_env_time = 0
    
    for traj in trajs:
        problem = traj.get('problem_statement', '')[:500]
        first_msg = f"Problem: {problem}"
        
        user_msgs = [first_msg]
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
    
    avg_env = total_env_time / len(all_data)
    print(f"准备完成, 平均env时间: {avg_env:.2f}秒/轨迹")
    
    # 开始处理
    print(f"开始处理...")
    start_time = time.time()
    processed = 0
    total_tokens = 0
    
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        
        for data in batch:
            try:
                state = multiturn_gen.run(
                    user_msgs=data["user_msgs"],
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
                    print(f"  {processed}/{len(all_data)} "
                          f"({100*processed/len(all_data):.0f}%) "
                          f"{processed/elapsed:.1f}条/秒 "
                          f"{elapsed:.1f}秒")
                        
            except Exception as e:
                if processed == 0:
                    print(f"  错误: {str(e)[:150]}")
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
    
    print(f"\n结果:")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  吞吐量: {result['throughput']:.2f}条/秒")
    print(f"  每条耗时: {result['avg_time_per_traj']:.2f}秒")
    print(f"  总tokens: {total_tokens}")
    print(f"  平均: {result['avg_tokens_per_traj']:.0f} tokens/轨迹")
    print(f"  平均: {result['avg_tokens_per_turn']:.1f} tokens/轮")
    print(f"  Env时间: {avg_env:.2f}秒/轨迹")
    print(f"{'='*70}")
    
    return result


def main():
    print("=" * 70)
    print("完整性能测试: 50轨迹 × 50轮 × BS128")
    print("GPU 2 | Qwen3-14B-Base")
    print("=" * 70)
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 加载
    print("[1/3] 加载...")
    config = load_config()
    trajs = load_trajectories(50)
    print(f"✓ {len(trajs)}条轨迹\n")
    
    # 加载模型
    print("[2/3] 加载模型...")
    start = time.time()
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.8,
        max_total_tokens=8192,
    )
    sgl.set_default_backend(runtime)
    load_time = time.time() - start
    print(f"✓ 完成! {load_time:.1f}秒")
    
    print("\nGPU 2显存:")
    os.system("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^2,'")
    
    # 测试
    print(f"\n[3/3] 性能测试...")
    
    all_results = []
    
    # 渐进测试: 5轮 -> 10轮 -> 20轮 -> 50轮
    for num_turns in [5, 10, 20, 50]:
        try:
            result = benchmark(runtime, trajs, 128, config, num_turns)
            all_results.append(result)
            time.sleep(1)
        except Exception as e:
            print(f"\n✗ {num_turns}轮失败: {str(e)[:150]}")
            break
    
    # 汇总
    print(f"\n\n{'='*70}")
    print("完整测试汇总")
    print("="*70)
    print(f"{'轮数':<8} {'总耗时(秒)':<14} {'吞吐量':<12} {'总Tokens':<12} {'平均Tok/轮':<12} {'Env时间(秒)':<12}")
    print("-"*70)
    for r in all_results:
        print(f"{r['num_turns']:<8} "
              f"{r['total_time']:<14.2f} "
              f"{r['throughput']:<12.2f} "
              f"{r['total_tokens']:<12} "
              f"{r['avg_tokens_per_turn']:<12.1f} "
              f"{r['avg_env_time']:<12.2f}")
    print("="*70)
    
    # 重点展示50轮结果
    if all_results and all_results[-1]['num_turns'] == 50:
        r = all_results[-1]
        print(f"\n🎯 50轮对话关键指标:")
        print(f"  完成50条×50轮: {r['total_time']:.2f}秒")
        print(f"  吞吐量: {r['throughput']:.2f}条/秒")
        print(f"  每条轨迹: {r['avg_time_per_traj']:.2f}秒")
        print(f"  总生成: {r['total_tokens']} tokens")
        print(f"  平均: {r['avg_tokens_per_traj']:.0f} tokens/轨迹")
        print(f"  平均: {r['avg_tokens_per_turn']:.1f} tokens/轮")
        print(f"  Env时间: {r['avg_env_time']:.2f}秒/轨迹")
    
    # 保存
    with open("benchmark_50x50_bs128_gpu2_qwen3.json", 'w') as f:
        json.dump({
            "model": "Qwen3-14B-Base",
            "gpu": "GPU 2",
            "batch_size": 128,
            "num_trajectories": 50,
            "model_load_time": load_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": all_results
        }, f, indent=2)
    print(f"\n结果已保存: benchmark_50x50_bs128_gpu2_qwen3.json")
    
    # 关闭
    runtime.shutdown()
    print("\n✓ 所有测试完成!")


if __name__ == "__main__":
    main()

