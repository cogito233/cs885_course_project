"""
渐进式测试版本 - 从少到多测试轮数
先测试5轮、10轮、20轮，确认能跑通后再测试50轮
使用Qwen3-14B-Base，GPU 3，Batch Size 128，50条轨迹
"""

import sglang as sgl
import os
import time
import json
import yaml


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen3-14B-Base"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"


def load_config():
    with open(YAML_FILE, 'r') as f:
        return yaml.safe_load(f)


def load_trajectories(num_trajectories=50):
    trajectories = []
    with open(JSONL_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_trajectories:
                break
            trajectories.append(json.loads(line))
    return trajectories


def prepare_trajectory_data(trajectory, system_prompt, instance_prompt_template, num_turns):
    """准备单条轨迹的数据"""
    initial_context = []
    initial_context.append({
        "role": "system",
        "content": system_prompt[:2000]
    })
    
    problem_statement = trajectory.get('problem_statement', '')[:1000]
    first_user_msg = instance_prompt_template.format(problem_statement=problem_statement)
    initial_context.append({
        "role": "user",
        "content": first_user_msg[:1500]
    })
    
    steps = trajectory.get('trajectory_steps', [])[:num_turns]
    turn_data = []
    total_env_time = 0
    
    for step in steps:
        env_time = float(step.get('env_exec_time', 0))
        total_env_time += env_time
        
        observation = step.get('observation', '')[:800]
        next_user_msg = f"{observation}\n\n[Env time: {env_time:.4f}s]"
        
        turn_data.append({
            "next_user_msg": next_user_msg,
            "env_time": env_time
        })
    
    return initial_context, turn_data, total_env_time


@sgl.function
def multiturn_generation(s, initial_context, turn_data):
    """多轮对话生成"""
    for msg in initial_context:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
    
    for i, turn in enumerate(turn_data):
        s += sgl.assistant(sgl.gen(
            f"turn_{i}",
            max_tokens=256,
            temperature=0.7
        ))
        s += sgl.user(turn["next_user_msg"])


def benchmark(runtime, trajectories, batch_size, config, num_turns):
    """执行性能测试"""
    print(f"\n{'='*70}")
    print(f"测试: {num_turns}轮对话, Batch Size={batch_size}")
    print(f"{'='*70}")
    
    system_prompt = config['system_prompt']
    instance_prompt_template = config['instance_prompt']
    
    # 准备数据
    all_data = []
    total_env_time_all = 0
    
    for traj in trajectories:
        initial_ctx, turn_data, env_time = prepare_trajectory_data(
            traj, system_prompt, instance_prompt_template, num_turns=num_turns
        )
        all_data.append({
            "initial_context": initial_ctx,
            "turn_data": turn_data,
            "env_time": env_time
        })
        total_env_time_all += env_time
    
    avg_turns = sum(len(d["turn_data"]) for d in all_data) / len(all_data)
    avg_env_time = total_env_time_all / len(all_data)
    
    print(f"准备: {len(all_data)}条轨迹, 平均{avg_turns:.1f}轮")
    print(f"平均env时间: {avg_env_time:.2f}秒/轨迹")
    
    # 开始处理
    print(f"开始处理...")
    start_time = time.time()
    processed = 0
    failed = 0
    total_gen_tokens = 0
    
    # 分批处理
    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i:i+batch_size]
        
        for data in batch_data:
            try:
                state = multiturn_generation.run(
                    initial_context=data["initial_context"],
                    turn_data=data["turn_data"]
                )
                
                processed += 1
                
                # 统计tokens
                batch_tokens = 0
                for j in range(len(data["turn_data"])):
                    try:
                        response = state[f"turn_{j}"]
                        if response:
                            batch_tokens += len(response.split())
                    except:
                        pass
                
                total_gen_tokens += batch_tokens
                
                # 每10条输出一次
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {processed}/{len(all_data)} "
                          f"({100*processed/len(all_data):.0f}%), "
                          f"{processed/elapsed:.1f}条/秒, "
                          f"用时{elapsed:.1f}秒")
                        
            except Exception as e:
                failed += 1
                if failed == 1:
                    print(f"  错误: {e}")
                continue
    
    total_time = time.time() - start_time
    
    result = {
        "num_turns": num_turns,
        "batch_size": batch_size,
        "processed": processed,
        "total_time": total_time,
        "throughput": processed/total_time,
        "avg_time_per_traj": total_time/max(processed,1),
        "total_tokens": total_gen_tokens,
        "avg_tokens_per_traj": total_gen_tokens/max(processed,1),
        "avg_tokens_per_turn": total_gen_tokens/max(processed,1)/num_turns,
        "avg_env_time": avg_env_time,
        "total_env_time": total_env_time_all
    }
    
    print(f"\n结果: {num_turns}轮")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  吞吐量: {result['throughput']:.2f}条/秒")
    print(f"  每条耗时: {result['avg_time_per_traj']:.2f}秒")
    print(f"  总tokens: {total_gen_tokens}")
    print(f"  平均: {result['avg_tokens_per_traj']:.0f} tokens/轨迹")
    print(f"  平均: {result['avg_tokens_per_turn']:.1f} tokens/轮")
    print(f"  Env时间: {avg_env_time:.2f}秒/轨迹")
    
    return result


def main():
    print("=" * 70)
    print("渐进式性能测试 - 从少到多")
    print("=" * 70)
    print(f"模型: Qwen3-14B-Base | GPU: 3 | Batch Size: 128")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 加载
    print("[1/3] 加载配置和数据...")
    config = load_config()
    num_trajectories = 50
    trajectories = load_trajectories(num_trajectories)
    print(f"✓ {len(trajectories)}条轨迹\n")
    
    # 加载模型
    print("[2/3] 加载模型...")
    start_time = time.time()
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.2,  # 降低到20%以适应有限显存
    )
    sgl.set_default_backend(runtime)
    load_time = time.time() - start_time
    print(f"✓ 完成! {load_time:.1f}秒")
    
    # GPU检查
    print("\nGPU 3显存:")
    os.system("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^3,'")
    
    # 渐进测试
    print("\n[3/3] 渐进式测试...")
    batch_size = 128
    
    # 测试不同轮数
    test_turns = [5, 10, 20, 50]
    all_results = []
    
    for num_turns in test_turns:
        print(f"\n{'='*70}")
        print(f"▶ 测试 {num_turns} 轮对话")
        print(f"{'='*70}")
        
        try:
            result = benchmark(runtime, trajectories, batch_size, config, num_turns)
            all_results.append(result)
            
            # 休息一下
            time.sleep(1)
        except Exception as e:
            print(f"✗ {num_turns}轮测试失败: {e}")
            break
    
    # 汇总
    print(f"\n\n{'='*70}")
    print("渐进式测试汇总")
    print(f"{'='*70}")
    print(f"{'轮数':<8} {'总耗时(秒)':<14} {'吞吐量':<12} {'平均tokens':<14} {'Env时间(秒)':<14}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['num_turns']:<8} "
              f"{r['total_time']:<14.2f} "
              f"{r['throughput']:<12.2f} "
              f"{r['avg_tokens_per_traj']:<14.0f} "
              f"{r['avg_env_time']:<14.2f}")
    print("=" * 70)
    
    # 保存
    output_file = f"progressive_results_bs{batch_size}_qwen3.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": "Qwen3-14B-Base",
            "gpu": "GPU 3",
            "batch_size": batch_size,
            "num_trajectories": num_trajectories,
            "model_load_time": load_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": all_results
        }, f, indent=2)
    print(f"\n结果已保存: {output_file}")
    
    # 关闭
    runtime.shutdown()
    print("\n✓ 所有测试完成！")


if __name__ == "__main__":
    main()

