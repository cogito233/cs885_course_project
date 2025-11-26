"""
最终正确版本 - 多轮对话性能测试
- 使用GPU 3
- Qwen3-14B-Base模型
- 50条轨迹 × 50轮对话
- Batch Size = 128
- 强制generate每轮，正确统计tokens
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


def prepare_trajectory_data(trajectory, system_prompt, instance_prompt_template, num_turns=50):
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
        next_user_msg = f"{observation}\n\n[Environment execution time: {env_time:.4f}s]"
        
        turn_data.append({
            "next_user_msg": next_user_msg,
            "env_time": env_time
        })
    
    return initial_context, turn_data, total_env_time


@sgl.function
def multiturn_generation(s, initial_context, turn_data):
    """多轮对话生成 - 强制generate每轮"""
    # 添加初始上下文
    for msg in initial_context:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
    
    # 逐轮生成
    for i, turn in enumerate(turn_data):
        # 强制生成assistant回复
        s += sgl.assistant(sgl.gen(
            f"turn_{i}",
            max_tokens=256,
            temperature=0.7,
            stop=None
        ))
        
        # 添加下一个user消息
        s += sgl.user(turn["next_user_msg"])


def benchmark(runtime, trajectories, batch_size, config, num_turns=50):
    """执行性能测试"""
    print(f"\n{'='*70}")
    print(f"性能测试: Batch Size={batch_size}, 轮数={num_turns}")
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
    
    print(f"准备了 {len(all_data)} 条轨迹")
    print(f"平均每条 {avg_turns:.1f} 轮对话")
    print(f"平均env时间: {avg_env_time:.2f}秒/轨迹")
    
    # 开始处理
    print(f"\n开始处理...")
    start_time = time.time()
    processed = 0
    failed = 0
    total_gen_tokens = 0
    total_gen_chars = 0
    
    # 分批处理
    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i:i+batch_size]
        
        for idx, data in enumerate(batch_data):
            try:
                state = multiturn_generation.run(
                    initial_context=data["initial_context"],
                    turn_data=data["turn_data"]
                )
                
                processed += 1
                
                # 正确统计生成的tokens
                batch_tokens = 0
                batch_chars = 0
                for j in range(len(data["turn_data"])):
                    key = f"turn_{j}"
                    try:
                        # 正确的方式：使用索引访问
                        if hasattr(state, '__getitem__'):
                            response = state[key]
                            if response:
                                tokens = len(response.split())
                                batch_tokens += tokens
                                batch_chars += len(response)
                    except:
                        pass
                
                total_gen_tokens += batch_tokens
                total_gen_chars += batch_chars
                
                # 输出详细进度
                if processed % 5 == 0 or processed == 1:
                    elapsed = time.time() - start_time
                    print(f"  进度: {processed}/{len(all_data)} "
                          f"({100*processed/len(all_data):.1f}%), "
                          f"速度: {processed/elapsed:.2f} 条/秒, "
                          f"已用时: {elapsed:.1f}秒, "
                          f"本轨迹生成: {batch_tokens} tokens")
                        
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  错误: {str(e)[:100]}")
                continue
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"测试结果:")
    print(f"  轨迹数: {len(trajectories)}")
    print(f"  成功: {processed}")
    print(f"  失败: {failed}")
    print(f"  轮数/轨迹: {num_turns}")
    print(f"  ")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  吞吐量: {processed/total_time:.2f} 条/秒")
    print(f"  每条轨迹耗时: {total_time/max(processed,1):.2f} 秒")
    print(f"  ")
    print(f"  生成统计:")
    print(f"    总生成tokens: {total_gen_tokens}")
    print(f"    总生成字符: {total_gen_chars}")
    print(f"    平均tokens/轨迹: {total_gen_tokens/max(processed,1):.1f}")
    print(f"    平均tokens/轮: {total_gen_tokens/max(processed,1)/num_turns:.1f}")
    print(f"    平均字符/轨迹: {total_gen_chars/max(processed,1):.1f}")
    print(f"  ")
    print(f"  环境执行时间 (来自原始数据):")
    print(f"    平均env时间/轨迹: {avg_env_time:.2f} 秒")
    print(f"    总env时间: {total_env_time_all:.2f} 秒")
    print(f"{'='*70}\n")
    
    return {
        "batch_size": batch_size,
        "num_turns": num_turns,
        "num_trajectories": len(trajectories),
        "processed": processed,
        "failed": failed,
        "total_time": total_time,
        "throughput": processed/total_time if total_time > 0 else 0,
        "avg_time_per_trajectory": total_time/max(processed,1),
        "total_gen_tokens": total_gen_tokens,
        "total_gen_chars": total_gen_chars,
        "avg_tokens_per_trajectory": total_gen_tokens/max(processed,1),
        "avg_tokens_per_turn": total_gen_tokens/max(processed,1)/num_turns if num_turns > 0 else 0,
        "avg_env_time_per_trajectory": avg_env_time,
        "total_env_time": total_env_time_all
    }


def main():
    print("=" * 70)
    print("最终正确版本 - 多轮对话性能测试")
    print("=" * 70)
    print(f"模型: Qwen3-14B-Base")
    print(f"GPU: GPU 3")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 加载配置和数据
    print("[1/4] 加载配置和数据...")
    config = load_config()
    num_trajectories = 50
    trajectories = load_trajectories(num_trajectories)
    sample_steps = len(trajectories[0].get('trajectory_steps', []))
    print(f"✓ 加载了 {len(trajectories)} 条轨迹")
    print(f"  原始数据约 {sample_steps} 步/轨迹\n")
    
    # 加载模型
    print("[2/4] 加载模型...")
    start_time = time.time()
    
    try:
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.75,
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start_time
        print(f"✓ 模型加载完成！耗时: {load_time:.2f}秒\n")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return
    
    # 检查GPU
    print("[3/4] GPU使用情况:")
    os.system("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4")
    print()
    
    # 执行测试
    print("[4/4] 开始测试...")
    print(f"配置: {num_trajectories}条轨迹 × 50轮对话 × Batch Size 128\n")
    
    num_turns = 50
    batch_size = 128
    
    result = benchmark(runtime, trajectories, batch_size, config, num_turns)
    
    # 输出关键指标
    print("\n" + "=" * 70)
    print("关键性能指标")
    print("=" * 70)
    print(f"完成: {result['processed']} 条轨迹 × {num_turns} 轮")
    print(f"总耗时: {result['total_time']:.2f} 秒")
    print(f"吞吐量: {result['throughput']:.2f} 条/秒")
    print(f"每条耗时: {result['avg_time_per_trajectory']:.2f} 秒")
    print(f"")
    print(f"生成统计:")
    print(f"  总tokens: {result['total_gen_tokens']}")
    print(f"  平均/轨迹: {result['avg_tokens_per_trajectory']:.1f} tokens")
    print(f"  平均/轮: {result['avg_tokens_per_turn']:.1f} tokens")
    print(f"")
    print(f"环境时间: {result['avg_env_time_per_trajectory']:.2f} 秒/轨迹")
    print("=" * 70)
    
    # 保存结果
    output_file = f"final_results_{num_trajectories}traj_{num_turns}turns_bs{batch_size}_qwen3.json"
    with open(output_file, 'w') as f:
        json.dump({
            "test_info": {
                "model_path": MODEL_PATH,
                "model_name": "Qwen3-14B-Base",
                "gpu_device": "GPU 3",
                "num_trajectories": num_trajectories,
                "num_turns": num_turns,
                "batch_size": batch_size,
                "model_load_time": load_time,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "result": result
        }, f, indent=2)
    print(f"\n结果已保存: {output_file}")
    
    # 关闭
    runtime.shutdown()
    print("✓ 完成！")


if __name__ == "__main__":
    main()

