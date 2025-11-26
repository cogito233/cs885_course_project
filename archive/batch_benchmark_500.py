"""
批量处理脚本 - 500条轨迹完整测试
测试不同batch_size下的性能，真实推理生成
"""

import sglang as sgl
import os
import time
import json
import yaml
from typing import List, Dict


# 设置使用GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/R2EGym-7B-Agent"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"


def load_config():
    """加载YAML配置"""
    with open(YAML_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trajectories(num_trajectories=500):
    """从jsonl文件加载轨迹数据"""
    trajectories = []
    with open(JSONL_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_trajectories:
                break
            data = json.loads(line)
            trajectories.append(data)
    return trajectories


def construct_conversation_context(trajectory, system_prompt, instance_prompt_template, max_steps=5):
    """
    构造对话上下文（前N步），用于生成下一步
    """
    messages = []
    
    # System prompt
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    # 第一条user prompt（包含问题描述）
    problem_statement = trajectory.get('problem_statement', '')
    first_user_msg = instance_prompt_template.format(problem_statement=problem_statement)
    messages.append({
        "role": "user",
        "content": first_user_msg
    })
    
    # 处理trajectory steps（取前max_steps步作为上下文）
    steps = trajectory.get('trajectory_steps', [])[:max_steps]
    for step in steps:
        # Assistant: thought + action
        thought = step.get('thought', '')
        action = step.get('action', '')
        assistant_msg = f"{thought}\n\n{action}"
        messages.append({
            "role": "assistant",
            "content": assistant_msg[:1000]  # 限制长度
        })
        
        # User: observation + env_exec_time
        observation = step.get('observation', '')
        env_exec_time = step.get('env_exec_time', 0)
        user_msg = f"{observation[:800]}\n\n[Environment execution time: {env_exec_time}s]"
        messages.append({
            "role": "user",
            "content": user_msg
        })
    
    return messages


@sgl.function
def generate_next_step(s, messages):
    """
    基于对话上下文生成下一步
    """
    for msg in messages:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
        elif msg["role"] == "assistant":
            s += sgl.assistant(msg["content"])
    
    # 生成下一步回复
    s += sgl.user("Continue with the next step.")
    s += sgl.assistant(sgl.gen(
        "next_step",
        max_tokens=256,
        temperature=0.0
    ))


def benchmark_batch(runtime, trajectories, batch_size, config):
    """测试单个batch_size的性能"""
    print(f"\n{'='*70}")
    print(f"测试 Batch Size: {batch_size}")
    print(f"{'='*70}")
    
    system_prompt = config['system_prompt']
    instance_prompt_template = config['instance_prompt']
    
    # 准备所有消息
    all_messages = []
    for traj in trajectories:
        messages = construct_conversation_context(traj, system_prompt, instance_prompt_template, max_steps=3)
        all_messages.append(messages)
    
    print(f"准备了 {len(all_messages)} 条轨迹")
    avg_msg_count = sum(len(m) for m in all_messages) / len(all_messages)
    print(f"平均每条轨迹有 {avg_msg_count:.1f} 条消息")
    
    # 统计原始数据中的env_exec_time
    total_env_time = 0
    for traj in trajectories:
        for step in traj.get('trajectory_steps', [])[:3]:
            total_env_time += float(step.get('env_exec_time', 0))
    avg_env_time = total_env_time / len(trajectories) if trajectories else 0
    print(f"平均环境执行时间: {avg_env_time:.4f}秒/轨迹")
    
    # 开始批量处理
    start_time = time.time()
    processed = 0
    total_tokens = 0
    total_gen_time = 0
    
    # 分批处理
    for i in range(0, len(all_messages), batch_size):
        batch_messages = all_messages[i:i+batch_size]
        batch_start = time.time()
        
        # 处理当前批次
        batch_results = []
        for messages in batch_messages:
            try:
                gen_start = time.time()
                state = generate_next_step.run(messages=messages)
                gen_time = time.time() - gen_start
                
                processed += 1
                total_gen_time += gen_time
                
                # 统计生成的token数
                if 'next_step' in state:
                    response = state['next_step']
                    total_tokens += len(response.split())
                    batch_results.append({
                        "success": True,
                        "tokens": len(response.split()),
                        "gen_time": gen_time
                    })
            except Exception as e:
                print(f"  处理出错: {str(e)[:100]}")
                batch_results.append({"success": False})
                continue
        
        batch_time = time.time() - batch_start
        
        # 定期输出进度
        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(all_messages):
            successful = sum(1 for r in batch_results if r.get("success"))
            print(f"  进度: {processed}/{len(all_messages)} "
                  f"({100*processed/len(all_messages):.1f}%), "
                  f"批次耗时: {batch_time:.2f}秒, "
                  f"速度: {len(batch_messages)/batch_time:.2f} 条/秒")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Batch Size {batch_size} 测试结果:")
    print(f"  总轨迹数: {len(trajectories)}")
    print(f"  成功处理: {processed}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  平均速度: {processed/total_time:.2f} 条/秒")
    print(f"  每条轨迹平均耗时: {total_time/processed:.2f} 秒")
    print(f"  总生成时间: {total_gen_time:.2f} 秒")
    print(f"  生成token总数: {total_tokens}")
    print(f"  平均tokens/轨迹: {total_tokens/processed:.1f}")
    print(f"  平均生成时间/轨迹: {total_gen_time/processed:.4f} 秒")
    print(f"  原始数据平均env时间: {avg_env_time:.4f} 秒")
    print(f"{'='*70}\n")
    
    return {
        "batch_size": batch_size,
        "total_trajectories": len(trajectories),
        "processed": processed,
        "total_time": total_time,
        "throughput": processed/total_time,
        "avg_time_per_trajectory": total_time/processed,
        "total_tokens": total_tokens,
        "avg_tokens_per_trajectory": total_tokens/processed,
        "total_gen_time": total_gen_time,
        "avg_gen_time_per_trajectory": total_gen_time/processed,
        "avg_env_time_original": avg_env_time
    }


def main():
    print("=" * 70)
    print("批量处理性能测试 - 500条轨迹")
    print("=" * 70)
    
    # 加载配置
    print("\n[1/5] 加载配置文件...")
    config = load_config()
    print(f"✓ 配置加载完成")
    print(f"  System prompt 长度: {len(config['system_prompt'])} 字符")
    
    # 加载轨迹数据
    print("\n[2/5] 加载轨迹数据...")
    num_trajectories = 500
    trajectories = load_trajectories(num_trajectories)
    print(f"✓ 加载了 {len(trajectories)} 条轨迹")
    
    # 初始化模型
    print("\n[3/5] 正在加载模型...")
    start_time = time.time()
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.8,
    )
    sgl.set_default_backend(runtime)
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成！耗时: {load_time:.2f}秒")
    
    # 测试不同的batch size
    print("\n[4/5] 开始性能测试...")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    batch_sizes = [1, 2, 4, 8, 16]
    results = []
    
    for batch_size in batch_sizes:
        result = benchmark_batch(runtime, trajectories, batch_size, config)
        results.append(result)
        
        # 短暂休息
        time.sleep(1)
    
    # 输出汇总结果
    print("\n[5/5] 性能测试汇总")
    print("=" * 70)
    print(f"{'Batch':<8} {'总耗时':<12} {'吞吐量':<14} {'每条耗时':<14} {'生成耗时':<14}")
    print(f"{'Size':<8} {'(秒)':<12} {'(条/秒)':<14} {'(秒)':<14} {'(秒)':<14}")
    print("-" * 70)
    for r in results:
        print(f"{r['batch_size']:<8} "
              f"{r['total_time']:<12.2f} "
              f"{r['throughput']:<14.2f} "
              f"{r['avg_time_per_trajectory']:<14.4f} "
              f"{r['avg_gen_time_per_trajectory']:<14.4f}")
    print("=" * 70)
    
    # 找出最快的配置
    best_result = max(results, key=lambda x: x['throughput'])
    print(f"\n最优配置: Batch Size = {best_result['batch_size']}, "
          f"吞吐量 = {best_result['throughput']:.2f} 条/秒")
    
    # 保存结果
    output_file = "benchmark_results_500.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": MODEL_PATH,
            "num_trajectories": num_trajectories,
            "model_load_time": load_time,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "best_batch_size": best_result['batch_size'],
            "best_throughput": best_result['throughput']
        }, f, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    # 关闭
    print("\n清理资源...")
    runtime.shutdown()
    print("✓ 所有测试完成！")


if __name__ == "__main__":
    main()

