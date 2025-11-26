"""
批量处理脚本 - 测试不同batch_size下的性能
从jsonl文件构造多轮对话，插入环境用时，测试推理速度
"""

import sglang as sgl
import os
import time
import json
import yaml
from typing import List, Dict
from pathlib import Path


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


def construct_messages(trajectory, system_prompt, instance_prompt_template):
    """
    构造多轮对话消息
    格式：system -> user (initial) -> assistant (thought+action) -> user (observation) -> ...
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
    
    # 处理trajectory steps
    steps = trajectory.get('trajectory_steps', [])
    for step in steps:
        # Assistant: thought + action
        thought = step.get('thought', '')
        action = step.get('action', '')
        assistant_msg = f"{thought}\n\n{action}"
        messages.append({
            "role": "assistant",
            "content": assistant_msg
        })
        
        # User: observation + env_exec_time
        observation = step.get('observation', '')
        env_exec_time = step.get('env_exec_time', 0)
        user_msg = f"{observation}\n\n[Environment execution time: {env_exec_time}s]"
        messages.append({
            "role": "user",
            "content": user_msg
        })
    
    return messages


@sgl.function
def process_single_trajectory(s, messages):
    """处理单条轨迹"""
    for msg in messages:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
        elif msg["role"] == "assistant":
            s += sgl.assistant(msg["content"])


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
        messages = construct_messages(traj, system_prompt, instance_prompt_template)
        all_messages.append(messages)
    
    print(f"准备了 {len(all_messages)} 条轨迹")
    print(f"平均每条轨迹有 {sum(len(m) for m in all_messages) / len(all_messages):.1f} 条消息")
    
    # 开始批量处理
    start_time = time.time()
    processed = 0
    
    # 分批处理
    for i in range(0, len(all_messages), batch_size):
        batch_messages = all_messages[i:i+batch_size]
        batch_start = time.time()
        
        # 处理当前批次（注意：这里简化处理，实际可能需要并行）
        for messages in batch_messages:
            try:
                # 只处理前几轮对话以加速测试
                truncated_messages = messages[:min(10, len(messages))]
                state = process_single_trajectory.run(messages=truncated_messages)
                processed += 1
            except Exception as e:
                print(f"处理出错: {e}")
                continue
        
        batch_time = time.time() - batch_start
        if (i // batch_size + 1) % 10 == 0:
            print(f"  已处理: {processed}/{len(all_messages)} 条, "
                  f"当前批次耗时: {batch_time:.2f}秒, "
                  f"平均速度: {len(batch_messages)/batch_time:.2f} 条/秒")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Batch Size {batch_size} 测试结果:")
    print(f"  总轨迹数: {len(trajectories)}")
    print(f"  成功处理: {processed}")
    print(f"  总耗时: {total_time:.2f} 秒")
    print(f"  平均速度: {processed/total_time:.2f} 条/秒")
    print(f"  每条轨迹平均耗时: {total_time/processed:.2f} 秒")
    print(f"{'='*70}\n")
    
    return {
        "batch_size": batch_size,
        "total_trajectories": len(trajectories),
        "processed": processed,
        "total_time": total_time,
        "throughput": processed/total_time,
        "avg_time_per_trajectory": total_time/processed
    }


def main():
    print("=" * 70)
    print("批量处理性能测试")
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
    batch_sizes = [1, 2, 4, 8, 16]
    results = []
    
    for batch_size in batch_sizes:
        result = benchmark_batch(runtime, trajectories, batch_size, config)
        results.append(result)
        
        # 短暂休息，避免显存溢出
        time.sleep(2)
    
    # 输出汇总结果
    print("\n[5/5] 性能测试汇总")
    print("=" * 70)
    print(f"{'Batch Size':<12} {'总耗时(秒)':<15} {'吞吐量(条/秒)':<18} {'每条耗时(秒)':<15}")
    print("-" * 70)
    for r in results:
        print(f"{r['batch_size']:<12} {r['total_time']:<15.2f} {r['throughput']:<18.2f} {r['avg_time_per_trajectory']:<15.2f}")
    print("=" * 70)
    
    # 保存结果
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": MODEL_PATH,
            "num_trajectories": num_trajectories,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    # 关闭
    print("\n清理资源...")
    runtime.shutdown()
    print("✓ 完成！")


if __name__ == "__main__":
    main()

