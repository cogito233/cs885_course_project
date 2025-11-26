"""
批量处理脚本 - 真正的多轮对话版本
测试不同batch_size下处理500条轨迹的多轮对话推理性能
支持更大的batch size: 1, 2, 4, 8, 16, 32, 64, 128, 256
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


def construct_multiturn_context(trajectory, system_prompt, instance_prompt_template, num_turns=3):
    """
    构造多轮对话上下文（前N-1轮），用于生成第N轮
    返回：历史消息列表和环境执行时间
    """
    messages = []
    total_env_time = 0
    
    # System prompt
    messages.append({
        "role": "system",
        "content": system_prompt[:2000]  # 限制长度
    })
    
    # 第一条user prompt（包含问题描述）
    problem_statement = trajectory.get('problem_statement', '')[:800]
    first_user_msg = instance_prompt_template.format(problem_statement=problem_statement)
    messages.append({
        "role": "user",
        "content": first_user_msg[:1500]
    })
    
    # 处理trajectory steps（取前num_turns-1轮作为历史）
    steps = trajectory.get('trajectory_steps', [])[:num_turns-1]
    for step in steps:
        # 累计环境执行时间
        env_time = float(step.get('env_exec_time', 0))
        total_env_time += env_time
        
        # Assistant: thought + action
        thought = step.get('thought', '')[:500]
        action = step.get('action', '')[:500]
        assistant_msg = f"{thought}\n\n{action}"
        messages.append({
            "role": "assistant",
            "content": assistant_msg
        })
        
        # User: observation + env_exec_time
        observation = step.get('observation', '')[:800]
        user_msg = f"{observation}\n\n[Environment execution time: {env_time:.4f}s]"
        messages.append({
            "role": "user",
            "content": user_msg
        })
    
    return messages, total_env_time


@sgl.function
def generate_multiturn_response(s, messages):
    """
    基于多轮对话历史生成下一轮回复
    """
    # 添加所有历史消息
    for msg in messages:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
        elif msg["role"] == "assistant":
            s += sgl.assistant(msg["content"])
    
    # 生成下一轮assistant回复
    s += sgl.assistant(sgl.gen(
        "next_response",
        max_tokens=256,
        temperature=0.0
    ))


def benchmark_batch_multiturn(runtime, trajectories, batch_size, config, num_turns=3):
    """测试单个batch_size的多轮对话性能"""
    print(f"\n{'='*70}")
    print(f"测试 Batch Size: {batch_size}")
    print(f"{'='*70}")
    
    system_prompt = config['system_prompt']
    instance_prompt_template = config['instance_prompt']
    
    # 准备所有多轮对话数据
    all_data = []
    total_env_time_all = 0
    
    for traj in trajectories:
        messages, env_time = construct_multiturn_context(
            traj, system_prompt, instance_prompt_template, num_turns
        )
        all_data.append({
            "messages": messages,
            "env_time": env_time
        })
        total_env_time_all += env_time
    
    avg_msg_count = sum(len(d["messages"]) for d in all_data) / len(all_data)
    avg_env_time = total_env_time_all / len(all_data)
    
    print(f"准备了 {len(all_data)} 条多轮对话")
    print(f"平均每条有 {avg_msg_count:.1f} 条消息")
    print(f"平均环境执行时间: {avg_env_time:.4f}秒/轨迹 (前{num_turns-1}轮)")
    
    # 开始批量处理
    start_time = time.time()
    processed = 0
    failed = 0
    total_tokens = 0
    total_gen_time = 0
    
    # 分批处理
    batch_count = 0
    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i:i+batch_size]
        batch_start = time.time()
        batch_count += 1
        
        # 处理当前批次
        for data in batch_data:
            try:
                gen_start = time.time()
                state = generate_multiturn_response.run(messages=data["messages"])
                gen_time = time.time() - gen_start
                
                processed += 1
                total_gen_time += gen_time
                
                # 统计生成的token数
                if 'next_response' in state:
                    response = state['next_response']
                    tokens = len(response.split())
                    total_tokens += tokens
            except Exception as e:
                failed += 1
                if failed <= 3:  # 只打印前3个错误
                    print(f"  处理出错: {str(e)[:100]}")
                continue
        
        batch_time = time.time() - batch_start
        
        # 定期输出进度
        if batch_count % 20 == 0 or i + batch_size >= len(all_data):
            print(f"  进度: {processed}/{len(all_data)} "
                  f"({100*processed/len(all_data):.1f}%), "
                  f"速度: {processed/(time.time()-start_time):.2f} 条/秒")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Batch Size {batch_size} 测试结果:")
    print(f"  总轨迹数: {len(trajectories)}")
    print(f"  成功处理: {processed}")
    print(f"  失败: {failed}")
    print(f"  推理总耗时: {total_time:.2f} 秒")
    print(f"  平均速度: {processed/total_time:.2f} 条/秒")
    print(f"  每条轨迹平均推理耗时: {total_time/max(processed,1):.4f} 秒")
    print(f"  生成token总数: {total_tokens}")
    print(f"  平均tokens/轨迹: {total_tokens/max(processed,1):.1f}")
    print(f"  总生成时间: {total_gen_time:.2f} 秒")
    print(f"  平均生成时间/轨迹: {total_gen_time/max(processed,1):.4f} 秒")
    print(f"  ")
    print(f"  环境执行时间统计（来自原始数据，前{num_turns-1}轮）:")
    print(f"    平均env执行时间: {avg_env_time:.4f} 秒/轨迹")
    print(f"    总env执行时间: {total_env_time_all:.2f} 秒")
    print(f"  ")
    print(f"  完成{processed}条轨迹所需总时间: {total_time:.2f} 秒")
    print(f"{'='*70}\n")
    
    return {
        "batch_size": batch_size,
        "total_trajectories": len(trajectories),
        "processed": processed,
        "failed": failed,
        "inference_total_time": total_time,
        "throughput": processed/total_time if total_time > 0 else 0,
        "avg_inference_time_per_trajectory": total_time/max(processed,1),
        "total_tokens": total_tokens,
        "avg_tokens_per_trajectory": total_tokens/max(processed,1),
        "total_gen_time": total_gen_time,
        "avg_gen_time_per_trajectory": total_gen_time/max(processed,1),
        "avg_env_time_per_trajectory": avg_env_time,
        "total_env_time": total_env_time_all,
        "num_turns": num_turns,
        "avg_messages_per_trajectory": avg_msg_count
    }


def main():
    print("=" * 70)
    print("批量处理性能测试 - 真正的多轮对话版本")
    print("=" * 70)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
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
    
    try:
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.75,
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start_time
        print(f"✓ 模型加载完成！耗时: {load_time:.2f}秒")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 测试不同的batch size
    print("\n[4/5] 开始多轮对话性能测试...")
    print("注意: 这是真正的多轮对话生成，每条轨迹会生成新的回复")
    
    # 扩展到更大的batch size
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    num_turns = 3  # 使用前2轮历史，生成第3轮
    results = []
    
    for batch_size in batch_sizes:
        try:
            result = benchmark_batch_multiturn(
                runtime, trajectories, batch_size, config, num_turns
            )
            results.append(result)
            
            # 短暂休息，确保显存释放
            time.sleep(2)
        except Exception as e:
            print(f"✗ Batch size {batch_size} 测试失败: {str(e)[:200]}")
            # 继续测试下一个batch size
            continue
    
    if not results:
        print("\n✗ 所有测试都失败了")
        runtime.shutdown()
        return
    
    # 输出汇总结果
    print("\n[5/5] 性能测试汇总")
    print("=" * 70)
    print(f"{'Batch':<8} {'推理总时':<12} {'吞吐量':<14} {'每条推理':<14} {'平均生成':<14}")
    print(f"{'Size':<8} {'(秒)':<12} {'(条/秒)':<14} {'(秒)':<14} {'Tokens':<14}")
    print("-" * 70)
    for r in results:
        print(f"{r['batch_size']:<8} "
              f"{r['inference_total_time']:<12.2f} "
              f"{r['throughput']:<14.2f} "
              f"{r['avg_inference_time_per_trajectory']:<14.4f} "
              f"{r['avg_tokens_per_trajectory']:<14.1f}")
    print("=" * 70)
    
    # 找出最快的配置
    best_result = max(results, key=lambda x: x['throughput'])
    print(f"\n最优配置:")
    print(f"  Batch Size = {best_result['batch_size']}")
    print(f"  吞吐量 = {best_result['throughput']:.2f} 条/秒")
    print(f"  完成500条轨迹需要: {best_result['inference_total_time']:.2f} 秒")
    print(f"  平均生成 {best_result['avg_tokens_per_trajectory']:.1f} tokens/轨迹")
    
    # 输出不同batch_size完成500条轨迹的时间对比
    print(f"\n完成500条轨迹所需时间对比:")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['inference_total_time']):
        success_rate = 100 * r['processed'] / r['total_trajectories']
        print(f"  Batch Size {r['batch_size']:>3}: {r['inference_total_time']:>8.2f} 秒 "
              f"(成功率: {success_rate:.1f}%, 平均{r['avg_tokens_per_trajectory']:.0f} tokens)")
    print("-" * 70)
    
    # 保存结果
    output_file = "benchmark_results_multiturn.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": MODEL_PATH,
            "num_trajectories": num_trajectories,
            "num_turns": num_turns,
            "model_load_time": load_time,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "multi-turn dialogue generation",
            "results": results,
            "best_batch_size": best_result['batch_size'],
            "best_throughput": best_result['throughput'],
            "best_total_time": best_result['inference_total_time']
        }, f, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    # 关闭
    print("\n清理资源...")
    runtime.shutdown()
    print("✓ 所有测试完成！")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

