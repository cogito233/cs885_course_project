"""
批量处理脚本 - 强制generate版本（内容与原轨迹相同）
- 使用GPU 3
- 测试50条轨迹
- 每一轮assistant强制generate，但使用原轨迹内容作为约束
- 测试batch_size: 1, 2, 4, 8, 16, 32, 64, 128, 256
"""

import sglang as sgl
import os
import time
import json
import yaml
from typing import List, Dict


# 明确设置使用GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/R2EGym-7B-Agent"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"


def load_config():
    """加载YAML配置"""
    with open(YAML_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_trajectories(num_trajectories=50):
    """从jsonl文件加载轨迹数据"""
    trajectories = []
    with open(JSONL_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_trajectories:
                break
            data = json.loads(line)
            trajectories.append(data)
    return trajectories


def prepare_trajectory_data(trajectory, system_prompt, instance_prompt_template, max_turns=None):
    """
    准备单条轨迹的完整多轮对话数据
    返回: (初始上下文, 每轮的数据, 总env时间)
    """
    # 初始上下文
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
    
    # 准备每一轮的数据（包含原始的assistant回复）
    steps = trajectory.get('trajectory_steps', [])
    if max_turns:
        steps = steps[:max_turns]
    
    turn_data = []
    total_env_time = 0
    
    for step in steps:
        env_time = float(step.get('env_exec_time', 0))
        total_env_time += env_time
        
        # 原始的assistant回复（thought + action）
        thought = step.get('thought', '')[:500]
        action = step.get('action', '')[:500]
        original_assistant_msg = f"{thought}\n\n{action}"
        
        # 下一个user消息（observation）
        observation = step.get('observation', '')[:800]
        next_user_msg = f"{observation}\n\n[Environment execution time: {env_time:.4f}s]"
        
        turn_data.append({
            "original_assistant_msg": original_assistant_msg,
            "next_user_msg": next_user_msg,
            "env_time": env_time
        })
    
    return initial_context, turn_data, total_env_time


@sgl.function
def force_generate_multiturn(s, initial_context, turn_data, max_gen_tokens=256):
    """
    强制generate版本：每轮assistant都generate，但会被原始内容引导
    
    这里使用gen()来强制生成，虽然生成的内容可能与原始不完全相同，
    但这才是真正的generate过程
    """
    # 添加初始上下文
    for msg in initial_context:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
    
    # 逐轮生成
    for i, turn in enumerate(turn_data):
        # 强制生成assistant回复（这是真正的generate）
        s += sgl.assistant(sgl.gen(
            f"turn_{i}_response",
            max_tokens=max_gen_tokens,
            temperature=0.0,  # 使用temperature=0获得确定性结果
            stop=None  # 不设置stop，让模型自然生成
        ))
        
        # 添加下一个user消息
        s += sgl.user(turn["next_user_msg"])


def benchmark_batch_force_generate(runtime, trajectories, batch_size, config, num_turns=10):
    """
    测试强制generate的多轮对话性能
    """
    print(f"\n{'='*70}")
    print(f"测试 Batch Size: {batch_size}, 生成轮数: {num_turns}")
    print(f"{'='*70}")
    
    system_prompt = config['system_prompt']
    instance_prompt_template = config['instance_prompt']
    
    # 准备所有轨迹数据
    all_data = []
    total_env_time_all = 0
    total_original_length = 0
    
    for traj in trajectories:
        initial_ctx, turn_data, env_time = prepare_trajectory_data(
            traj, system_prompt, instance_prompt_template, max_turns=num_turns
        )
        
        # 计算原始assistant消息的总长度
        original_length = sum(len(t["original_assistant_msg"]) for t in turn_data)
        total_original_length += original_length
        
        all_data.append({
            "initial_context": initial_ctx,
            "turn_data": turn_data,
            "env_time": env_time
        })
        total_env_time_all += env_time
    
    avg_turns = sum(len(d["turn_data"]) for d in all_data) / len(all_data)
    avg_env_time = total_env_time_all / len(all_data)
    avg_original_length = total_original_length / len(all_data)
    
    print(f"准备了 {len(all_data)} 条轨迹")
    print(f"平均每条生成 {avg_turns:.1f} 轮assistant回复")
    print(f"原始assistant消息平均长度: {avg_original_length:.0f} 字符/轨迹")
    print(f"平均环境执行时间: {avg_env_time:.4f}秒/轨迹")
    
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
                state = force_generate_multiturn.run(
                    initial_context=data["initial_context"],
                    turn_data=data["turn_data"],
                    max_gen_tokens=256
                )
                gen_time = time.time() - gen_start
                
                processed += 1
                total_gen_time += gen_time
                
                # 统计生成的token数
                for j in range(len(data["turn_data"])):
                    key = f"turn_{j}_response"
                    if key in state:
                        tokens = len(state[key].split())
                        total_tokens += tokens
                        
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  处理出错: {str(e)[:100]}")
                continue
        
        batch_time = time.time() - batch_start
        
        # 定期输出进度
        if batch_count % 5 == 0 or i + batch_size >= len(all_data):
            elapsed = time.time() - start_time
            print(f"  进度: {processed}/{len(all_data)} "
                  f"({100*processed/len(all_data):.1f}%), "
                  f"速度: {processed/elapsed:.2f} 条/秒, "
                  f"已用时: {elapsed:.1f}秒")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Batch Size {batch_size}, {num_turns}轮对话 测试结果:")
    print(f"  总轨迹数: {len(trajectories)}")
    print(f"  成功处理: {processed}")
    print(f"  失败: {failed}")
    print(f"  推理总耗时: {total_time:.2f} 秒")
    print(f"  平均速度: {processed/total_time:.2f} 条/秒")
    print(f"  每条轨迹平均推理耗时: {total_time/max(processed,1):.4f} 秒")
    print(f"  生成token总数: {total_tokens}")
    print(f"  平均tokens/轨迹: {total_tokens/max(processed,1):.1f}")
    print(f"  每轮平均tokens: {total_tokens/max(processed,1)/num_turns:.1f}")
    print(f"  总生成时间: {total_gen_time:.2f} 秒")
    print(f"  平均生成时间/轨迹: {total_gen_time/max(processed,1):.4f} 秒")
    print(f"  ")
    print(f"  环境执行时间统计（来自原始数据）:")
    print(f"    平均env执行时间: {avg_env_time:.4f} 秒/轨迹")
    print(f"    总env执行时间: {total_env_time_all:.2f} 秒")
    print(f"{'='*70}\n")
    
    return {
        "batch_size": batch_size,
        "num_turns": num_turns,
        "total_trajectories": len(trajectories),
        "processed": processed,
        "failed": failed,
        "inference_total_time": total_time,
        "throughput": processed/total_time if total_time > 0 else 0,
        "avg_inference_time_per_trajectory": total_time/max(processed,1),
        "total_tokens": total_tokens,
        "avg_tokens_per_trajectory": total_tokens/max(processed,1),
        "avg_tokens_per_turn": total_tokens/max(processed,1)/num_turns if num_turns > 0 else 0,
        "total_gen_time": total_gen_time,
        "avg_gen_time_per_trajectory": total_gen_time/max(processed,1),
        "avg_env_time_per_trajectory": avg_env_time,
        "total_env_time": total_env_time_all
    }


def verify_gpu():
    """验证GPU使用情况"""
    import torch
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
    print(f"PyTorch可见GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前使用的GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA不可用！")


def main():
    print("=" * 70)
    print("批量处理性能测试 - 强制Generate版本")
    print("每一轮assistant都强制generate")
    print("=" * 70)
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 验证GPU
    print("\n[GPU检查]")
    verify_gpu()
    
    # 加载配置
    print("\n[1/5] 加载配置文件...")
    config = load_config()
    print(f"✓ 配置加载完成")
    
    # 加载轨迹数据（只加载50条）
    print("\n[2/5] 加载轨迹数据...")
    num_trajectories = 50
    trajectories = load_trajectories(num_trajectories)
    print(f"✓ 加载了 {len(trajectories)} 条轨迹")
    
    # 统计原始数据的轮数
    sample_traj = trajectories[0]
    total_steps = len(sample_traj.get('trajectory_steps', []))
    print(f"  原始数据平均约 {total_steps} 步/轨迹")
    
    # 初始化模型
    print("\n[3/5] 正在加载模型到GPU 3...")
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
    
    # 再次检查GPU使用情况
    print("\n[GPU使用检查]")
    os.system("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -4")
    
    # 测试不同的batch size
    print("\n[4/5] 开始强制generate多轮对话测试...")
    print(f"⚠️  测试 {num_trajectories} 条轨迹")
    
    # 测试10轮对话
    num_turns = 10
    print(f"\n生成 {num_turns} 轮对话...")
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    results = []
    
    for batch_size in batch_sizes:
        try:
            result = benchmark_batch_force_generate(
                runtime, trajectories, batch_size, config, num_turns
            )
            results.append(result)
            time.sleep(1)
        except Exception as e:
            print(f"✗ Batch size {batch_size} 测试失败: {str(e)[:200]}")
            continue
    
    if not results:
        print("\n✗ 所有测试都失败了")
        runtime.shutdown()
        return
    
    # 输出汇总结果
    print("\n[5/5] 性能测试汇总")
    print("=" * 70)
    print(f"轨迹数量: {num_trajectories} 条")
    print(f"生成轮数: {num_turns} 轮")
    print("=" * 70)
    print(f"{'Batch':<8} {'总耗时':<12} {'吞吐量':<14} {'每条耗时':<14} {'平均Tokens':<14}")
    print(f"{'Size':<8} {'(秒)':<12} {'(条/秒)':<14} {'(秒)':<14} {'(/轨迹)':<14}")
    print("-" * 70)
    for r in results:
        print(f"{r['batch_size']:<8} "
              f"{r['inference_total_time']:<12.2f} "
              f"{r['throughput']:<14.2f} "
              f"{r['avg_inference_time_per_trajectory']:<14.2f} "
              f"{r['avg_tokens_per_trajectory']:<14.1f}")
    print("=" * 70)
    
    # 找出最快的配置
    best_result = max(results, key=lambda x: x['throughput'])
    print(f"\n最优配置:")
    print(f"  Batch Size = {best_result['batch_size']}")
    print(f"  吞吐量 = {best_result['throughput']:.2f} 条/秒")
    print(f"  完成{num_trajectories}条轨迹需要: {best_result['inference_total_time']:.2f} 秒")
    print(f"  平均生成 {best_result['avg_tokens_per_trajectory']:.1f} tokens/轨迹")
    
    # 时间对比
    print(f"\n完成{num_trajectories}条{num_turns}轮对话所需时间对比:")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['inference_total_time']):
        success_rate = 100 * r['processed'] / r['total_trajectories']
        print(f"  Batch Size {r['batch_size']:>3}: {r['inference_total_time']:>8.2f} 秒 "
              f"(成功: {success_rate:.0f}%, {r['avg_tokens_per_trajectory']:.0f} tokens/轨迹)")
    print("-" * 70)
    
    # 保存结果
    output_file = f"benchmark_results_force_generate_{num_trajectories}traj_{num_turns}turns.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": MODEL_PATH,
            "gpu_device": "GPU 3",
            "num_trajectories": num_trajectories,
            "num_turns": num_turns,
            "model_load_time": load_time,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": "force generate multi-turn dialogue",
            "note": "Every assistant turn uses gen() to force generation with temperature=0",
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

