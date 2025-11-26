"""
批量处理脚本 - 稳定版本
专注于测量不同batch_size下处理500条轨迹的时间
包含环境执行时间信息
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


@sgl.function
def simple_inference(s, prompt):
    """
    简单的推理函数
    """
    s += sgl.user(prompt)
    s += sgl.assistant(sgl.gen("response", max_tokens=128, temperature=0.0))


def benchmark_batch_simple(runtime, trajectories, batch_size, config):
    """
    测试单个batch_size的性能
    使用简化的推理方式来确保稳定性
    """
    print(f"\n{'='*70}")
    print(f"测试 Batch Size: {batch_size}")
    print(f"{'='*70}")
    
    system_prompt = config['system_prompt']
    instance_prompt_template = config['instance_prompt']
    
    # 统计环境执行时间
    env_times = []
    prompts = []
    
    for traj in trajectories:
        # 计算环境执行时间
        traj_env_time = 0
        steps = traj.get('trajectory_steps', [])
        for step in steps[:5]:  # 只取前5步
            traj_env_time += float(step.get('env_exec_time', 0))
        env_times.append(traj_env_time)
        
        # 构造简化的prompt
        problem_statement = traj.get('problem_statement', '')[:500]
        prompt = f"Problem: {problem_statement}\n\nProvide next action."
        prompts.append(prompt)
    
    print(f"准备了 {len(prompts)} 个prompts")
    print(f"平均环境执行时间: {sum(env_times)/len(env_times):.4f}秒/轨迹")
    print(f"总环境执行时间: {sum(env_times):.2f}秒")
    
    # 开始批量处理
    start_time = time.time()
    processed = 0
    total_tokens = 0
    failed = 0
    
    # 分批处理
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_start = time.time()
        
        # 处理当前批次
        for prompt in batch_prompts:
            try:
                state = simple_inference.run(prompt=prompt)
                processed += 1
                if 'response' in state:
                    total_tokens += len(state['response'].split())
            except Exception as e:
                failed += 1
                if failed == 1:  # 只打印第一个错误
                    print(f"  处理出错: {str(e)[:100]}")
                continue
        
        batch_time = time.time() - batch_start
        
        # 定期输出进度
        batch_num = i // batch_size + 1
        if batch_num % 50 == 0 or i + batch_size >= len(prompts):
            print(f"  进度: {processed}/{len(prompts)} "
                  f"({100*processed/len(prompts):.1f}%), "
                  f"速度: {processed/(time.time()-start_time):.2f} 条/秒")
    
    total_time = time.time() - start_time
    avg_env_time = sum(env_times) / len(env_times)
    total_env_time = sum(env_times)
    
    print(f"\n{'='*70}")
    print(f"Batch Size {batch_size} 测试结果:")
    print(f"  总轨迹数: {len(trajectories)}")
    print(f"  成功处理: {processed}")
    print(f"  失败: {failed}")
    print(f"  推理总耗时: {total_time:.2f} 秒")
    print(f"  平均速度: {processed/total_time:.2f} 条/秒")
    print(f"  每条轨迹平均推理耗时: {total_time/processed:.4f} 秒")
    print(f"  生成token总数: {total_tokens}")
    print(f"  平均tokens/轨迹: {total_tokens/processed:.1f}")
    print(f"  ")
    print(f"  环境执行时间统计（来自原始数据）:")
    print(f"    平均env执行时间: {avg_env_time:.4f} 秒/轨迹")
    print(f"    总env执行时间: {total_env_time:.2f} 秒")
    print(f"  ")
    print(f"  完成{processed}条轨迹所需总时间: {total_time:.2f} 秒")
    print(f"{'='*70}\n")
    
    return {
        "batch_size": batch_size,
        "total_trajectories": len(trajectories),
        "processed": processed,
        "failed": failed,
        "inference_total_time": total_time,
        "throughput": processed/total_time,
        "avg_inference_time_per_trajectory": total_time/processed,
        "total_tokens": total_tokens,
        "avg_tokens_per_trajectory": total_tokens/processed if processed > 0 else 0,
        "avg_env_time_per_trajectory": avg_env_time,
        "total_env_time": total_env_time,
        "total_time_for_trajectories": total_time
    }


def main():
    print("=" * 70)
    print("批量处理性能测试 - 500条轨迹 (稳定版)")
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
            mem_fraction_static=0.75,  # 降低内存使用
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start_time
        print(f"✓ 模型加载完成！耗时: {load_time:.2f}秒")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 测试不同的batch size
    print("\n[4/5] 开始性能测试...")
    
    batch_sizes = [1, 2, 4, 8, 16]
    results = []
    
    for batch_size in batch_sizes:
        try:
            result = benchmark_batch_simple(runtime, trajectories, batch_size, config)
            results.append(result)
            
            # 短暂休息，确保显存释放
            time.sleep(2)
        except Exception as e:
            print(f"✗ Batch size {batch_size} 测试失败: {e}")
            continue
    
    if not results:
        print("\n✗ 所有测试都失败了")
        runtime.shutdown()
        return
    
    # 输出汇总结果
    print("\n[5/5] 性能测试汇总")
    print("=" * 70)
    print(f"{'Batch':<8} {'推理总时':<12} {'吞吐量':<14} {'每条推理':<14} {'平均Env':<14}")
    print(f"{'Size':<8} {'(秒)':<12} {'(条/秒)':<14} {'(秒)':<14} {'时间(秒)':<14}")
    print("-" * 70)
    for r in results:
        print(f"{r['batch_size']:<8} "
              f"{r['inference_total_time']:<12.2f} "
              f"{r['throughput']:<14.2f} "
              f"{r['avg_inference_time_per_trajectory']:<14.4f} "
              f"{r['avg_env_time_per_trajectory']:<14.4f}")
    print("=" * 70)
    
    # 找出最快的配置
    best_result = max(results, key=lambda x: x['throughput'])
    print(f"\n最优配置:")
    print(f"  Batch Size = {best_result['batch_size']}")
    print(f"  吞吐量 = {best_result['throughput']:.2f} 条/秒")
    print(f"  完成500条轨迹需要: {best_result['total_time_for_trajectories']:.2f} 秒")
    
    # 输出不同batch_size完成500条轨迹的时间对比
    print(f"\n完成500条轨迹所需时间对比:")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x['total_time_for_trajectories']):
        print(f"  Batch Size {r['batch_size']:>2}: {r['total_time_for_trajectories']:>8.2f} 秒")
    print("-" * 70)
    
    # 保存结果
    output_file = "benchmark_results_500_stable.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model_path": MODEL_PATH,
            "num_trajectories": num_trajectories,
            "model_load_time": load_time,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "best_batch_size": best_result['batch_size'],
            "best_throughput": best_result['throughput'],
            "best_total_time": best_result['total_time_for_trajectories']
        }, f, indent=2)
    print(f"\n结果已保存到: {output_file}")
    
    # 关闭
    print("\n清理资源...")
    runtime.shutdown()
    print("✓ 所有测试完成！")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

