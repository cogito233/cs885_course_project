"""
最终测试版本
- GPU 2（空闲）
- Qwen3-14B-Base
- 50条轨迹 × 50轮对话
- Batch Size 128
- 强制generate每轮assistant
- 解决输入长度限制问题
"""

import sglang as sgl
import os
import time
import json
import yaml


# 使用GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    """准备单条轨迹数据，控制内容长度"""
    initial_context = []
    
    # System prompt - 大幅缩短
    initial_context.append({
        "role": "system",
        "content": system_prompt[:500]  # 只保留前500字符
    })
    
    # 第一条user消息 - 控制长度
    problem_statement = trajectory.get('problem_statement', '')[:300]
    first_user_msg = instance_prompt_template.format(problem_statement=problem_statement)
    initial_context.append({
        "role": "user",
        "content": first_user_msg[:500]
    })
    
    # 处理trajectory steps
    steps = trajectory.get('trajectory_steps', [])[:num_turns]
    turn_data = []
    total_env_time = 0
    
    for step in steps:
        env_time = float(step.get('env_exec_time', 0))
        total_env_time += env_time
        
        # 大幅缩短observation
        observation = step.get('observation', '')[:200]
        next_user_msg = f"{observation}...\n[Env: {env_time:.2f}s]"
        
        turn_data.append({
            "next_user_msg": next_user_msg,
            "env_time": env_time
        })
    
    return initial_context, turn_data, total_env_time


@sgl.function
def multiturn_gen(s, initial_context, turn_data):
    """多轮对话生成"""
    for msg in initial_context:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
    
    for i, turn in enumerate(turn_data):
        # 强制生成
        s += sgl.assistant(sgl.gen(
            f"turn_{i}",
            max_tokens=128,  # 降低到128
            temperature=0.7
        ))
        s += sgl.user(turn["next_user_msg"])


def test_one_turn(runtime, trajectories, batch_size, config, num_turns):
    """测试指定轮数"""
    print(f"测试 {num_turns} 轮...")
    
    system_prompt = config['system_prompt']
    instance_prompt_template = config['instance_prompt']
    
    # 准备数据
    all_data = []
    total_env_time = 0
    
    for traj in trajectories:
        initial_ctx, turn_data, env_time = prepare_trajectory_data(
            traj, system_prompt, instance_prompt_template, num_turns
        )
        all_data.append({
            "initial_context": initial_ctx,
            "turn_data": turn_data
        })
        total_env_time += env_time
    
    # 处理
    start_time = time.time()
    processed = 0
    total_tokens = 0
    
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        
        for data in batch:
            try:
                state = multiturn_gen.run(
                    initial_context=data["initial_context"],
                    turn_data=data["turn_data"]
                )
                processed += 1
                
                # 统计tokens
                for j in range(len(data["turn_data"])):
                    try:
                        resp = state[f"turn_{j}"]
                        if resp:
                            total_tokens += len(resp.split())
                    except:
                        pass
                
                # 每10条输出一次
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {processed}/{len(all_data)} ({elapsed:.1f}秒)")
                        
            except Exception as e:
                if processed == 0:
                    print(f"  错误: {str(e)[:150]}")
                continue
    
    total_time = time.time() - start_time
    avg_env_time = total_env_time / len(all_data)
    
    print(f"  ✓ 完成: {total_time:.2f}秒, {processed/total_time:.1f}条/秒, "
          f"{total_tokens}tokens, {total_tokens/max(processed,1):.0f}tok/轨迹, "
          f"env:{avg_env_time:.1f}秒")
    
    return {
        "num_turns": num_turns,
        "processed": processed,
        "total_time": total_time,
        "throughput": processed/total_time,
        "total_tokens": total_tokens,
        "avg_tokens_per_traj": total_tokens/max(processed,1),
        "avg_env_time": avg_env_time
    }


def main():
    print("=" * 70)
    print("最终测试 - GPU 2 - Qwen3-14B-Base")
    print("=" * 70)
    print(f"时间: {time.strftime('%H:%M:%S')}\n")
    
    # 加载
    print("[1/3] 加载...")
    config = load_config()
    trajectories = load_trajectories(50)
    print(f"✓ 50条轨迹\n")
    
    # 加载模型
    print("[2/3] 加载模型到GPU 2...")
    print("配置: mem_fraction=0.2（适应有限显存）")
    start_time = time.time()
    
    try:
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.2,
            max_total_tokens=4096,  # 限制最大token数
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start_time
        print(f"✓ 完成! {load_time:.1f}秒")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return
    
    # GPU检查
    print("\nGPU 2显存:")
    os.system("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^2,'")
    
    # 测试
    print(f"\n[3/3] 渐进式测试 (Batch Size 128)...")
    print("="* 70)
    
    results = []
    for num_turns in [5, 10, 20, 50]:
        print(f"\n▶ {num_turns}轮对话:")
        try:
            result = test_one_turn(runtime, trajectories, 128, config, num_turns)
            results.append(result)
            time.sleep(1)
        except Exception as e:
            print(f"  ✗ 失败: {str(e)[:100]}")
            break
    
    # 汇总
    print(f"\n\n{'='*70}")
    print("汇总结果")
    print("="*70)
    print(f"{'轮数':<8} {'总耗时':<12} {'吞吐量':<12} {'总Tokens':<12} {'Env时间':<12}")
    print("-"*70)
    for r in results:
        print(f"{r['num_turns']:<8} "
              f"{r['total_time']:<12.2f} "
              f"{r['throughput']:<12.1f} "
              f"{r['total_tokens']:<12} "
              f"{r['avg_env_time']:<12.1f}")
    print("="*70)
    
    # 保存
    with open("final_results_gpu2_qwen3.json", 'w') as f:
        json.dump({
            "model": "Qwen3-14B-Base",
            "gpu": "GPU 2",
            "batch_size": 128,
            "num_trajectories": 50,
            "model_load_time": load_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": results
        }, f, indent=2)
    print("\n结果已保存: final_results_gpu2_qwen3.json")
    
    # 关闭
    runtime.shutdown()
    print("✓ 完成!")


if __name__ == "__main__":
    main()

