"""
最终可用版本 - 后台运行
GPU 2, Qwen3-14B-Base
50条轨迹 × 渐进测试(5/10/20/50轮) × BS128
每轮强制generate assistant回复
"""

import sglang as sgl
import os, time, json, yaml, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen3-14B-Base"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

@sgl.function
def multiturn_generate(s, user_msgs, num_turns):
    """多轮生成 - 添加stop tokens防止无限生成"""
    for i in range(num_turns):
        if i < len(user_msgs):
            s += sgl.user(user_msgs[i])
        # 关键：添加stop tokens
        s += sgl.assistant(sgl.gen(
            f"t{i}",
            max_tokens=200,
            temperature=0.7,
            stop=["<|im_end|>", "<|endoftext|>", "\n\nUSER:", "\n\nASSISTANT:"]
        ))

def benchmark(runtime, trajs, batch_size, config, num_turns):
    log(f"开始测试 {num_turns} 轮...")
    
    # 准备数据
    all_data = []
    total_env_time = 0
    
    for traj in trajs:
        problem = traj.get('problem_statement', '')[:400]
        user_msgs = [f"Problem: {problem}"]
        traj_env_time = 0
        
        steps = traj.get('trajectory_steps', [])[:num_turns]
        for s in steps:
            env_time = float(s.get('env_exec_time', 0))
            traj_env_time += env_time
            obs = s.get('observation', '')[:150]
            user_msgs.append(f"{obs}...\n[Env: {env_time:.2f}s]")
        
        all_data.append({"user_msgs": user_msgs, "env_time": traj_env_time})
        total_env_time += traj_env_time
    
    # 处理
    start_time = time.time()
    processed = 0
    total_tokens = 0
    
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        
        for data in batch:
            try:
                state = multiturn_generate.run(
                    user_msgs=data["user_msgs"],
                    num_turns=num_turns
                )
                processed += 1
                
                # 统计tokens
                for j in range(num_turns):
                    try:
                        resp = state[f"t{j}"]
                        if resp:
                            total_tokens += len(resp.split())
                    except:
                        pass
                
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    log(f"  {processed}/{len(all_data)} ({elapsed:.1f}秒, {processed/elapsed:.1f}条/秒)")
                        
            except Exception as e:
                if processed == 0:
                    log(f"  错误: {str(e)[:100]}")
                continue
    
    total_time = time.time() - start_time
    avg_env = total_env_time / len(all_data)
    
    result = {
        "num_turns": num_turns,
        "batch_size": batch_size,
        "processed": processed,
        "total_time": total_time,
        "throughput": processed/total_time if total_time > 0 else 0,
        "total_tokens": total_tokens,
        "avg_tokens_per_traj": total_tokens/max(processed,1),
        "avg_tokens_per_turn": total_tokens/max(processed,1)/num_turns,
        "avg_env_time": avg_env,
        "total_env_time": total_env_time
    }
    
    log(f"✓ 完成: {total_time:.2f}秒, {result['throughput']:.2f}条/秒, {total_tokens}tokens")
    return result


def main():
    if __name__ != '__main__':
        return
    
    log("="*70)
    log("最终基准测试")
    log("="*70)
    
    # 加载
    log("[1/3] 加载数据...")
    config = load_config()
    trajs = load_trajectories(50)
    log(f"✓ {len(trajs)}条")
    
    # 加载模型
    log("\n[2/3] 加载模型到GPU 2...")
    start = time.time()
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.8,
        max_total_tokens=8192
    )
    sgl.set_default_backend(runtime)
    load_time = time.time() - start
    log(f"✓ {load_time:.1f}秒")
    
    # 测试
    log("\n[3/3] 渐进测试: 5->10->20->50轮")
    all_results = []
    
    for num_turns in [5, 10, 20, 50]:
        try:
            result = benchmark(runtime, trajs, 128, config, num_turns)
            all_results.append(result)
            time.sleep(1)
        except Exception as e:
            log(f"✗ {num_turns}轮失败: {e}")
            break
    
    # 汇总
    log("\n" + "="*70)
    log("汇总结果")
    log("="*70)
    print(f"{'轮数':<8} {'耗时':<10} {'吞吐量':<10} {'总Tokens':<12} {'Tok/轮':<10} {'Env时间':<10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['num_turns']:<8} {r['total_time']:<10.2f} {r['throughput']:<10.2f} {r['total_tokens']:<12} {r['avg_tokens_per_turn']:<10.1f} {r['avg_env_time']:<10.2f}")
    log("="*70)
    
    # 保存
    with open("FINAL_RESULTS.json", 'w') as f:
        json.dump({
            "model": "Qwen3-14B-Base",
            "gpu": "GPU 2",
            "batch_size": 128,
            "num_trajectories": 50,
            "model_load_time": load_time,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": all_results
        }, f, indent=2)
    log("\n✓ 结果已保存: FINAL_RESULTS.json")
    
    runtime.shutdown()
    log("✓ 完成!")


if __name__ == '__main__':
    main()

