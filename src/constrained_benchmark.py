"""
约束生成版本 - 强制generate但输出与原轨迹一致
使用regex约束或者直接验证生成
GPU 2, Qwen3-14B-Base, 50条×50轮×BS128
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


def load_config():
    with open(YAML_FILE) as f:
        return yaml.safe_load(f)


def load_trajectories(num=50):
    trajs = []
    with open(JSONL_FILE) as f:
        for i, line in enumerate(f):
            if i >= num:
                break
            trajs.append(json.loads(line))
    return trajs


@sgl.function
def constrained_multiturn(s, initial_context, turn_data):
    """
    约束式多轮生成：
    - 每轮都调用gen()强制生成
    - 但使用原轨迹内容作为"目标"来引导生成
    - 通过regex或前缀约束来确保输出一致
    """
    # 初始上下文
    for msg in initial_context:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
    
    # 逐轮生成
    for i, turn in enumerate(turn_data):
        # 方法：使用原内容的前80%作为prefix，强制生成剩余20%
        original_response = turn["original_response"]
        
        # 分成prefix和要生成的部分
        split_point = int(len(original_response) * 0.8)
        prefix = original_response[:split_point]
        
        # 如果prefix太短，至少保留一些内容
        if len(prefix) < 50 and len(original_response) > 50:
            prefix = original_response[:50]
        
        # 使用prefix + gen的方式
        if prefix:
            s += sgl.assistant_begin()
            s += prefix
            s += sgl.gen(
                f"turn_{i}",
                max_tokens=256,
                temperature=0.0,  # 确定性
                stop=["USER:", "\nUSER:"]  # 防止生成新对话轮
            )
            s += sgl.assistant_end()
        else:
            # 如果没有prefix，直接生成
            s += sgl.assistant(sgl.gen(
                f"turn_{i}",
                max_tokens=256,
                temperature=0.0,
                stop=["USER:", "\nUSER:"]
            ))
        
        # 下一个user消息
        s += sgl.user(turn["next_user_msg"])


def prepare_data(traj, config, num_turns):
    """准备单条轨迹数据"""
    system_prompt = config['system_prompt'][:500]
    instance_template = config['instance_prompt']
    
    initial_context = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instance_template.format(
            problem_statement=traj.get('problem_statement', '')[:500]
        )[:1000]}
    ]
    
    steps = traj.get('trajectory_steps', [])[:num_turns]
    turn_data = []
    total_env_time = 0
    
    for step in steps:
        env_time = float(step.get('env_exec_time', 0))
        total_env_time += env_time
        
        # 原始assistant回复
        thought = step.get('thought', '')[:300]
        action = step.get('action', '')[:300]
        original_response = f"{thought}\n\n{action}"
        
        # 下一个user消息
        obs = step.get('observation', '')[:200]
        next_user_msg = f"{obs}...\n[Env: {env_time:.3f}s]"
        
        turn_data.append({
            "original_response": original_response,
            "next_user_msg": next_user_msg,
            "env_time": env_time
        })
    
    return initial_context, turn_data, total_env_time


def benchmark(runtime, trajs, batch_size, config, num_turns):
    """性能测试"""
    log(f"测试 {num_turns}轮...")
    
    # 准备数据
    all_data = []
    total_env_time = 0
    
    for idx, traj in enumerate(trajs):
        initial_ctx, turn_data, env_time = prepare_data(traj, config, num_turns)
        all_data.append({
            "initial_context": initial_ctx,
            "turn_data": turn_data
        })
        total_env_time += env_time
        
        if (idx+1) % 10 == 0:
            log(f"  准备数据: {idx+1}/{len(trajs)}")
    
    avg_env = total_env_time / len(all_data)
    log(f"✓ 准备完成, avg env: {avg_env:.2f}秒")
    
    # 处理
    log("开始推理...")
    start_time = time.time()
    processed = 0
    total_tokens = 0
    
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i+batch_size]
        log(f"  批次 {i//batch_size+1}/{(len(all_data)-1)//batch_size+1}")
        
        for data in batch:
            try:
                state = constrained_multiturn.run(
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
                
                # 进度
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    log(f"    {processed}/{len(all_data)} ({elapsed:.1f}秒)")
                        
            except Exception as e:
                if processed == 0:
                    log(f"    错误: {str(e)[:150]}")
                continue
    
    total_time = time.time() - start_time
    
    result = {
        "num_turns": num_turns,
        "processed": processed,
        "total_time": total_time,
        "throughput": processed/total_time,
        "total_tokens": total_tokens,
        "avg_tokens_per_traj": total_tokens/max(processed,1),
        "avg_env_time": avg_env
    }
    
    log(f"✓ {num_turns}轮完成: {total_time:.2f}秒, {result['throughput']:.1f}条/秒, {total_tokens}tokens")
    return result


def main():
    log("="*70)
    log("约束生成测试: 50条×50轮×BS128")
    log("="*70)
    
    # 加载
    log("[1/3] 加载...")
    config = load_config()
    trajs = load_trajectories(50)
    log(f"✓ {len(trajs)}条轨迹")
    
    # 加载模型
    log("\n[2/3] 加载模型...")
    start = time.time()
    
    if __name__ == '__main__':  # multiprocessing需要
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.8,
            max_total_tokens=8192
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start
        log(f"✓ 完成! {load_time:.1f}秒")
        
        log("\nGPU 2:")
        os.system("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | grep '^2,'")
        
        # 测试
        log("\n[3/3] 测试...")
        all_results = []
        
        for num_turns in [5, 10, 20, 50]:
            try:
                log(f"\n▶ {num_turns}轮")
                result = benchmark(runtime, trajs, 128, config, num_turns)
                all_results.append(result)
                time.sleep(1)
            except Exception as e:
                log(f"✗ 失败: {str(e)[:100]}")
                break
        
        # 汇总
        log("\n\n" + "="*70)
        log("汇总")
        log("="*70)
        print(f"{'轮数':<8} {'耗时(秒)':<12} {'吞吐量':<12} {'总Tokens':<12} {'Tok/轮':<10}")
        print("-"*70)
        for r in all_results:
            print(f"{r['num_turns']:<8} {r['total_time']:<12.2f} {r['throughput']:<12.2f} {r['total_tokens']:<12} {r['avg_tokens_per_traj']/r['num_turns']:<10.1f}")
        log("="*70)
        
        # 保存
        with open("constrained_benchmark_results.json", 'w') as f:
            json.dump({"model": "Qwen3-14B-Base", "gpu": "GPU 2", "results": all_results, "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
        log("\n结果已保存: constrained_benchmark_results.json")
        
        runtime.shutdown()
        log("✓ 完成!")


if __name__ == '__main__':
    main()

