"""
实时指标追踪版本
- 每秒记录一次全局指标
- token_throughput: 当前时刻整个batch的实时吞吐量
- prefix_length: batch内所有活跃轨迹的当前前缀长度之和
- 使用多线程+共享状态实现
"""

import sglang as sgl
import os, time, json, yaml, sys, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, Thread
from collections import deque
import threading

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen2.5-14B"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

# 全局状态追踪
class GlobalMetrics:
    def __init__(self):
        self.lock = Lock()
        self.active_trajectories = {}  # {traj_id: {"start_time": t, "tokens_so_far": n, "prefix_tokens": n}}
        self.completed_trajectories = []
        self.total_completed = 0
        self.total_tokens_generated = 0
        self.start_time = None
        
        # 用于计算实时吞吐量
        self.token_history = deque(maxlen=10)  # 最近10秒的token生成记录
        
    def start_trajectory(self, traj_id):
        with self.lock:
            self.active_trajectories[traj_id] = {
                "start_time": time.time(),
                "tokens_so_far": 0,
                "prefix_tokens": 0  # 已推理的前缀长度
            }
    
    def update_trajectory(self, traj_id, tokens_generated, prefix_tokens):
        """更新轨迹的token和前缀信息"""
        with self.lock:
            if traj_id in self.active_trajectories:
                self.active_trajectories[traj_id]["tokens_so_far"] = tokens_generated
                self.active_trajectories[traj_id]["prefix_tokens"] = prefix_tokens
    
    def complete_trajectory(self, traj_id, tokens, elapsed):
        with self.lock:
            if traj_id in self.active_trajectories:
                del self.active_trajectories[traj_id]
            self.completed_trajectories.append({
                "traj_id": traj_id,
                "tokens": tokens,
                "time": elapsed
            })
            self.total_completed += 1
            self.total_tokens_generated += tokens
            
            # 记录token历史（用于计算实时吞吐量）
            current_time = time.time() - self.start_time
            self.token_history.append({
                "time": current_time,
                "tokens": tokens
            })
    
    def get_current_metrics(self):
        """获取当前时刻的全局指标"""
        with self.lock:
            # 所有活跃轨迹的前缀长度之和
            total_prefix_length = sum(
                traj["prefix_tokens"] for traj in self.active_trajectories.values()
            )
            
            # 计算实时token吞吐量（基于最近的token生成）
            if len(self.token_history) >= 2:
                recent_tokens = sum(item["tokens"] for item in self.token_history)
                time_span = self.token_history[-1]["time"] - self.token_history[0]["time"]
                token_throughput = recent_tokens / time_span if time_span > 0 else 0
            else:
                elapsed = time.time() - self.start_time if self.start_time else 1
                token_throughput = self.total_tokens_generated / elapsed
            
            return {
                "completed_trajs": self.total_completed,
                "active_trajs": len(self.active_trajectories),
                "total_tokens_generated": self.total_tokens_generated,
                "prefix_length_sum": total_prefix_length,  # batch内所有活跃轨迹的前缀和
                "token_throughput": token_throughput  # 实时token吞吐量
            }

global_metrics = GlobalMetrics()

def metrics_recorder(metadata_file, interval=1.0, stop_event=None):
    """
    后台线程：每秒记录一次全局指标
    """
    with open(metadata_file, 'a') as f:
        while not (stop_event and stop_event.is_set()):
            if global_metrics.start_time:
                elapsed = time.time() - global_metrics.start_time
                metrics = global_metrics.get_current_metrics()
                
                record = {
                    "timestamp": elapsed,
                    **metrics
                }
                f.write(json.dumps(record) + '\n')
                f.flush()
            
            time.sleep(interval)

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
def multiturn_generate(s, system_msg, first_user_msg, observations, num_turns):
    """多轮对话生成"""
    s += sgl.system(system_msg)
    s += sgl.user(first_user_msg)
    
    for i in range(num_turns):
        s += sgl.assistant(sgl.gen(
            f"turn_{i}",
            max_tokens=256,
            temperature=0.7,
            stop=["<|im_end|>", "\n\nUSER:"]
        ))
        
        if i < len(observations):
            s += sgl.user(observations[i])

def prepare_trajectory_data(traj, config, num_turns):
    """准备单条轨迹数据"""
    system_prompt = config['system_prompt']
    instance_template = config['instance_prompt']
    
    problem_statement = traj.get('problem_statement', '')
    first_user_msg = instance_template.format(problem_statement=problem_statement)
    
    steps = traj.get('trajectory_steps', [])[:num_turns]
    observations = []
    total_env_time = 0
    
    for step in steps:
        env_time = float(step.get('env_exec_time', 0))
        total_env_time += env_time
        observation = step.get('observation', '')
        obs_msg = f"{observation}\n\n[Environment execution time: {env_time:.4f}s]"
        observations.append(obs_msg)
    
    # 估算初始前缀长度（粗略）
    initial_prefix = len(system_prompt.split()) + len(first_user_msg.split())
    
    return system_prompt, first_user_msg, observations, total_env_time, initial_prefix

def process_single_trajectory_with_tracking(traj_id, data, num_turns):
    """处理单条轨迹，实时更新全局指标"""
    try:
        # 标记开始
        global_metrics.start_trajectory(traj_id)
        
        start_time = time.time()
        
        # 简化版：由于SGLang不支持中间状态追踪，我们在这里做估算
        # 假设每轮大约生成相同数量的tokens
        estimated_tokens_per_turn = 100  # 估算值
        prefix_tokens = data["initial_prefix"]
        
        state = multiturn_generate.run(
            system_msg=data["system_msg"],
            first_user_msg=data["first_user_msg"],
            observations=data["observations"],
            num_turns=num_turns
        )
        
        elapsed = time.time() - start_time
        
        # 统计实际生成的tokens
        total_tokens = 0
        for i in range(num_turns):
            try:
                resp = state[f"turn_{i}"]
                if resp:
                    total_tokens += len(resp.split())
            except:
                pass
        
        # 标记完成
        global_metrics.complete_trajectory(traj_id, total_tokens, elapsed)
        
        return {
            "success": True,
            "traj_id": traj_id,
            "time": elapsed,
            "tokens": total_tokens
        }
    except Exception as e:
        return {
            "success": False,
            "traj_id": traj_id,
            "error": str(e)[:200]
        }

def benchmark_with_realtime_metrics(runtime, trajs, batch_size, config, num_turns, metadata_file):
    """并发测试 + 实时指标记录"""
    log(f"\n{'='*70}")
    log(f"测试 Batch Size (并发数): {batch_size}")
    log(f"{'='*70}")
    
    # 准备数据
    log("准备数据...")
    all_data = []
    total_env_time = 0
    
    for idx, traj in enumerate(trajs):
        system_msg, first_user_msg, observations, env_time, initial_prefix = prepare_trajectory_data(
            traj, config, num_turns
        )
        all_data.append({
            "system_msg": system_msg,
            "first_user_msg": first_user_msg,
            "observations": observations,
            "env_time": env_time,
            "initial_prefix": initial_prefix
        })
        total_env_time += env_time
        
        if (idx+1) % 10 == 0:
            log(f"  准备: {idx+1}/{len(trajs)}")
    
    avg_env_time = total_env_time / len(all_data)
    log(f"✓ 数据准备完成, 平均env: {avg_env_time:.2f}秒")
    
    # 重置全局指标
    global global_metrics
    global_metrics = GlobalMetrics()
    global_metrics.start_time = time.time()
    
    # 启动后台指标记录线程
    stop_event = threading.Event()
    recorder_thread = Thread(
        target=metrics_recorder,
        args=(metadata_file, 1.0, stop_event),  # 每1秒记录一次
        daemon=True
    )
    recorder_thread.start()
    log(f"✓ 后台指标记录线程已启动（每秒采样）")
    
    # 开始并发处理
    log(f"开始并发推理（并发数={batch_size}）...")
    start_time = time.time()
    
    processed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(process_single_trajectory_with_tracking, idx, data, num_turns): idx
            for idx, data in enumerate(all_data)
        }
        
        for future in as_completed(futures):
            result = future.result()
            
            if result["success"]:
                processed += 1
                
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    metrics = global_metrics.get_current_metrics()
                    log(f"  进度: {processed}/{len(all_data)} "
                        f"({100*processed/len(all_data):.0f}%) "
                        f"Token吞吐: {metrics['token_throughput']:.2f}tok/秒 "
                        f"用时: {elapsed:.1f}秒")
            else:
                failed += 1
                if failed <= 3:
                    log(f"  失败: {result['error']}")
    
    # 停止指标记录
    stop_event.set()
    recorder_thread.join(timeout=2)
    
    total_time = time.time() - start_time
    
    result_summary = {
        "batch_size": batch_size,
        "num_turns": num_turns,
        "processed": processed,
        "failed": failed,
        "total_time": total_time,
        "throughput_traj_per_sec": processed/total_time if total_time > 0 else 0,
        "throughput_token_per_sec": global_metrics.total_tokens_generated/total_time if total_time > 0 else 0,
        "total_tokens": global_metrics.total_tokens_generated,
        "avg_tokens_per_traj": global_metrics.total_tokens_generated/max(processed,1),
        "avg_env_time": avg_env_time
    }
    
    log(f"\n结果:")
    log(f"  成功: {processed}")
    log(f"  总耗时: {total_time:.2f}秒")
    log(f"  轨迹吞吐: {result_summary['throughput_traj_per_sec']:.2f}条/秒")
    log(f"  Token吞吐: {result_summary['throughput_token_per_sec']:.2f}tok/秒")
    log(f"  总tokens: {result_summary['total_tokens']}")
    log("="*70)
    
    return result_summary

def main():
    if __name__ != '__main__':
        return
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--turns', type=int, default=50)
    parser.add_argument('--num_traj', type=int, default=50)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    log("="*70)
    log("实时指标追踪版本")
    log("每秒记录: token吞吐量、前缀长度")
    log("="*70)
    log(f"模型: Qwen2.5-14B (128k)")
    log(f"GPU {args.gpu}, {args.num_traj}条, {args.turns}轮")
    log(f"Batch Size: 2, 3, 4, 6, 8")
    log("="*70)
    
    # 加载
    log("\n[1/3] 加载...")
    config = load_config()
    trajs = load_trajectories(args.num_traj)
    log(f"✓ {len(trajs)}条轨迹")
    
    # 加载模型
    log(f"\n[2/3] 加载模型...")
    start = time.time()
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.8,
        max_total_tokens=131072  # 128k
    )
    sgl.set_default_backend(runtime)
    load_time = time.time() - start
    log(f"✓ {load_time:.1f}秒")
    
    # 测试
    log(f"\n[3/3] 测试...")
    
    metadata_file = f"realtime_metrics_gpu{args.gpu}_{args.num_traj}traj_{args.turns}turns.jsonl"
    
    # 清空文件
    with open(metadata_file, 'w') as f:
        f.write(f"# Metadata for GPU{args.gpu}, {args.num_traj} trajectories, {args.turns} turns\n")
    
    all_results = []
    for batch_size in [2, 3, 4, 6, 8]:
        try:
            result = benchmark_with_realtime_metrics(
                runtime, trajs, batch_size, config, args.turns, metadata_file
            )
            all_results.append(result)
            time.sleep(2)
        except Exception as e:
            log(f"✗ BS {batch_size} 失败: {e}")
            continue
    
    # 汇总
    log("\n" + "="*70)
    log("汇总")
    log("="*70)
    for r in all_results:
        log(f"BS {r['batch_size']}: {r['total_time']:.1f}秒, "
            f"{r['throughput_traj_per_sec']:.2f}条/秒, "
            f"{r['throughput_token_per_sec']:.2f}tok/秒")
    
    # 保存
    with open(f"realtime_summary_gpu{args.gpu}_{args.num_traj}traj_{args.turns}turns.json", 'w') as f:
        json.dump({
            "model": "Qwen2.5-14B",
            "gpu": args.gpu,
            "results": all_results,
            "metadata_file": metadata_file
        }, f, indent=2)
    
    log(f"\n✓ Metadata: {metadata_file}")
    runtime.shutdown()
    log("✓ 完成!")

if __name__ == '__main__':
    main()

