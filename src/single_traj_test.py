"""
单条轨迹测试 - 从最简单开始调试
GPU 2, Qwen3-14B-Base
"""

import sglang as sgl
import os
import time
import json
import yaml


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen3-14B-Base"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"


def load_config():
    with open(YAML_FILE, 'r') as f:
        return yaml.safe_load(f)


def load_one_trajectory():
    with open(JSONL_FILE, 'r') as f:
        return json.loads(f.readline())


@sgl.function
def simple_multiturn(s, user_msgs, num_turns):
    """简单的多轮对话 - 测试用"""
    for i in range(num_turns):
        if i < len(user_msgs):
            s += sgl.user(user_msgs[i])
        s += sgl.assistant(sgl.gen(f"turn_{i}", max_tokens=100, temperature=0.7))


def main():
    print("=" * 70)
    print("单条轨迹测试 - GPU 2")
    print("=" * 70)
    
    # 加载
    print("\n[1/4] 加载数据...")
    config = load_config()
    trajectory = load_one_trajectory()
    steps = trajectory.get('trajectory_steps', [])
    print(f"✓ 加载1条轨迹（原始{len(steps)}步）\n")
    
    # 加载模型
    print("[2/4] 加载模型...")
    start_time = time.time()
    
    try:
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.8,  # GPU 2空闲，可以用多一点
            max_total_tokens=8192,  # 增大上下文窗口
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start_time
        print(f"✓ 完成! {load_time:.1f}秒")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return
    
    # GPU检查
    print("\nGPU 2显存:")
    os.system("nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | grep '^2,'")
    
    # 测试1: 最简单的3轮对话
    print(f"\n[3/4] 测试1: 简单的3轮对话")
    print("-" * 70)
    
    user_msgs = [
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?"
    ]
    
    try:
        state = simple_multiturn.run(user_msgs=user_msgs, num_turns=3)
        
        print("生成结果:")
        total_tokens = 0
        for i in range(3):
            try:
                resp = state[f"turn_{i}"]
                tokens = len(resp.split())
                total_tokens += tokens
                print(f"  Turn {i}: {resp[:80]}... ({tokens} tokens)")
            except:
                print(f"  Turn {i}: 未生成")
        
        print(f"总tokens: {total_tokens}")
        
    except Exception as e:
        print(f"✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 使用真实轨迹数据
    print(f"\n[4/4] 测试2: 真实轨迹数据")
    print("-" * 70)
    
    # 提取真实数据
    problem = trajectory.get('problem_statement', '')[:500]
    
    # 构造多轮对话
    user_msgs_real = [f"Problem: {problem}"]
    
    # 添加几轮observation
    for i, step in enumerate(steps[:5]):  # 只测试5轮
        obs = step.get('observation', '')[:200]
        env_time = step.get('env_exec_time', 0)
        user_msgs_real.append(f"Observation {i+1}: {obs}...\n[Env: {env_time:.3f}s]")
    
    print(f"测试5轮对话，使用真实轨迹数据...")
    
    try:
        start = time.time()
        state = simple_multiturn.run(user_msgs=user_msgs_real, num_turns=5)
        elapsed = time.time() - start
        
        print(f"生成结果 (耗时{elapsed:.2f}秒):")
        total_tokens = 0
        total_env_time = sum(float(s.get('env_exec_time', 0)) for s in steps[:5])
        
        for i in range(5):
            try:
                resp = state[f"turn_{i}"]
                tokens = len(resp.split())
                total_tokens += tokens
                print(f"  Turn {i}: {resp[:60]}... ({tokens} tokens)")
            except Exception as ex:
                print(f"  Turn {i}: 提取失败 - {ex}")
        
        print(f"\n总tokens: {total_tokens}")
        print(f"平均: {total_tokens/5:.1f} tokens/轮")
        print(f"推理耗时: {elapsed:.2f}秒")
        print(f"Env总时间: {total_env_time:.2f}秒")
        
    except Exception as e:
        print(f"✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 关闭
    print(f"\n{'='*70}")
    runtime.shutdown()
    print("✓ 测试完成!")


if __name__ == "__main__":
    main()

