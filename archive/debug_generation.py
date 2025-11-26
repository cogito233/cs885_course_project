"""
深度调试版本 - 研究为什么token生成为0
使用Qwen3-14B-Base模型
详细输出每一步的生成过程
"""

import sglang as sgl
import os
import time
import json
import yaml


# 使用GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen3-14B-Base"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"
YAML_FILE = "/data/minimax-dialogue/users/ruobai/r2e-gym-xiancai/src/r2egym/agenthub/config/r2egym/edit_non_fn_calling_.yaml"


def load_config():
    """加载YAML配置"""
    with open(YAML_FILE, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_one_trajectory():
    """只加载一条轨迹用于调试"""
    with open(JSONL_FILE, 'r') as f:
        line = f.readline()
        return json.loads(line)


@sgl.function
def simple_test_gen(s, user_msg):
    """最简单的生成测试"""
    s += sgl.user(user_msg)
    s += sgl.assistant(sgl.gen("response", max_tokens=100, temperature=0.7))


@sgl.function
def multiturn_test(s, system_msg, user_msg_1, user_msg_2):
    """简单的2轮对话测试"""
    s += sgl.system(system_msg)
    s += sgl.user(user_msg_1)
    s += sgl.assistant(sgl.gen("turn_1", max_tokens=50, temperature=0.7))
    s += sgl.user(user_msg_2)
    s += sgl.assistant(sgl.gen("turn_2", max_tokens=50, temperature=0.7))


@sgl.function
def detailed_multiturn(s, initial_context, turn_data, num_turns=3):
    """详细的多轮对话，带调试输出"""
    # 添加初始上下文
    for msg in initial_context:
        if msg["role"] == "system":
            s += sgl.system(msg["content"])
        elif msg["role"] == "user":
            s += sgl.user(msg["content"])
    
    # 逐轮生成
    for i in range(min(num_turns, len(turn_data))):
        turn = turn_data[i]
        
        # 强制生成
        s += sgl.assistant(sgl.gen(
            f"turn_{i}",
            max_tokens=256,
            temperature=0.7,  # 改用非零temperature
            stop=["\n\n\n"]  # 简单的stop条件
        ))
        
        # 添加下一个user消息
        s += sgl.user(turn["next_user_msg"])


def prepare_trajectory_data(trajectory, system_prompt, instance_prompt_template, num_turns=3):
    """准备轨迹数据"""
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
    
    steps = trajectory.get('trajectory_steps', [])[:num_turns]
    turn_data = []
    
    for step in steps:
        env_time = float(step.get('env_exec_time', 0))
        observation = step.get('observation', '')[:800]
        next_user_msg = f"{observation}\n\n[Environment execution time: {env_time:.4f}s]"
        
        turn_data.append({
            "next_user_msg": next_user_msg,
            "env_time": env_time
        })
    
    return initial_context, turn_data


def main():
    print("=" * 70)
    print("深度调试 - 研究token生成问题")
    print("=" * 70)
    print(f"模型: Qwen3-14B-Base")
    print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 加载配置
    print("[1] 加载配置和数据...")
    config = load_config()
    trajectory = load_one_trajectory()
    print(f"✓ 完成\n")
    
    # 初始化模型
    print("[2] 加载模型到GPU 3...")
    start_time = time.time()
    
    try:
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.75,
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start_time
        print(f"✓ 模型加载完成！耗时: {load_time:.2f}秒\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 测试1: 最简单的生成
    print("=" * 70)
    print("测试1: 最简单的单轮生成")
    print("=" * 70)
    
    test_msg = "Hello! Please introduce yourself in 2-3 sentences."
    print(f"User: {test_msg}")
    
    try:
        state = simple_test_gen.run(user_msg=test_msg)
        response = state.get("response", "")
        
        print(f"\n生成结果:")
        print(f"  内容: {response[:200]}...")
        print(f"  长度: {len(response)} 字符")
        print(f"  Token数(粗略): {len(response.split())} tokens")
        print(f"  类型: {type(response)}")
        print(f"\nState keys: {list(state.keys())}")
        
    except Exception as e:
        print(f"✗ 测试1失败: {e}")
    
    # 测试2: 简单的2轮对话
    print("\n" + "=" * 70)
    print("测试2: 简单的2轮对话")
    print("=" * 70)
    
    system_msg = "You are a helpful assistant."
    user_msg_1 = "What is Python?"
    user_msg_2 = "Give me a simple example."
    
    print(f"System: {system_msg}")
    print(f"User 1: {user_msg_1}")
    print(f"User 2: {user_msg_2}")
    
    try:
        state = multiturn_test.run(
            system_msg=system_msg,
            user_msg_1=user_msg_1,
            user_msg_2=user_msg_2
        )
        
        print(f"\n生成结果:")
        for key in ["turn_1", "turn_2"]:
            if key in state:
                content = state[key]
                print(f"  {key}:")
                print(f"    内容: {content[:150]}...")
                print(f"    长度: {len(content)} 字符")
                print(f"    Token数: {len(content.split())} tokens")
        
        print(f"\nState keys: {list(state.keys())}")
        
    except Exception as e:
        print(f"✗ 测试2失败: {e}")
    
    # 测试3: 使用真实轨迹数据的多轮对话
    print("\n" + "=" * 70)
    print("测试3: 真实轨迹数据 - 3轮对话")
    print("=" * 70)
    
    system_prompt = config['system_prompt']
    instance_prompt_template = config['instance_prompt']
    
    initial_ctx, turn_data = prepare_trajectory_data(
        trajectory, system_prompt, instance_prompt_template, num_turns=3
    )
    
    print(f"准备了 {len(turn_data)} 轮对话数据")
    
    try:
        state = detailed_multiturn.run(
            initial_context=initial_ctx,
            turn_data=turn_data,
            num_turns=3
        )
        
        print(f"\n生成结果:")
        total_tokens = 0
        for i in range(3):
            key = f"turn_{i}"
            if key in state:
                content = state[key]
                tokens = len(content.split())
                total_tokens += tokens
                print(f"  {key}:")
                print(f"    内容: {content[:100]}...")
                print(f"    长度: {len(content)} 字符")
                print(f"    Token数: {tokens} tokens")
            else:
                print(f"  {key}: ⚠️ 未生成")
        
        print(f"\n总Token数: {total_tokens}")
        print(f"State keys: {list(state.keys())}")
        print(f"State内容预览:")
        for k, v in state.items():
            if isinstance(v, str):
                print(f"  {k}: {len(v)} 字符, 类型={type(v)}")
        
    except Exception as e:
        print(f"✗ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 诊断信息
    print("\n" + "=" * 70)
    print("诊断信息")
    print("=" * 70)
    print(f"SGLang版本: {sgl.__version__ if hasattr(sgl, '__version__') else '未知'}")
    print(f"Runtime类型: {type(runtime)}")
    print(f"Backend: {type(sgl.get_default_backend())}")
    
    # 关闭
    print("\n清理资源...")
    runtime.shutdown()
    print("✓ 调试完成！")


if __name__ == "__main__":
    main()

