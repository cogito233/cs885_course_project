"""
深度调试版本v2 - 正确处理ProgramState对象
"""

import sglang as sgl
import os
import time
import json
import yaml


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
def simple_test(s, user_msg):
    s += sgl.user(user_msg)
    s += sgl.assistant(sgl.gen("response", max_tokens=100, temperature=0.7))


@sgl.function
def multiturn_test(s, system_msg, user_msg_1, user_msg_2):
    s += sgl.system(system_msg)
    s += sgl.user(user_msg_1)
    s += sgl.assistant(sgl.gen("turn_1", max_tokens=50, temperature=0.7))
    s += sgl.user(user_msg_2)
    s += sgl.assistant(sgl.gen("turn_2", max_tokens=50, temperature=0.7))


def inspect_state(state, name="State"):
    """深度检查State对象"""
    print(f"\n{'='*60}")
    print(f"{name} 对象分析:")
    print(f"{'='*60}")
    print(f"类型: {type(state)}")
    print(f"方法和属性: {[x for x in dir(state) if not x.startswith('_')]}")
    
    # 尝试获取文本内容
    if hasattr(state, 'text'):
        print(f"\ntext(): {state.text()[:500] if callable(state.text) else state.text[:500]}")
    
    # 尝试获取变量
    if hasattr(state, 'variables'):
        print(f"\nvariables(): {state.variables()}")
    
    # 尝试其他方法
    for attr in ['get_meta_info', 'ret_dict', 'messages']:
        if hasattr(state, attr):
            try:
                val = getattr(state, attr)
                if callable(val):
                    result = val()
                    print(f"\n{attr}(): {str(result)[:500]}")
                else:
                    print(f"\n{attr}: {str(val)[:500]}")
            except Exception as e:
                print(f"\n{attr}: Error - {e}")
    
    # 尝试直接访问属性
    if hasattr(state, '__dict__'):
        print(f"\n__dict__: {list(state.__dict__.keys())}")
    
    # 尝试索引访问
    try:
        # 尝试字典式访问
        for key in ["response", "turn_1", "turn_2", "turn_0"]:
            try:
                val = state[key]
                print(f"\nstate['{key}']: {val[:200] if isinstance(val, str) else val}")
            except:
                pass
    except:
        pass


def main():
    print("=" * 70)
    print("深度调试v2 - 正确处理ProgramState")
    print("=" * 70)
    print(f"模型: Qwen3-14B-Base\n")
    
    # 加载配置
    print("[1] 加载配置...")
    config = load_config()
    trajectory = load_one_trajectory()
    print("✓ 完成\n")
    
    # 加载模型
    print("[2] 加载模型...")
    start_time = time.time()
    
    try:
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.75,
        )
        sgl.set_default_backend(runtime)
        load_time = time.time() - start_time
        print(f"✓ 完成！耗时: {load_time:.2f}秒\n")
    except Exception as e:
        print(f"✗ 失败: {e}")
        return
    
    # 测试1: 简单生成
    print("=" * 70)
    print("测试1: 简单单轮生成")
    print("=" * 70)
    
    test_msg = "Hello! Please tell me what 1+1 equals."
    print(f"User: {test_msg}\n")
    
    try:
        state = simple_test.run(user_msg=test_msg)
        inspect_state(state, "测试1结果")
        
        # 尝试提取响应
        if hasattr(state, '__getitem__'):
            try:
                response = state["response"]
                print(f"\n✓ 成功提取响应:")
                print(f"  内容: {response}")
                print(f"  长度: {len(response)} 字符")
                print(f"  Token数: {len(response.split())} tokens")
            except Exception as e:
                print(f"\n✗ 无法提取response: {e}")
        
    except Exception as e:
        print(f"✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 2轮对话
    print("\n" + "=" * 70)
    print("测试2: 2轮对话")
    print("=" * 70)
    
    try:
        state = multiturn_test.run(
            system_msg="You are helpful.",
            user_msg_1="What is 2+2?",
            user_msg_2="What is 3+3?"
        )
        inspect_state(state, "测试2结果")
        
        # 尝试提取多轮响应
        total_tokens = 0
        for key in ["turn_1", "turn_2"]:
            try:
                if hasattr(state, '__getitem__'):
                    content = state[key]
                    tokens = len(content.split())
                    total_tokens += tokens
                    print(f"\n{key}:")
                    print(f"  内容: {content}")
                    print(f"  Token数: {tokens}")
            except Exception as e:
                print(f"\n{key}: 无法提取 - {e}")
        
        print(f"\n总Token数: {total_tokens}")
        
    except Exception as e:
        print(f"✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 关闭
    print("\n清理...")
    runtime.shutdown()
    print("✓ 调试完成！")


if __name__ == "__main__":
    main()

