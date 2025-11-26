"""
简单测试脚本 - 确认SGLang环境能正常工作
"""

import sglang as sgl
import os
import time

# 设置使用GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/R2EGym-7B-Agent"


@sgl.function
def simple_test(s, user_msg):
    """简单测试"""
    s += sgl.user(user_msg)
    s += sgl.assistant(sgl.gen("response", max_tokens=128, temperature=0.0))


def main():
    print("=" * 70)
    print("SGLang 环境测试")
    print("=" * 70)
    
    # 初始化
    print("\n[1/3] 正在加载模型...")
    start_time = time.time()
    
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.8,
    )
    sgl.set_default_backend(runtime)
    
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成！耗时: {load_time:.2f}秒")
    
    # 测试推理
    print("\n[2/3] 测试推理...")
    test_msg = "Hello, can you introduce yourself briefly?"
    
    infer_start = time.time()
    state = simple_test.run(user_msg=test_msg)
    infer_time = time.time() - infer_start
    
    print(f"✓ 推理完成！耗时: {infer_time:.2f}秒")
    print(f"\n用户: {test_msg}")
    print(f"助手: {state['response'][:200]}...")
    
    # 关闭
    print("\n[3/3] 清理资源...")
    runtime.shutdown()
    print("✓ 完成！")
    
    print("\n" + "=" * 70)
    print("测试结果: 成功! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()

