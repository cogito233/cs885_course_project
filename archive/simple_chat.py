"""
简化版多轮对话示例
快速测试SGLang多轮对话功能
"""

import sglang as sgl
import os

# 设置使用GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/R2EGym-7B-Agent"


@sgl.function
def chat(s, user_message):
    """单轮对话"""
    s += sgl.user(user_message)
    s += sgl.assistant(sgl.gen("response", max_tokens=512, temperature=0.0))


def main():
    # 初始化
    print("正在加载模型...")
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.8,
    )
    sgl.set_default_backend(runtime)
    print("模型加载完成！\n")
    
    # 多轮对话
    conversation = [
        "你好，请简单介绍一下自己。",
        "什么是深度学习？",
        "能举个深度学习的应用例子吗？"
    ]
    
    for i, user_msg in enumerate(conversation, 1):
        print(f"[第{i}轮对话]")
        print(f"用户: {user_msg}")
        
        state = chat.run(user_message=user_msg)
        response = state["response"]
        
        print(f"助手: {response}")
        print("-" * 60)
    
    # 关闭
    runtime.shutdown()
    print("\n对话结束！")


if __name__ == "__main__":
    main()

