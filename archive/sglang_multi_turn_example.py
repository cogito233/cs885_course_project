"""
基于SGLang的多轮对话示例
使用R2EGym-7B-Agent模型进行确定性推理
"""

import sglang as sgl
import os

# 设置使用GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 模型路径
MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/R2EGym-7B-Agent"


@sgl.function
def multi_turn_chat(s, messages):
    """
    多轮对话函数
    
    Args:
        s: SGLang状态对象
        messages: 对话消息列表，格式为 [{"role": "user/assistant", "content": "..."}]
    """
    # 遍历消息历史，构建多轮对话
    for msg in messages:
        if msg["role"] == "user":
            s += sgl.user(msg["content"])
        elif msg["role"] == "assistant":
            # 使用temperature=0实现确定性输出
            s += sgl.assistant(sgl.gen(
                "response",
                max_tokens=512,
                temperature=0.0,  # 确定性输出
                top_p=1.0,
            ))
    
    return s


def main():
    """主函数"""
    # 初始化运行时环境
    print("正在初始化SGLang运行时...")
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,  # 单GPU
        mem_fraction_static=0.8,  # 使用80%的GPU内存
    )
    
    sgl.set_default_backend(runtime)
    
    print(f"模型加载完成: {MODEL_PATH}")
    print(f"使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')}")
    print("-" * 60)
    
    # 示例1: 简单的多轮对话
    print("\n示例1: 简单多轮对话")
    print("=" * 60)
    
    messages_1 = [
        {"role": "user", "content": "你好，请介绍一下自己。"},
        {"role": "assistant", "content": ""},  # 待生成
    ]
    
    state_1 = multi_turn_chat.run(messages=messages_1)
    print(f"用户: {messages_1[0]['content']}")
    print(f"助手: {state_1['response']}")
    
    # 示例2: 多轮对话（3轮）
    print("\n\n示例2: 三轮对话")
    print("=" * 60)
    
    messages_2 = [
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": ""},  # 第一轮回复
    ]
    
    # 第一轮
    state_2 = multi_turn_chat.run(messages=messages_2)
    first_response = state_2['response']
    print(f"[第1轮]")
    print(f"用户: {messages_2[0]['content']}")
    print(f"助手: {first_response}")
    
    # 第二轮
    messages_2.append({"role": "assistant", "content": first_response})
    messages_2.append({"role": "user", "content": "能举个例子吗？"})
    messages_2.append({"role": "assistant", "content": ""})
    
    state_2 = multi_turn_chat.run(messages=messages_2)
    second_response = state_2['response']
    print(f"\n[第2轮]")
    print(f"用户: 能举个例子吗？")
    print(f"助手: {second_response}")
    
    # 第三轮
    messages_2[-1] = {"role": "assistant", "content": second_response}
    messages_2.append({"role": "user", "content": "谢谢你的解释！"})
    messages_2.append({"role": "assistant", "content": ""})
    
    state_2 = multi_turn_chat.run(messages=messages_2)
    third_response = state_2['response']
    print(f"\n[第3轮]")
    print(f"用户: 谢谢你的解释！")
    print(f"助手: {third_response}")
    
    # 示例3: 任务型对话
    print("\n\n示例3: 任务型对话")
    print("=" * 60)
    
    messages_3 = [
        {"role": "user", "content": "请帮我写一个Python函数，计算斐波那契数列的第n项。"},
        {"role": "assistant", "content": ""},
    ]
    
    state_3 = multi_turn_chat.run(messages=messages_3)
    task_response = state_3['response']
    print(f"用户: {messages_3[0]['content']}")
    print(f"助手: {task_response}")
    
    # 继续追问
    messages_3.append({"role": "assistant", "content": task_response})
    messages_3.append({"role": "user", "content": "能优化一下性能吗？"})
    messages_3.append({"role": "assistant", "content": ""})
    
    state_3 = multi_turn_chat.run(messages=messages_3)
    optimize_response = state_3['response']
    print(f"\n用户: 能优化一下性能吗？")
    print(f"助手: {optimize_response}")
    
    print("\n" + "=" * 60)
    print("所有示例完成！")
    
    # 关闭运行时
    runtime.shutdown()


if __name__ == "__main__":
    main()

