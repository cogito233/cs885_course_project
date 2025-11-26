"""
高级多轮对话示例 - 保持完整对话上下文
展示如何在多轮对话中保持历史记录
"""

import sglang as sgl
import os

# 设置使用GPU 3
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/R2EGym-7B-Agent"


@sgl.function
def multi_turn_conversation(s, history):
    """
    多轮对话，保持完整历史
    
    Args:
        s: SGLang状态对象
        history: 历史对话列表 [{"role": "user/assistant", "content": "..."}]
    """
    for msg in history:
        if msg["role"] == "user":
            s += sgl.user(msg["content"])
        elif msg["role"] == "assistant":
            # 如果是历史消息，直接添加内容
            if msg["content"]:
                s += sgl.assistant(msg["content"])
            # 如果内容为空，需要生成
            else:
                s += sgl.assistant(sgl.gen(
                    "response",
                    max_tokens=512,
                    temperature=0.0,  # 确定性输出
                    top_p=1.0,
                ))


class ChatSession:
    """聊天会话管理器"""
    
    def __init__(self, runtime):
        self.runtime = runtime
        self.history = []
    
    def add_user_message(self, content):
        """添加用户消息"""
        self.history.append({"role": "user", "content": content})
    
    def get_assistant_response(self):
        """获取助手回复"""
        # 添加空的助手消息用于生成
        self.history.append({"role": "assistant", "content": ""})
        
        # 运行对话
        state = multi_turn_conversation.run(history=self.history)
        response = state["response"]
        
        # 更新历史记录
        self.history[-1]["content"] = response
        
        return response
    
    def chat(self, user_message):
        """完整的对话流程"""
        self.add_user_message(user_message)
        return self.get_assistant_response()
    
    def get_history(self):
        """获取对话历史"""
        return self.history.copy()
    
    def clear_history(self):
        """清空历史"""
        self.history = []


def main():
    # 初始化运行时
    print("正在初始化SGLang运行时...")
    runtime = sgl.Runtime(
        model_path=MODEL_PATH,
        tp_size=1,
        mem_fraction_static=0.8,
    )
    sgl.set_default_backend(runtime)
    print(f"模型加载完成！使用GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
    
    # 创建聊天会话
    session = ChatSession(runtime)
    
    print("=" * 70)
    print("场景：技术问答 - 多轮对话保持上下文")
    print("=" * 70)
    
    # 第一轮：询问概念
    print("\n[第1轮]")
    user_msg_1 = "什么是Transformer模型？"
    print(f"👤 用户: {user_msg_1}")
    
    response_1 = session.chat(user_msg_1)
    print(f"🤖 助手: {response_1}")
    
    # 第二轮：追问细节（依赖第一轮上下文）
    print("\n[第2轮]")
    user_msg_2 = "它和RNN有什么区别？"  # 这里的"它"指代Transformer
    print(f"👤 用户: {user_msg_2}")
    
    response_2 = session.chat(user_msg_2)
    print(f"🤖 助手: {response_2}")
    
    # 第三轮：继续深入（依赖前面的上下文）
    print("\n[第3轮]")
    user_msg_3 = "能举个实际应用的例子吗？"
    print(f"👤 用户: {user_msg_3}")
    
    response_3 = session.chat(user_msg_3)
    print(f"🤖 助手: {response_3}")
    
    # 第四轮：总结
    print("\n[第4轮]")
    user_msg_4 = "谢谢你的详细解释！"
    print(f"👤 用户: {user_msg_4}")
    
    response_4 = session.chat(user_msg_4)
    print(f"🤖 助手: {response_4}")
    
    # 显示完整对话历史
    print("\n" + "=" * 70)
    print("完整对话历史")
    print("=" * 70)
    history = session.get_history()
    for i, msg in enumerate(history, 1):
        role = "👤 用户" if msg["role"] == "user" else "🤖 助手"
        print(f"\n[消息 {i}] {role}:")
        print(msg["content"])
    
    print("\n" + "=" * 70)
    print(f"对话轮数: {len(history) // 2}")
    print("=" * 70)
    
    # 演示新会话
    print("\n\n" + "=" * 70)
    print("场景：代码生成 - 新会话")
    print("=" * 70)
    
    # 清空历史，开始新对话
    session.clear_history()
    
    print("\n[新会话 - 第1轮]")
    user_msg = "写一个Python函数来计算列表的平均值"
    print(f"👤 用户: {user_msg}")
    
    response = session.chat(user_msg)
    print(f"🤖 助手: {response}")
    
    print("\n[新会话 - 第2轮]")
    user_msg = "添加异常处理"
    print(f"👤 用户: {user_msg}")
    
    response = session.chat(user_msg)
    print(f"🤖 助手: {response}")
    
    # 关闭运行时
    print("\n" + "=" * 70)
    runtime.shutdown()
    print("所有对话完成！")


if __name__ == "__main__":
    main()

