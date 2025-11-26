#!/bin/bash
# SGLang多轮对话示例启动脚本

# 激活虚拟环境
source .venv/bin/activate

# 显示菜单
echo "=========================================="
echo "SGLang 多轮对话示例"
echo "=========================================="
echo "请选择要运行的示例："
echo ""
echo "1. 简化版示例 (simple_chat.py) - 快速测试"
echo "2. 高级示例 (advanced_multi_turn.py) - 保持上下文 [推荐]"
echo "3. 完整示例 (sglang_multi_turn_example.py) - 多场景"
echo ""
echo -n "请输入选项 (1-3): "
read choice

case $choice in
    1)
        echo ""
        echo "运行简化版示例..."
        python simple_chat.py
        ;;
    2)
        echo ""
        echo "运行高级示例..."
        python advanced_multi_turn.py
        ;;
    3)
        echo ""
        echo "运行完整示例..."
        python sglang_multi_turn_example.py
        ;;
    *)
        echo "无效选项！"
        exit 1
        ;;
esac

