# 快速开始指南

## 一键运行

```bash
./run.sh
```

选择选项2（高级示例）即可体验完整的多轮对话功能。

## 三种示例对比

| 示例文件 | 难度 | 特点 | 适用场景 |
|---------|------|------|---------|
| `simple_chat.py` | ⭐ | 最简单，3轮独立对话 | 快速测试环境 |
| `advanced_multi_turn.py` | ⭐⭐⭐ | 保持上下文，支持代词引用 | 实际应用开发 |
| `sglang_multi_turn_example.py` | ⭐⭐ | 多场景展示 | 学习不同用例 |

## 核心特性

### 1. 确定性输出

所有示例都使用 `temperature=0.0`，确保：
- 每次运行结果一致
- 适合需要可重复结果的场景

### 2. GPU配置

默认使用GPU 3：
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
```

如需修改，编辑对应Python文件的这一行。

### 3. 多轮对话实现

**方法A: 简单方式（simple_chat.py）**
- 每轮对话独立
- 不保持历史

**方法B: 高级方式（advanced_multi_turn.py）** ⭐推荐
- 使用ChatSession类管理状态
- 自动维护对话历史
- 支持上下文相关的追问

## 常见问题

### Q1: 如何修改最大生成长度？

修改 `max_tokens` 参数：
```python
sgl.gen("response", max_tokens=512, temperature=0.0)
```

### Q2: 如何启用非确定性输出？

修改 `temperature` 参数：
```python
sgl.gen("response", max_tokens=512, temperature=0.7)  # 0.7为常用值
```

### Q3: 内存不足怎么办？

调整 `mem_fraction_static` 参数：
```python
runtime = sgl.Runtime(
    model_path=MODEL_PATH,
    tp_size=1,
    mem_fraction_static=0.6,  # 降低到60%
)
```

### Q4: 如何使用多个GPU？

修改 `CUDA_VISIBLE_DEVICES` 和 `tp_size`：
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 使用GPU 2和3
runtime = sgl.Runtime(
    model_path=MODEL_PATH,
    tp_size=2,  # 张量并行度=GPU数量
)
```

## 输出示例

运行 `advanced_multi_turn.py` 后，你会看到类似这样的输出：

```
正在初始化SGLang运行时...
模型加载完成！使用GPU: 3

======================================================================
场景：技术问答 - 多轮对话保持上下文
======================================================================

[第1轮]
👤 用户: 什么是Transformer模型？
🤖 助手: [模型生成的回答]

[第2轮]
👤 用户: 它和RNN有什么区别？
🤖 助手: [模型基于上下文生成的回答]

...
```

## 下一步

- 修改对话内容，测试不同场景
- 调整参数，优化性能
- 参考代码，集成到你的项目中

## 技术支持

如有问题，请检查：
1. 模型路径是否正确
2. GPU是否可用且有足够内存
3. sglang是否正确安装

