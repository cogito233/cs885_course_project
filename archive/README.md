# SGLang 多轮对话性能测试 - 最终版本

## 项目概述

测试在不同batch_size下，使用SGLang处理多轮对话的性能。

**最终配置：**
- 模型: R2EGym-7B-Agent
- GPU: GPU 3
- 轨迹数: 50条
- 对话轮数: 50轮
- Batch Size: 128
- 强制generate每轮assistant回复

## 快速开始

```bash
# 激活环境
source .venv/bin/activate

# 运行最终测试
python final_benchmark.py
```

## 测试内容

### 1. 数据准备
- 从jsonl文件读取50条轨迹数据
- 每条轨迹原始约48步
- 使用yaml配置生成system prompt和user prompt

### 2. 多轮对话生成
- 生成50轮完整对话
- 每轮assistant强制调用gen()生成回复
- 使用temperature=0确保确定性
- 尝试约束生成以匹配原轨迹内容

### 3. 性能指标
- 总耗时
- 吞吐量（条/秒）
- 每条轨迹耗时
- 生成token统计
- 环境执行时间（从原始数据提取）

## 项目结构

```
.
├── final_benchmark.py          # 最终测试脚本 ⭐
├── batch_benchmark_force_generate.py  # 之前的版本（10轮）
├── archive/                    # 旧版本和日志
├── README.md                   # 本文件
├── QUICKSTART.md              # 快速开始指南
└── requirements.txt           # 依赖列表
```

## 关键特性

### 1. GPU配置
- 正确使用GPU 3（通过CUDA_VISIBLE_DEVICES=3）
- 验证GPU显存使用情况

### 2. 多轮对话
- 真正的多轮生成（不是简单的prefill）
- 每轮assistant都调用gen()函数
- 支持长对话（50轮）

### 3. 约束生成
- 使用stop tokens控制生成
- temperature=0确保确定性
- 尝试让生成内容接近原轨迹

### 4. 环境时间记录
- 从原始数据提取env_exec_time
- 统计每轮环境执行时间
- 计算总环境时间

## 输出结果

测试完成后会生成：
- `final_benchmark_results_50traj_50turns_bs128.json` - 详细结果
- 控制台输出完整的性能统计

## 性能指标说明

- **总耗时**: 完成所有轨迹推理的总时间
- **吞吐量**: 每秒处理的轨迹数
- **生成tokens**: 模型实际生成的token总数
- **环境时间**: 原始数据中的环境执行时间（env_exec_time）

## 环境要求

- Python 3.11+
- SGLang >= 0.5.5
- CUDA支持的GPU（H200）
- 约110GB GPU显存

## 注意事项

1. **GPU使用**: 确保GPU 3空闲可用
2. **显存**: 模型约需110GB显存
3. **时间**: 50轮×50条轨迹可能需要较长时间
4. **约束生成**: 完全匹配原轨迹内容较困难，输出可能有差异

## 故障排除

### GPU未正确使用
检查 CUDA_VISIBLE_DEVICES 设置：
```bash
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
```

### 显存不足
降低batch_size或减少轨迹数量

### 生成token为0
检查max_tokens设置和stop tokens配置

## 版本历史

- v1.0 (最终版): 50轮×50条×batch_size=128，强制generate
- v0.x (archive/): 各种测试版本

## 联系

如有问题，查看日志文件或检查archive/目录中的旧版本测试结果。
