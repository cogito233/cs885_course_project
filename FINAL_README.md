# 测试不同Batch Size（并发数）的影响 - 使用指南

## 🎯 测试目标

**探索问题**: 在不同batch_size（并发数）下，跑50条5轮对话轨迹分别要用多久？

**Batch Size含义**: **同时运行的轨迹数量**（并发数）
- 完成一条轨迹后，立即启动下一条
- 类似线程池，保持batch_size个任务并发运行

**测试配置**:
- 轨迹数: 50条
- 对话轮数: 5轮
- Batch Size (并发数): 1, 2, 4, 8, 16, 32, 64, 128, 256
- 内容: **完整**（不缩短problem_statement和observation）
- 生成方式: **强制generate**（每轮assistant都调用gen()）
- 并发方式: ThreadPoolExecutor

## 🚀 后台运行命令

```bash
cd /data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854
./RUN_COMMAND.sh
```

或者直接：
```bash
cd /data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854
source .venv/bin/activate
nohup python concurrent_benchmark.py > CONCURRENT_BENCHMARK.log 2>&1 &
```

## 📊 监控进度

```bash
# 实时查看（Ctrl+C退出）
tail -f CONCURRENT_BENCHMARK.log

# 每5秒自动刷新
watch -n 5 'tail -20 CONCURRENT_BENCHMARK.log'
```

## 🔍 测试配置详情

### 模型配置
- **模型**: Qwen3-14B-Base (14B参数)
- **GPU**: GPU 2
- **显存配置**: mem_fraction=0.8 (约115GB)
- **最大tokens**: 16384

### 数据配置
- **数据源**: `20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl`
- **System prompt**: 从yaml配置加载（完整）
- **Problem statement**: **完整保留**（不缩短）
- **Observations**: **完整保留**（不缩短）
- **Environment time**: 从原始数据提取

### 生成配置
- **每轮调用**: `sgl.gen()` 强制生成
- **Max tokens**: 256/轮
- **Temperature**: 0.7
- **Stop tokens**: `["<|im_end|>", "\n\nUSER:"]` 防止无限生成

### 并发配置 ⭐
- **并发方式**: ThreadPoolExecutor
- **Batch Size = 并发数**: 同时运行的轨迹数量
- **动态调度**: 完成一条就补充一条（类似线程池）

## 📁 输出文件

- `CONCURRENT_BENCHMARK.log` - 详细运行日志
- `concurrent_batch_size_comparison_5turns.json` - 测试结果（JSON格式）

## ⏱️ 预计时间

- 模型加载: ~40秒
- Batch Size 1: 可能较慢
- Batch Size 2-64: 逐渐加快
- Batch Size 128-256: 最快
- **总计**: 约10-20分钟（取决于batch_size效率）

## 📈 输出示例

测试完成后会输出类似下表：

```
BS     总耗时(秒)    吞吐量          每条(秒)      总Tokens    Tok/轮    Env(秒)
----------------------------------------------------------------------
1      XX.XX        XX.XX          XX.XX        XXXXX      XX.X      XX.XX
2      XX.XX        XX.XX          XX.XX        XXXXX      XX.X      XX.XX
4      XX.XX        XX.XX          XX.XX        XXXXX      XX.X      XX.XX
...
```

## ✅ 已修复的问题

1. **模型无限生成** ✓ 
   - 添加stop tokens: `["<|im_end|>", "\n\nUSER:"]`
   
2. **Token统计为0** ✓
   - 正确使用: `state[f"turn_{i}"]`
   
3. **GPU使用错误** ✓
   - 从GPU 3改为GPU 2（空闲）
   
4. **缩短了内容** ✓
   - 现在保持完整的problem_statement和observation
   
5. **测试目标混淆** ✓
   - 现在测试不同batch_size（不是不同轮数）

## 🎯 下一步

1. **启动测试**: 运行上面的命令
2. **监控进度**: 用`tail -f TEST_BATCH_SIZES.log`
3. **等待完成**: 约10-20分钟
4. **查看结果**: `cat batch_size_comparison_5turns.json`
5. **告诉我完成**: 我会帮你分析结果

## 📝 如果需要测试50轮

确认5轮测试成功后，可以修改`test_batch_sizes.py`中的`num_turns = 5`改为`num_turns = 50`，重新运行。

现在可以启动测试了！🚀
