"""
快速测试 - 5条×5轮，验证流程
"""
import sglang as sgl
import os, time, json, yaml, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MODEL_PATH = "/data/minimax-dialogue/users/ruobai/cogito/base_model/Qwen3-14B-Base"
JSONL_FILE = "/data/minimax-dialogue/users/ruobai/cogito_dev/course_project_854/20250826_070310_MULTI_xiancai-80_swe_verified_train_0_5000.jsonl"

@sgl.function
def multiturn(s, user_msgs, num_turns):
    for i in range(num_turns):
        if i < len(user_msgs):
            s += sgl.user(user_msgs[i])
        # 关键：添加stop tokens防止模型无限生成对话
        s += sgl.assistant(sgl.gen(
            f"t{i}", 
            max_tokens=128, 
            temperature=0.7,
            stop=["USER:", "\nUSER:", "ASSISTANT:", "\nASSISTANT:"]  # 阻止生成新的对话轮
        ))

print("快速测试: 5条×5轮×BS128")
print("加载模型...")
runtime = sgl.Runtime(
    model_path=MODEL_PATH,
    tp_size=1,
    mem_fraction_static=0.8,
    max_total_tokens=8192
)
sgl.set_default_backend(runtime)
print("✓ 模型就绪\n")

# 加载5条轨迹
trajs = []
with open(JSONL_FILE) as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        trajs.append(json.loads(line))

print(f"测试5条轨迹...")
start = time.time()
processed = 0
total_tokens = 0

for idx, traj in enumerate(trajs):
    problem = traj.get('problem_statement', '')[:300]
    user_msgs = [f"Problem: {problem}"]
    
    steps = traj.get('trajectory_steps', [])[:5]
    for s in steps:
        obs = s.get('observation', '')[:200]
        user_msgs.append(f"{obs}...")
    
    try:
        print(f"  处理轨迹 {idx+1}/5...")
        state = multiturn.run(user_msgs=user_msgs, num_turns=5)
        processed += 1
        
        # 统计tokens
        for j in range(5):
            try:
                resp = state[f"t{j}"]
                if resp:
                    total_tokens += len(resp.split())
            except:
                pass
        
        print(f"    ✓ 完成")
    except Exception as e:
        print(f"    ✗ 错误: {e}")

elapsed = time.time() - start
print(f"\n结果:")
print(f"  耗时: {elapsed:.2f}秒")
print(f"  成功: {processed}/5")
print(f"  总tokens: {total_tokens}")
print(f"  平均: {total_tokens/max(processed,1)/5:.1f} tokens/轮")

runtime.shutdown()
print("✓ 快速测试完成!")

