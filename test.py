import os
import json
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 加载环境变量
load_dotenv()

# 从环境变量获取模型路径（必须是绝对路径）
model_id = os.environ.get("MODEL_PATH")
if not model_id:
    raise ValueError("请设置环境变量 MODEL_PATH，指向模型的绝对路径")

# ================= 1. 加载 Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

# 准备测试数据
messages = [
    {"role": "system", "content": "你是一个专业的旅游助手。"},
    {"role": "user", "content": "从南京出发去杭州5天，有什么建议？"},
    {"role": "assistant", "content": "杭州西湖是必去景点，建议第一天先游览西湖。"}
]

# 生成推理文本和训练文本预览
inference_text = tokenizer.apply_chat_template(
    messages[:2],
    tokenize=False,
    add_generation_prompt=True
)
train_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print("--- [1] 推理模式预览 ---")
print(repr(inference_text))

print("\n--- [2] 训练数据模式预览 ---")
print(repr(train_text))

# ================= 2. Token 深度拆解 =================
print("\n" + "="*50)
print("--- [3] Token 级别深度拆解 ---")
tokens = tokenizer.encode(train_text, add_special_tokens=False)
decoded_tokens = [tokenizer.decode([t]) for t in tokens]

for i, (t_id, t_text) in enumerate(zip(tokens[:15], decoded_tokens[:15])):
    print(f"Token {i:02d} | ID: {t_id:<6} | Text: {repr(t_text)}")

print("\n--- [4] 模版源码 (Jinja2 Template) ---")
print(tokenizer.chat_template)

# ================= 3. 修复后的特殊 Token 打印 =================
print("\n--- [5] 特殊 Token 字典 ---")
special_tokens = tokenizer.special_tokens_map
all_special_list = tokenizer.all_special_tokens

print("映射字典:")
print(json.dumps(special_tokens, ensure_ascii=False, indent=4))
print("\n所有特殊 Token 列表:")
print(json.dumps(all_special_list, ensure_ascii=False, indent=4))

# ================= 4. 加载模型并进行推理 =================
print("\n" + "="*50)
print("--- [6] 加载模型中... ---")

# 自动检测设备
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"检测到可用设备: {device}")

# 加载模型
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     device_map=device,
#     local_files_only=True,
#     attn_implementation="eager",   # <--- 新增这一行：强制使用传统注意力实现，绕过 MPS 维度 Bug
# )

# # 编码输入
# model_inputs = tokenizer([inference_text], return_tensors="pt").to(device)
#
# # 执行推理
# print("--- [7] 正在生成回复 ---")
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=512,
#     do_sample=True,
#     top_p=0.9,
#     temperature=0.7,
#     eos_token_id=tokenizer.eos_token_id
# )
#
# # 仅解码新生成的 token
# input_length = model_inputs.input_ids.shape[1]
# response_ids = generated_ids[0][input_length:]
# response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
#
# print("\n--- [8] 最终推理结果 ---")
# print(response_text)