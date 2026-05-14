import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 1. 配置路径 =================
# 从 load_data.py 中导入你训练时一模一样的 System Prompt
from data.load_data import SYS_PROMPT

current_dir = os.path.dirname(os.path.abspath(__file__))

# 基础模型路径 (从环境变量获取)
model_id = os.environ.get("MODEL_PATH")
if not model_id:
    raise EnvironmentError("未设置 MODEL_PATH 环境变量！")

# LoRA Checkpoint 路径 (根据你的截图，使用最新的 checkpoint-150)
lora_path = os.path.join(current_dir, "qwen-dsl-qlora-output", "checkpoint-150")

# ================= 2. 加载与参数融合 =================
print(f"🚀 正在加载基础模型 (bfloat16): {model_id}")
# 注意：为了 merge，这里不能用 BitsAndBytesConfig 4-bit 加载
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"🧩 正在加载 LoRA 权重并与基础模型融合: {lora_path}")
# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, lora_path)
# 将 LoRA 权重合并到主模型中，并释放独立的 LoRA 显存
model = model.merge_and_unload()

print("✅ 模型加载与融合完成！\n")


# ================= 3. 定义推理函数 =================
def generate_dsl(user_text):
    """
    根据自然语言生成对应的 DSL (JSON格式)
    """
    # 构建严格匹配训练格式的消息体
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_text}
    ]

    # 使用模型的 Chat Template 拼接文本
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    # 生成预测
    # 因为是代码/JSON生成任务，关闭采样(do_sample=False)可以保证输出的确定性
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,  # 允许生成的最大长度
        do_sample=False,  # 贪婪解码，输出最确定的结果
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # 截取新生成的部分（去掉 Prompt 部分）
    input_length = model_inputs.input_ids.shape[1]
    response_ids = generated_ids[0][input_length:]

    # 解码成可读文本
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response


# ================= 4. 交互式测试 =================
if __name__ == "__main__":
    print("=" * 50)
    print("🤖 NL2DSL 推理测试已启动 (输入 'quit' 退出)")
    print("=" * 50)

    while True:
        user_input = input("\n👤 请输入旅游需求 (例如: 帮我安排3个人去北京玩4天): ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if not user_input.strip():
            continue

        print("\n⏳ 模型解析中...")
        result = generate_dsl(user_input)

        print("\n✨ 解析结果:")
        print("-" * 40)
        print(result)
        print("-" * 40)