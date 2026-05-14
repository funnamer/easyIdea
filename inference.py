import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm  # 用于显示批量处理的进度条
import re
from data.load_data import SYS_PROMPT

# ================= 初始化与环境变量 =================
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_id = os.environ.get("MODEL_PATH")
if not model_id:
    raise EnvironmentError("未设置 MODEL_PATH 环境变量！")
lora_path = os.path.join(current_dir, "qwen-dsl-qlora-output", "checkpoint-150")


# ================= 1. 模型加载 =================
def load_model():
    print(f"🚀 正在加载基础模型 (bfloat16): {model_id}")
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
    model = PeftModel.from_pretrained(base_model, lora_path)
    # 融合并卸载lora结构，这么设计是因为，可能有好几个lora用来做不同的任务，方便切换
    model = model.merge_and_unload()

    print("✅ 模型加载与融合完成！\n")
    return tokenizer, model


# ================= 2. 统一的 Message 组装 =================
def build_messages(text):
    """统一组装 System Prompt 和用户输入"""
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": text}
    ]



# ================= 3. 推理生成 (核心) =================
def generate_dsl(user_text, tokenizer, model):
    """
    根据自然语言生成对应的 DSL。
    完全保留模型的原始字符串输出，不做任何截断或 JSON 解析。
    """
    messages = build_messages(user_text)

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text_input)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_length = model_inputs.input_ids.shape[1]
    response_ids = generated_ids[0][input_length:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return response


# ================= 4. 批量处理函数 =================
def batch_inference(input_file,tokenizer, model):
    """
    读取 JSON 数据集，进行批量推理，并将【原始输入 + 完整输出】保存到文件中。
    """
    print(f"\n📂 正在加载测试集: {input_file}")
    # 这里假设你的数据集是一个包含字典的列表 JSON 文件
    dataset = load_dataset(path="json", data_files=input_file, split="train")

    results = []
    print(f"🚀 开始批量推理 (共 {len(dataset)} 条数据)...")

    # 使用 tqdm 包装一下，终端会显示很漂亮的进度条
    for item in tqdm(dataset, desc="推理进度"):
        uid=item.get("uid")
        start_city=item.get("start_city")
        target_city=item.get('target_city')
        days=item.get("days")
        people_number=item.get("people_number")
        nature_language = item.get("nature_language")
        if not nature_language:
            print(f"⚠️  无效的输入: {nature_language}")
            continue

        # 调用推理函数，获取完整原始输出
        raw_output = generate_dsl(nature_language, tokenizer, model)

        # 将结果存入列表
        result_item = {
            "uid": uid,
            "start_city": start_city,
            "target_city": target_city,
            "days": days,
            "people_number": people_number,
            "nature_language": nature_language,
            "hard_logic_py": raw_output
        }
        # 逐条保存为单独的JSON文件，文件名为uid.json
        res_dir = os.path.join(current_dir, "res")
        os.makedirs(res_dir, exist_ok=True)
        file_path = os.path.join(res_dir, f"{uid}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result_item, f, ensure_ascii=False, indent=2)
    print(f"✅ 批量推理完成！")


# ================= 5. 主程序入口 =================
if __name__ == "__main__":
    print("=" * 50)
    print("⚙️ 正在初始化推理环境...")
    global_tokenizer, global_model = load_model()

    # 默认配置
    input_file = "data/test"

    print(f"\n📂 开始批量推理: {input_file}")
    batch_inference(input_file, global_tokenizer, global_model)
    print("✅ 全部完成！")