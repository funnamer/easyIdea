import os
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig

# ================= 0. SwanLab 集成 =================
# 导入 SwanLab 的 HuggingFace 专属回调函数
from swanlab.integration.transformers import SwanLabCallback

# 获取当前设备的 local rank，用于多卡时只让主卡(rank 0)打印日志，避免满屏重复
local_rank = int(os.environ.get("LOCAL_RANK", 0))


def print_main(*args, **kwargs):
    if local_rank == 0:
        print(*args, **kwargs)


# ================= 1. 导入数据与 Tokenizer =================
print_main("正在执行数据加载模块...")
from data.load_data import processed_dataset, tokenizer, model_id

# ================= 2. 基础配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "qwen-dsl-qlora-output")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 3. 模型与量化加载 =================
print_main("\n⚙️ 正在配置 4-bit 量化参数...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print_main("🚀 正在以 Q4 模式加载基础模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    # device_map="auto", # 【关键】多卡 DeepSpeed 必须注释掉这一行！
    local_files_only=True
)
model.config.use_cache = False

# ================= 4. 配置 QLoRA 适配器 =================
print_main("🧩 正在预处理量化模型并注入 LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 打印参数量也只在主卡打印
if local_rank == 0:
    model.print_trainable_parameters()

# ================= 5. 配置 SwanLab =================
print_main("\n🦢 初始化 SwanLab 本地记录...")
swanlab_callback = SwanLabCallback(
    project="Qwen-DSL-Finetune",  # SwanLab 项目面板的名称
    experiment_name="qlora-deepspeed-zero1",  # 本次实验的名称
    mode="local",  # 设置为 local，数据仅保存在本地 swanlog 文件夹
    # 可以在这里记录一些额外的超参数配置
    config={
        "lora_r": 16,
        "lora_alpha": 32,
        "model": "Qwen3-4B"
    }
)

# ================= 6. 设置训练参数 =================
sft_config = SFTConfig(
    output_dir=output_dir,
    assistant_only_loss=True,
    max_seq_length=2048,               # 👈 修改：防止长文档和长代码被截断
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,                # 可保持，若 Loss 震荡再降为 1e-4
    num_train_epochs=3,
    logging_steps=5,                   # 👈 修改：每 5 步记录一次日志，曲线更平滑
    save_strategy="steps",             # 👈 修改：按步数保存
    save_steps=50,                     # 👈 修改：每 50 步存一个 checkpoint
    save_total_limit=3,
    bf16=True,
    optim="paged_adamw_32bit",
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
    disable_tqdm=False,
    remove_unused_columns=True
)

# ================= 7. 启动 SFTTrainer =================
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_dataset,
    tokenizer=tokenizer,
    args=sft_config,
    callbacks=[swanlab_callback]  # 将 SwanLab 注入 Trainer
)

print_main("\n🔥 开始多卡 DeepSpeed 微调训练...")
trainer.train()

# ================= 8. 保存最终模型 =================
# 分布式训练中，只有主进程才负责写出最终的模型权重
if local_rank == 0:
    final_model_path = os.path.join(output_dir, "final_lora")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print_main(f"🎉 训练完成，最终 QLoRA 权重已保存至: {final_model_path}")