```bash
python -m accelerate.commands.launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file ds_config.json \
  train_sft.py
```

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes deepspeed python-dotenv swanlab -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```bash
python data/load_data.py
```