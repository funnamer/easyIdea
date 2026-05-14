```bash
python -m accelerate.commands.launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --use_deepspeed \
  --deepspeed_config_file ds_config.json \
  train_sft.py
```
