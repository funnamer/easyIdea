```bash
accelerate launch --multi_gpu --num_processes 2 --use_deepspeed --deepspeed_config_file ds_config.json train_sft.py
```