#!/usr/bin/env bash
python train_llama_squad.py \
--bf16 \
--max_seq_length 4096 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--max_steps 65000 \
--merge_and_push \
--save_steps 1000 \
--lr_scheduler_type=cosine \
--learning_rate=1e-6 \
--warmup_ratio=0.03 \
# --learning_rate=5e-7 \
# --warmup_ratio=0.15 \
