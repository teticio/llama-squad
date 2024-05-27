#!/usr/bin/env bash
python3 train_llama_squad.py \
--bf16 \
--max_seq_length=4096 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=16 \
--max_steps=1000 \
--merge_and_push \
--save_steps=100 \
--learning_rate=0.001 \
--embedding_only
