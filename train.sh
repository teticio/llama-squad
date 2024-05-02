#!/usr/bin/env bash
python train_llama_squad.py \
--model_name meta-llama/Meta-Llama-3-8B-Instruct \
--dataset_name data/squad_v2 \
--bf16 \
--max_seq_length 4096 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--max_steps 65000 \
--merge_and_push \
--save_steps 1000 \
--learning_rate=1e-6 \
--lr_scheduler_type=cosine_with_restarts \
--lr_scheduler_kwargs='{"num_cycles": 3}'
