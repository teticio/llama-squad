#!/usr/bin/env bash
python train_llama_squad.py \
--model_name meta-llama/Llama-2-7b-chat-hf \
--dataset_name data/squad_v2 \
--bf16 \
--max_seq_length 4096 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--max_steps 10000 \
--merge_and_push \
--save_steps 1000 \
--learning_rate=2e-7 \
--resume_from_checkpoint results/final_checkpoints_single_turn
