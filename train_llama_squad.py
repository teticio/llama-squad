# based on ttps://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da

# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig
from transformers import HfArgumentParser, TrainingArguments

from llama_squad import SquadDataCollator
from model import SquadSFTTrainer, get_model_and_tokenizer


@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    max_seq_length: Optional[int] = field(default=512)
    use_4bit: Optional[bool] = field(
        default=True,
        metadata={"help": "Activate 4bit precision base model loading"},
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    lr_scheduler_kwargs: str = field(
        default="{}",
        metadata={
            "help": "Learning rate scheduler kwargs. For example: '{\"num_cycles\": 3}'"
        },
    )
    max_steps: int = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "How many steps to take before evaluating model"},
    )
    warmup_ratio: float = field(
        default=0,
        metadata={"help": "Fraction of steps to do a warmup for"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(
        default=10, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=10, metadata={"help": "Log every X updates steps."}
    )
    merge_and_push: Optional[bool] = field(
        default=False,
        metadata={"help": "Merge and push weights after training"},
    )
    output_dir: str = field(
        default="./results",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    resume_from_checkpoint: Optional[str] = field(default=None)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))


def create_and_prepare_model(args):
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
    model, tokenizer = get_model_and_tokenizer(
        model_name=config.model_name,
        quantize=args.use_4bit,
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.use_nested_quant,
    )

    # check: https://github.com/huggingface/transformers/pull/24906
    model.config.pretraining_tp = 1

    model_modules = str(model.modules)
    pattern = r"\((\w+)\): Linear"
    linear_layer_names = re.findall(pattern, model_modules)

    # target all linear layers
    names = []
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))

    peft_config = LoraConfig(
        target_modules=target_modules,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        r=script_args.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    return model, peft_config, tokenizer


training_arguments = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    optim=script_args.optim,
    save_steps=script_args.save_steps,
    logging_steps=script_args.logging_steps,
    learning_rate=script_args.learning_rate,
    fp16=script_args.fp16,
    bf16=script_args.bf16,
    max_grad_norm=script_args.max_grad_norm,
    max_steps=script_args.max_steps,
    warmup_ratio=script_args.warmup_ratio,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    lr_scheduler_kwargs=json.loads(script_args.lr_scheduler_kwargs),
    evaluation_strategy="steps",
    eval_steps=script_args.eval_steps,
)

model, peft_config, tokenizer = create_and_prepare_model(script_args)
model.config.use_cache = False
train_dataset = load_from_disk(config.dataset_name)["train"]
eval_dataset = load_from_disk(config.dataset_name)["val"]

# Fix weird overflow issue with fp16 training
tokenizer.padding_side = "right"

blah_token = tokenizer.vocab.get("<blah>")

if "Llama-3" in tokenizer.name_or_path:
    answer_start_tokens = torch.tensor(
        tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
    )
    answer_end_tokens = torch.tensor(
        tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    )

else:
    answer_start_tokens = torch.tensor(
        tokenizer.encode("[/INST] ", add_special_tokens=False)
    )
    answer_end_tokens = torch.tensor(tokenizer.encode("</s>", add_special_tokens=False))

data_collator = SquadDataCollator(
    answer_start_tokens=answer_start_tokens,
    answer_end_tokens=torch.tensor([-100]),  # Hugging Face sets the end token to -100
    blah_token=blah_token,
    tokenizer=tokenizer,
    mlm=False,
)

trainer = SquadSFTTrainer(
    answer_start_tokens=answer_start_tokens,
    answer_end_tokens=answer_end_tokens,
    reasoning_tokens=config.reasoning_tokens,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=script_args.packing,
    data_collator=data_collator,
    formatting_func=lambda items: tokenizer.apply_chat_template(
        items["messages"], tokenize=False
    ),
)

trainer.train(resume_from_checkpoint=script_args.resume_from_checkpoint)

if script_args.merge_and_push:
    output_dir = os.path.join(script_args.output_dir, "final_checkpoints")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights
    del model
    torch.cuda.empty_cache()

    from peft import AutoPeftModelForCausalLM

    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir, device_map="auto", torch_dtype=torch.bfloat16
    )
    model = model.merge_and_unload()

    output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
