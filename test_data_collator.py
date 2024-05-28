from types import SimpleNamespace

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

from llama_squad import LlamaSquadDataCollator
from model import add_reasoning_tokens

config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))

train_dataset = load_from_disk(config.dataset_name)["train"]

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

reasoning_tokens = add_reasoning_tokens(
    num_reasoning_tokens=config.num_reasoning_tokens,
    multiple_reasoning_tokens=config.multiple_reasoning_tokens,
    tokenizer=tokenizer,
)

if "Llama-3" in tokenizer.name_or_path:
    answer_start_tokens = torch.tensor(
        tokenizer.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
    )

else:
    assert "Llama-2" in tokenizer.name_or_path
    answer_start_tokens = torch.tensor(
        tokenizer.encode("[/INST] ", add_special_tokens=False)
    )

data_collator = LlamaSquadDataCollator(
    answer_start_tokens=answer_start_tokens,
    answer_end_tokens=torch.tensor([-100]),
    reasoning_tokens=reasoning_tokens,
    tokenizer=tokenizer,
    mlm=False,
)

formatting_func = lambda items: tokenizer.apply_chat_template(
    items["messages"], tokenize=False
)

batch = tokenizer(formatting_func(train_dataset[0]), return_tensors="pt")
batch = [
    {col: array[i] for col, array in batch.items()}
    for i in range(len(batch[list(batch.keys())[0]]))
]
batch = data_collator(batch)

GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

index = 0
labels = batch["labels"][0]
input_ids = batch["input_ids"][0]
toggle = True
while index < labels.shape[0]:
    old_index = index
    while index < labels.shape[0] and (labels[index] == -100) == toggle:
        index += 1
    print(
        f"{YELLOW if toggle else GREEN}{tokenizer.decode(input_ids[old_index : index])}{RESET}",
        end="",
    )
    toggle = not toggle
print()
