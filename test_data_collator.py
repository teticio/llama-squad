from types import SimpleNamespace

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoTokenizer

from llama_squad import SquadDataCollator

config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))

train_dataset = load_from_disk(config.dataset_name)["train"]

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# add special <blah> token
if config.reasoning_tokens > 0:
    tokenizer.add_special_tokens(
        special_tokens_dict={"additional_special_tokens": ["<blah>"]}
    )
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
    answer_end_tokens = torch.tensor([-100])

data_collator = SquadDataCollator(
    answer_start_tokens=answer_start_tokens,
    answer_end_tokens=answer_end_tokens,
    blah_token=blah_token,
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