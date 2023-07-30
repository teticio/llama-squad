import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="results/final_merged_checkpoint")
    tokenizer_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    dataset: Optional[str] = field(
        default="data/squad_v2",
    )
    adapter_name: Optional[str] = field(
        default=None,
    )
    seed: Optional[int] = field(default=None)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=True,
    trust_remote_code=True,
)
if script_args.adapter_name is not None:
    model = PeftModel.from_pretrained(model, script_args.adapter_name)
    model.eval()

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=script_args.tokenizer_name,
)

if script_args.seed is not None:
    random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    torch.cuda.manual_seed(script_args.seed)

while True:
    dataset_dict = load_from_disk(script_args.dataset)
    text = random.choice(dataset_dict["test"])["text"]
    question = text[: text.find("[/INST] ") + len("[/INST] ")]
    answer = text[text.rfind("```json") :]
    print(f"Question: {question}")
    print()
    print(f"Correct answer: {answer}")
    print("=" * 80)
    print()

    response = ""
    while True:
        prompt = response
        instruction = text.find("[/INST] ")
        if instruction == -1:
            break
        instruction += len("[/INST] ")
        prompt += text[:instruction] + "</s>"
        text = text[instruction:]
        text = text[text.find("<s>") :]
        response = pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.95,
            top_k=50
        )[0]["generated_text"].strip()
        print(f"Response: {response[len(prompt) :]}")
        print("=" * 80)
        print()

    input("Press enter to continue...")
    print()
