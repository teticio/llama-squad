import csv
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import json5
import torch
import transformers
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser

from utils import extract_answer

logger = logging.getLogger()
transformers.logging.set_verbosity_error()


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    tokenizer_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    dataset: Optional[str] = field(
        default="data/squad_v2",
    )
    adapter_name: Optional[str] = field(
        default=None,
    )
    quantize: Optional[bool] = field(default=False)
    output_csv_file: Optional[str] = field(default="results/results.csv")
    debug: Optional[bool] = field(default=False)
    shuffle: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=None)
    num_samples: Optional[int] = field(default=None)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logging.basicConfig(level=logging.DEBUG if script_args.debug else logging.INFO)

if script_args.quantize:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
else:
    bnb_config = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    device_map="auto",
    use_auth_token=True,
    trust_remote_code=True,
    quantization_config=bnb_config,
)

if script_args.adapter_name is not None:
    model = PeftModel.from_pretrained(
        model, script_args.adapter_name, device_map="auto"
    )
    if not script_args.quantize:
        model = model.merge_and_unload()

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=script_args.tokenizer_name,
)


def get_answer(prompt, pipeline):
    response = ""
    while True:
        instruction = prompt.find("[/INST] ")
        if instruction == -1:
            break
        instruction += len("[/INST] ")
        current_prompt = response.strip()
        current_prompt += prompt[:instruction] + "</s>"
        logger.debug("Instruction: %s", prompt[:instruction])
        prompt = prompt[instruction:]
        prompt = prompt[prompt.find("<s>") :]
        response = pipeline(
            current_prompt,
            do_sample=False,
            num_return_sequences=1,
            max_new_tokens=512,
        )[0]["generated_text"]
        logger.debug("Response: %s", response[len(current_prompt) :].strip())

    response = response[len(current_prompt) :].strip()
    return extract_answer(response), response


with open(script_args.output_csv_file, "w") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Context",
            "Question",
            "Correct answers",
            "Model answer",
            "Full response",
            "Exact match",
        ]
    )

    dataset = load_from_disk(script_args.dataset)["test"]
    if script_args.shuffle:
        dataset = dataset.shuffle(seed=script_args.seed)
    if script_args.num_samples is not None:
        dataset = dataset.select(range(script_args.num_samples))

    for text in tqdm(dataset["text"]):
        answer_start = text.rfind("```json")
        prompt = text[:answer_start]
        answers = extract_answer(text[answer_start:])
        context = prompt[prompt.find("Context: ") + 9 : prompt.find("Question: ") - 1]
        logger.debug("Context: %s", context)
        question = prompt[prompt.find("Question: ") + 10 : prompt.find("[/INST] ")]
        question = question[: question.find("[/INST]")]
        logger.debug("Question: %s", question)
        logger.debug("Correct answers: %s", answers)
        model_answer, full_response = get_answer(prompt, pipeline)
        logger.debug("Model answer: %s", model_answer)
        exact_match = model_answer is not None and model_answer in answers

        writer.writerow(
            [
                context,
                question,
                json.dumps(answers),
                model_answer,
                full_response,
                exact_match,
            ]
        )
        file.flush()
