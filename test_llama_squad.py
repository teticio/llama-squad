import csv
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

logger = logging.getLogger()
logging.basicConfig(level=logging.ERROR)
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


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.quantize:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nfq",
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


def extract_answer(text):
    text = text[text.find("{") :]
    text = text[: text.find("}") + 1]
    try:
        # JSON5 is a little less picky than JSON
        answer = json5.loads(text)["answer"]
    except:
        answer = None
    return answer


def get_answer(prompt, pipeline):
    response = ""
    while True:
        instruction = prompt.find("[/INST] ")
        if instruction == -1:
            break
        instruction += len("[/INST] ")
        current_prompt = response.strip()
        current_prompt += prompt[:instruction] + "</s>"
        prompt = prompt[instruction:]
        prompt = prompt[prompt.find("<s>") :]
        response = pipeline(
            current_prompt,
            do_sample=False,
            num_return_sequences=1,
            max_new_tokens=512,
        )[0]["generated_text"]

    response = response[len(current_prompt) :].strip()
    return extract_answer(response), response


with open(script_args.output_csv_file, "w") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Context",
            "Question",
            "Correct answer",
            "Model answer",
            "Full response",
        ]
    )

    dataset_dict = load_from_disk(script_args.dataset)
    for text in tqdm(dataset_dict["test"]["text"]):
        prompt = text[: text.find("[/INST] ") + len("[/INST] ")]
        answer = extract_answer(text[text.rfind("```json") :])
        context = prompt[prompt.find("Context: ") + 9 : prompt.find("Question: ") - 1]
        logger.info(f"Context: {context}")
        question = prompt[prompt.find("Question: ") + 10 : -9]
        logger.info(f"Question: {question}")
        logger.info(f"Correct answer: {answer}")
        model_answer, full_response = get_answer(prompt, pipeline)
        logger.info(f"Model answer: {model_answer}")
        logger.info(f"Full response: {full_response}")

        writer.writerow([context, question, answer, model_answer, full_response])
        file.flush()
