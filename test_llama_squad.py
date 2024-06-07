import csv
import json
import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import transformers
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import HfArgumentParser

from create_squad_dataset import is_exact_match
from model import extract_answer, get_answer, get_model_and_tokenizer


@dataclass
class ScriptArguments:
    tokenizer_name: Optional[str] = field(
        default=None,
    )
    adapter_name: Optional[str] = field(
        default=None,
    )
    quantize: Optional[bool] = field(default=False)
    dataset: Optional[str] = field(
        default="data/squad_v2",
    )
    output_csv_file: Optional[str] = field(default="results/results.csv")
    debug: Optional[bool] = field(default=False)
    shuffle: Optional[bool] = field(default=False)
    seed: Optional[int] = field(default=None)
    num_samples: Optional[int] = field(default=None)
    skip_samples: Optional[int] = field(default=None)
    num_beams: Optional[int] = field(default=1)
    force_answer: Optional[bool] = field(default=True)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))

logger = logging.getLogger("test_llama_squad")
logger.setLevel(level=logging.DEBUG if script_args.debug else logging.INFO)
transformers.logging.set_verbosity_error()

model, tokenizer, _ = get_model_and_tokenizer(
    model_name=config.model_name,
    adapter_name=script_args.adapter_name,
    tokenizer_name=script_args.tokenizer_name,
    quantize=script_args.quantize,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


with open(script_args.output_csv_file, "w") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Prompt",
            "Correct answers",
            "Model answer",
            "Full response",
            "Exact match",
        ]
    )

    dataset = load_from_disk(config.dataset_name)["test"]
    if script_args.shuffle:
        dataset = dataset.shuffle(seed=script_args.seed)
    if script_args.num_samples is not None:
        dataset = dataset.select(range(script_args.num_samples))

    for _, messages in enumerate(tqdm(dataset["messages"])):
        if script_args.skip_samples is not None and _ < script_args.skip_samples:
            continue

        answers = extract_answer(messages[-1]["content"])
        prompt = messages[1]["content"]
        logger.debug("Prompt: %s", prompt)
        logger.debug("Correct answers: %s", answers)

        model_answer, full_response = get_answer(
            messages=messages,
            pipeline=pipeline,
            num_beams=script_args.num_beams,
            force_answer=script_args.force_answer,
        )
        logger.debug("Model answer: %s", model_answer)
        logger.debug("Response: %s", full_response)
        exact_match = is_exact_match(model_answer, answers)

        writer.writerow(
            [
                prompt,
                json.dumps(answers),
                model_answer,
                full_response,
                exact_match,
            ]
        )
        file.flush()
