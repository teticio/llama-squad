import csv
import json
import logging
from dataclasses import dataclass, field
from time import sleep
from types import SimpleNamespace
from typing import Optional

import yaml
from datasets import load_from_disk
from openai import APITimeoutError, OpenAI, RateLimitError
from tqdm import tqdm
from transformers import HfArgumentParser

from create_squad_dataset import is_exact_match
from model import extract_answer


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt-4")
    output_csv_file: Optional[str] = field(default="results/results_openai.csv")
    debug: Optional[bool] = field(default=False)
    shuffle: Optional[bool] = field(default=True)
    seed: Optional[int] = field(default=None)
    num_samples: Optional[int] = field(default=1000)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logger = logging.getLogger("test_openai_squad")
logger.setLevel(level=logging.DEBUG if script_args.debug else logging.INFO)

config = SimpleNamespace(**yaml.safe_load(open("config.yaml")))
dataset = load_from_disk(config.dataset_name)["test"]
if script_args.shuffle:
    dataset = dataset.shuffle(seed=script_args.seed)
if script_args.num_samples is not None and script_args.num_samples < len(dataset):
    dataset = dataset.select(range(script_args.num_samples))

client = OpenAI()

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

    for sample in tqdm(dataset):
        answers = sample["answers"]
        logger.debug("Correct answers: %s", answers)

        for _ in range(5):
            try:
                completion = client.chat.completions.create(
                    model=script_args.model_name,
                    messages=sample["messages"][:-1],
                )
                break
            except (APITimeoutError, RateLimitError):
                logger.warning("Sleeping for %s seconds", 2**_)
                sleep(2**_)
                continue

        full_response = completion.choices[0].message.content
        model_answer = extract_answer(full_response)
        logger.debug("Model answer: %s", model_answer)
        logger.debug("Response: %s", full_response)
        exact_match = is_exact_match(model_answer, answers)

        writer.writerow(
            [
                sample["messages"][1],
                json.dumps(answers),
                model_answer,
                full_response,
                exact_match,
            ]
        )
        file.flush()
