import csv
import json
import logging
import os
from dataclasses import dataclass, field
from time import sleep
from typing import Optional

import openai
from datasets import load_dataset
from tqdm import tqdm
from transformers import HfArgumentParser

from utils import extract_answer

logger = logging.getLogger()

openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="gpt-3.5-turbo")
    output_csv_file: Optional[str] = field(default="results/results_openai.csv")
    debug: Optional[bool] = field(default=False)
    shuffle: Optional[bool] = field(default=True)
    seed: Optional[int] = field(default=None)
    num_samples: Optional[int] = field(default=1000)


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

logging.basicConfig(level=logging.DEBUG if script_args.debug else logging.INFO)

dataset = load_dataset("squad_v2")["validation"]
if script_args.shuffle:
    dataset = dataset.shuffle(seed=script_args.seed)
if script_args.num_samples is not None:
    dataset = dataset.select(range(script_args.num_samples))

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

    for sample in tqdm(dataset):
        prompt = f"""\
Extract from the following context the minimal span word for word that best answers the question. Think step by step and explain your reasoning. Then give the answer in JSON format as follows:
```json
{{
  "answer": ...
}}
```
If the answer is not in the context, the answer should be "?".
Context: {sample["context"]}
Question: {sample["question"]}"""

        answers = sample["answers"]["text"]
        if len(answers) == 0:
            answers = ["?"]
        logger.debug("Correct answers: %s", answers)

        for _ in range(5):
            try:
                completion = openai.ChatCompletion.create(
                    model=script_args.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
                break
            except (openai.error.Timeout, openai.error.RateLimitError):
                logger.warning("Sleeping for %s seconds", 2**_)
                sleep(2**_)
                continue

        full_response = completion.choices[0].message.content
        model_answer = extract_answer(full_response)
        logger.debug("Model answer: %s", model_answer)
        exact_match = model_answer is not None and model_answer in answers

        writer.writerow(
            [
                sample["context"],
                sample["question"],
                json.dumps(answers),
                model_answer,
                full_response,
                exact_match,
            ]
        )
        file.flush()
